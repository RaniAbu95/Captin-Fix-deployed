import json
import os
import pandas as pd
import requests
from pydantic import BaseModel
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from config import ANTHROPIC_API_KEY
from urllib.parse import urlparse, unquote


# -----------------------------
# Data Schemas
# -----------------------------
class TestCase(BaseModel):
    id: str
    suite: str
    steps: List[str]
    expected: str
    priority: str
    negative: bool = False

class TestPlan(BaseModel):
    website: str
    suites: List[str]
    cases: List[TestCase]




from executor import extract_full_html


def _parse_cases_from_json(parsed) -> tuple:
    """Return (cases, suites_list) from any Claude JSON format."""
    cases = []

    # Claude sometimes returns a bare array instead of a wrapped object.
    if isinstance(parsed, list):
        for c in parsed:
            if isinstance(c, dict) and "id" in c:
                c.setdefault("negative", False)
                cases.append(TestCase(**c))
        return cases, list(dict.fromkeys(c.suite for c in cases))

    raw = parsed.get("testPlan", parsed)

    # Format 1: flat {"cases": [...]}
    flat_cases = raw.get("cases") if isinstance(raw, dict) else None
    if flat_cases and isinstance(flat_cases, list) and all(isinstance(c, dict) and "id" in c for c in flat_cases):
        for c in flat_cases:
            cases.append(TestCase(**c))
        return cases, list(dict.fromkeys(c.suite for c in cases))

    suites_raw = raw.get("suites", {}) if isinstance(raw, dict) else {}

    # Format 2: {"suites": [{"name": "Smoke", "testCases": [...]}]}
    if isinstance(suites_raw, list):
        suites_list = []
        for suite_obj in suites_raw:
            if isinstance(suite_obj, dict):
                suite_name = suite_obj.get("name", "")
                suites_list.append(suite_name)
                for c in suite_obj.get("testCases", suite_obj.get("cases", [])):
                    if "suite" not in c:
                        c["suite"] = suite_name
                    cases.append(TestCase(**c))
            elif isinstance(suite_obj, str):
                suites_list.append(suite_obj)
        return cases, suites_list

    # Format 3: {"suites": {"Smoke": [...cases], ...}}
    if isinstance(suites_raw, dict):
        suites_list = []
        for suite_name, suite_cases in suites_raw.items():
            suites_list.append(suite_name)
            for c in suite_cases:
                cases.append(TestCase(**c))
        return cases, suites_list

    return cases, list(dict.fromkeys(c.suite for c in cases))


def _strip_json(text: str) -> str:
    import re as _re
    text = _re.sub(r"^```[a-z]*\s*", "", text, flags=_re.IGNORECASE).strip()
    text = _re.sub(r"```$", "", text).strip()
    m = _re.search(r"(\{.*\}|\[.*\])", text, _re.DOTALL)
    return m.group(0) if m else text


_FORM_LINK_KEYWORDS = ("contact", "search", "subscribe", "newsletter", "apply", "register", "login", "careers", "signup", "enquir")

def _fetch_linked_form_htmls(homepage_html: str, base_url: str) -> dict:
    """Find nav links that likely lead to form pages and fetch their HTML."""
    import re
    from urllib.parse import urljoin, urlparse
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', homepage_html)
    base = urlparse(base_url)
    seen, results = set(), {}
    for href in hrefs:
        if not any(k in href.lower() for k in _FORM_LINK_KEYWORDS):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc != base.netloc:
            continue
        if full in seen or len(results) >= 3:
            continue
        seen.add(full)
        try:
            html = extract_full_html(full)
            results[href] = html
            print(f"[planner] fetched linked form page: {full} ({len(html)} chars)")
        except Exception as e:
            print(f"[planner] could not fetch {full}: {e}")
    return results


def generate_testplan(url: str, links: List[str], num_tests: int) -> TestPlan:
    page_html = extract_full_html(url)
    linked_form_htmls = _fetch_linked_form_htmls(page_html, url)

    max_negative = round(num_tests / 3)

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=8192,
    )

    template = ChatPromptTemplate.from_template("""
You are an expert QA automation engineer. Your job is to generate a structured, logical, and fully verifiable test plan based ONLY on the HTML provided below.

CAPTCHA / BOT-DETECTION GUARD — check first:
If the HTML contains any of: ShieldSquare, h-captcha, hcaptcha, recaptcha, "Are you for real", "robot-captcha", "captcha-mid", or any bot-detection page — the site is BLOCKING automated access. In this case return an empty cases array: {{"website": "", "cases": []}} — do NOT generate tests for the captcha page itself.

HTML OF THE TARGET PAGE:
{page_html}

{linked_pages_section}
---

OUTPUT FORMAT
Return only valid JSON. Each test case must have:
  "id"       — unique string, e.g. "TC001"
  "suite"    — one of: Smoke | Navigation | Forms
  "steps"    — ordered list of 3–8 steps (see STEP FORMAT below). Navigation tests should have 5–7 steps (navigate → click → verify URL → verify element on destination → optional interaction). Forms tests should have 4–6 steps (navigate → find form → type → submit → verify outcome).
  "expected" — a single, concrete, machine-checkable outcome (see EXPECTED FORMAT below)
  "priority" — High | Medium | Low
  "negative" — true only for intentional error/failure tests (see NEGATIVE TESTS)

Generate EXACTLY {num_tests} test cases.

---

STEP FORMAT — each step must be a complete sentence that states:
  1. The ACTION (navigate, click, type, hover, submit)
  2. The TARGET — the exact id, href, or visible attribute that identifies the element
  3. The OUTCOME to observe immediately after (optional but preferred)

CLICK REQUIREMENT — strictly enforced:
  - Every Navigation test MUST contain at least one step that says "Click the link with href '<href>'" or "Click the anchor with id '<id>'". Scan the HTML for <a href="..."> elements and pick one to click.
  - Every Navigation test MUST have at least one step AFTER the URL verification that interacts with or verifies something on the destination page — this is REQUIRED, not optional.
  - Every Forms test MUST contain: (1) a step that types a value into an input field, AND (2) a step that clicks the submit/search button to submit the form, AND (3) a step that verifies the result. All three steps are required — a Forms test without a submit and outcome verification is invalid.
  - A test that only verifies element presence without any click or type action is NOT a valid Navigation or Forms test. It belongs to Smoke — and there is only 1 Smoke slot.
  - If no form exists on the homepage, check navigation links: if any href points to /contact/, /search/, /apply/, /careers/, /register/ — generate a Forms test that navigates there first. Only skip Forms entirely if the HTML has zero forms AND zero navigation links suggesting a form page.

Good step examples:
  - "Navigate to the website homepage at '<url>'"
  - "Click the anchor element with href '/categories.aspx' inside the element with id 'headerMenu'"
  - "Click the button with id 'search-btn'"
  - "Type 'מנהל' into the input element with id 'searchKeyword'"
  - "Verify the page URL contains '/categories.aspx'"
  - "Verify the element with id 'search-results' is visible on the page"
  - "Click the link with href '/about-us/' and verify the page URL contains '/about-us/'"
  - "Click the secondary link with href '/team/' visible on the /about-us/ page"
  - "Type 'test query' into the search input and click the submit button"

Bad step examples (NEVER write these):
  - "Verify the element with id 'app' is present"                     ← no click, belongs to Smoke
  - "Verify the logo is visible"                                        ← no click, belongs to Smoke
  - "Click the menu"                                                    ← which element? must name id or href
  - "Verify it works"                                                   ← too vague
  - "Click on any available navigation link"                            ← FORBIDDEN: must name a specific href or id
  - "Click any link in the navigation"                                  ← FORBIDDEN: must name a specific href or id
  - "Click on one of the navigation links"                              ← FORBIDDEN: must name a specific href or id
  - "Verify the main heading element h1 or h2 is visible"               ← FORBIDDEN: too generic, every page has h1/h2. Must name a specific id, class, or text fragment.
  - "Verify the element with class 'container' is visible"              ← FORBIDDEN: 'container' exists on every page. Must name a content-specific id or class.
  - "Verify the element with class 'main' is visible"                   ← FORBIDDEN: too generic. Same rule applies.
  - "Verify the element with class 'wrapper' is visible"                ← FORBIDDEN: too generic.
  Every navigation step MUST target ONE specific element identified by its exact href or id. Steps using "any", "available", or "one of" are never acceptable.
  Every post-navigation verification MUST name a SPECIFIC, CONTENT-MEANINGFUL element — never a structural wrapper like container, main, wrapper, section, or a generic tag like h1/h2 without an id or class.

---

EXPECTED FORMAT — must be ONE of these concrete, checkable forms:
  - "The page URL contains '<fragment from href>'"
  - "The element with id '<id>' is visible on the page"
  - "The element with aria-label '<label>' is visible on the page" — ONLY if that exact aria-label value appears verbatim in the HTML snippet provided. NEVER infer or guess an aria-label.
  - "A validation error message is displayed"
  - "The form field '<name>' shows an error state"
  - "The dropdown/menu with items [...] is visible"
  NEVER use page title as an expected result — titles are locale-dependent and unreliable.
  DO NOT write vague expectations like "the page loads correctly" or "the user sees the result".

ELEMENT AVAILABILITY — strictly enforced:
  - Only test elements that are ALWAYS present regardless of: login state, geographic region, server IP, or A/B test.
  - NEVER test elements that require the user to be logged in (e.g. account menus, rewards widgets, personalized content).
  - NEVER test elements that are region-specific or only shown to certain IP ranges (e.g. Microsoft Rewards, regional banners, country-specific promotions).
  - NEVER test elements that are shown only on first visit or behind feature flags.
  - NEVER test third-party or ad-injected elements — these include Google Publisher Tag slots (e.g. gpt-*, GPT slots), Outbrain (ob_iframe, ob_holder), Taboola, or any element injected by external ad/analytics scripts. Their presence depends on external services and they will not load in automated test environments.
  - NEVER test elements that have style="display:none" or style="visibility:hidden" in the HTML — they are invisible to the user and cannot be reliably clicked in automated tests.
  - NEVER test mobile-only UI elements: hamburger buttons, side drawers, mobile navigation menus, bottom navigation bars, or any element whose id/class contains 'mobile', 'hamburger', 'drawer', 'side-toolbar', 'offcanvas', or 'nav-toggle'. The browser runs at 1440×900 desktop resolution — mobile elements are hidden or irrelevant.
  - GOOD elements to test: search inputs, desktop navigation links, logo, footer, main content area, forms, headings, images that are part of the page HTML.
  - BAD elements to test: rewards widgets, login-state UI, regional offers, personalized recommendations, ad slots, third-party iframes injected after page load, mobile menus, hamburger icons, side drawers.

TEST VALUE — every test case must check something that matters to a real user:
  - GOOD tests: "Can the user navigate to the About page?", "Does the search form accept input and submit?", "Is the main navigation visible and working?", "Does the contact form show a validation error on empty submit?"
  - BAD tests: "Is a div present in the DOM?", "Does an ad container element exist?", "Is a span visible?" — these have no user value.
  - Each test must represent a complete user action with a meaningful, observable outcome.

LOCALE SAFETY — strictly enforced for ALL text assertions:
  - The browser renders the site in its native locale. For Israeli sites (.co.il, .org.il, .net.il, etc.) ALL visible text — titles, headings, buttons, links — will be in HEBREW.
  - NEVER assert any English word that is a translation of Hebrew content. The Hebrew page will never contain it.
  - NEVER assert the visible text of any UI element — it will be in Hebrew.
  - NEVER use an English transliteration of a Hebrew brand name in a title check (e.g. NEVER "drushim" — the title says "דרושים". NEVER "ynet" if the title says "ynet" only if it literally appears that way in the HTML title tag).
  - For page title assertions: look at the actual <title> tag value in the HTML. Use ONLY a fragment that appears exactly as-is in that tag — a TLD abbreviation like "IL", a Latin brand name that appears in the title as-is, or a number. If the title is entirely in Hebrew with no Latin fragment, DO NOT assert the title at all — verify a URL fragment instead.
  - ALWAYS verify element PRESENCE or VISIBILITY using locale-stable attributes: element ID, aria-label, data-testid, CSS class, or href.
  - For URL assertions: URL paths are always in Latin characters regardless of locale — these are safe to assert.

  GOOD expected results (locale-safe):
    - "The element with id 'headerMenu' is visible on the page"
    - "The element with aria-label 'search' is visible"
    - "The page URL contains '/companies.aspx'"
    - "The page title contains 'IL'" ← only if 'IL' literally appears in the <title> tag

  BAD expected results (locale-broken):
    - "The element with text 'Sign in' is visible"         ← Hebrew site, text is in Hebrew
    - "The button labelled 'Search' is visible"            ← Hebrew site, text is in Hebrew
    - "The page title contains 'drushim'"                  ← title is "דרושים IL", not "drushim"
    - "The page title contains 'Jobs'"                     ← title is in Hebrew, no English word

---

NO GUESSING — CRITICAL, STRICTLY ENFORCED:
This is the single most important rule. Every locator, attribute value, class name, id, href, input name, placeholder, or aria-label you write in a step MUST appear verbatim in the HTML provided below. If it is not in the HTML, do not write it.

Specific things you must NEVER guess:
- CSS class names for elements on pages you have not seen (e.g. never '.section-head', '.intro-main', '.job-results' unless they appear in the provided HTML)
- Input field names, ids, or placeholders on linked pages (e.g. never "input[name='your-name']" for a contact form on /contact/ — you have not seen that page's HTML)
- Nav element class attributes (e.g. NEVER "//nav[@class='nav-2025']" — you cannot know the nav's class; use "//nav//a[contains(@href,'...')]" instead)
- href patterns for links on pages you have not seen (e.g. never "/job/123", "/position/", "/apply/" for a careers sub-page you have not seen)
- aria-label values that are not literally in the HTML
- Element IDs that are not literally in the HTML

When writing steps, only describe what you can see in the provided HTML. If a step requires knowledge of a page you have not seen, stop at the boundary you CAN see — navigate there and verify a heading or URL, but do not interact with content you cannot verify from the HTML.

---

HTML RULES (strictly enforced):
- Base EVERY test case ONLY on elements that actually appear in the provided HTML.
- Before writing a step, confirm the element (button, link, input, heading) exists in the HTML.
- Do NOT use prior knowledge about the website — ignore anything you know from training.
- If an element does not appear in the HTML, do not write a step about it.
- NEVER assume a link opens in a new tab unless the HTML explicitly shows target="_blank".
- ONLY use <a href="..."> elements found in the <body> as clickable navigation links. NEVER use <link> tags from the <head> — those are stylesheet/font/resource references (e.g. Google Fonts, CSS files) and cannot be clicked by a user.
- A valid clickable link has a visible text label and an href pointing to a page path (e.g. /about, /categories.aspx) or domain. hrefs pointing to .css, .js, fonts.googleapis.com, cdn URLs, or external resources are NOT clickable links.
- IMPORTANT — DIRECT REACHABILITY: Before writing a "Click the link with href '...'" step, ask yourself: is this link directly visible and clickable on the page without any prior interaction? If the link is inside a dropdown, mega-menu, footer accordion, or any container that requires a prior click to expand/reveal it, you MUST include the intermediate step (e.g. "Click the menu button with id '...' to open the dropdown") BEFORE the click step. Never write a click step for a link that is not directly reachable in the initial page state.
- IMPORTANT — ELEMENTS INSIDE THE NAV DRAWER: Elements such as country pickers, language selectors, or any input/link whose id or class contains 'country', 'language', 'locale', or 'picker' are typically rendered inside the navigation drawer and are NOT directly accessible. You MUST include a step to click the element with aria-label 'Open main navigation' BEFORE any step that interacts with these elements.

---

LINKED PAGE CONTENT — strictly enforced:
- The planner has ONLY the homepage HTML. You do NOT have the HTML for any linked page (/careers/, /contact/, /blog/, /shop/, etc.).
- NEVER write steps that interact with elements on a linked page whose HTML you have not seen. This includes: job listing links, apply buttons, blog post links, product cards, search result items, article titles, or any dynamically-populated list items on destination pages.
- You do NOT know what href patterns, class names, or DOM structure those pages have. Job application links may go to external domains (e.g. personio.de, greenhouse.io, lever.co). Product links may use opaque IDs. Blog post hrefs may be slugs. Never guess these patterns.
- SAFE steps on linked pages (you may write these): verify the URL contains a known path stem, verify h1/h2/nav/footer is visible, type into a search/contact form whose input is visible in the homepage HTML or whose linked page is clearly a form (contact, search, subscribe).
- UNSAFE steps on linked pages (NEVER write these): "Click the first job listing", "Click the Apply button for an open position", "Click a product card", "Click a blog post link", "Click the link with href containing '/job'", "Click the link with class 'job-btn'" — you cannot know any of this from the homepage HTML alone.
- If a test needs to go deeper than one level (e.g. homepage → careers → individual job), that is only valid if the intermediate page's HTML was provided. Since it was not, stop the test at the intermediate page level.

---

NAVIGATION RULES:
- Every Navigation test MUST have a URL verification step followed by at least one more step on the destination page. A Navigation test that ends at URL verification is INVALID.
- The URL fragment MUST come from the href attribute in the HTML — never from the link's visible text.
  Example: HTML shows <a href="/about-us">About</a> → step says "Verify the URL contains '/about-us'"
- If the href is an external domain, verify that domain: <a href="https://shop.example.com/"> → "Verify the URL contains 'shop.example.com'"
- After verifying the URL, ALWAYS add: "Verify the page heading (h1 or h2) is visible on the destination page" — use the generic heading tag, NOT a CSS class you invented. You only have the homepage HTML; you do NOT know the destination page's class names. NEVER write a step like "Verify the element with class 'section-head' is visible" for a page you have not seen — that class may not exist. Safe post-navigation assertions: h1, h2, footer, nav — these exist on every page.
- If the destination page likely has a form or interactive element, add a step that interacts with it (type into a search box, click a secondary link, expand an accordion).

---

UNIQUENESS RULES (strictly enforced):
- Every test case must test a DIFFERENT feature, interaction, or user flow.
- No two test cases may duplicate steps or expected results, even if worded differently.
- If you run out of distinct features, generate fewer test cases rather than duplicating.

---

SUITE ASSIGNMENT:
- Smoke      — EXACTLY 1 test only: navigate to homepage, verify the page loads and key elements are present. No clicks. Verify at least 2 distinct elements (heading, logo, nav link, etc.).
- Navigation — click an <a href="..."> link in the page body, verify the URL changes to the expected destination. MUST include a click step.
  REQUIRED DEPTH: every Navigation test MUST have at least 2 steps on the destination page AFTER the URL verification:
    Step N:   "Verify the page URL contains '/target-path'"
    Step N+1: "Verify the [heading/banner/form/landmark element with id '...'] is visible on the destination page"
    Step N+2: (optional) interact with something on the destination page — click a secondary link, fill a field, or hover a menu item
  A Navigation test that ends at URL verification only is INVALID.
- Forms      — fill and SUBMIT a form. MUST include: (1) navigate to the page containing the form (could be the homepage OR a linked page like /contact/), (2) type a value into an input field, (3) click the submit/search button, (4) verify the result (success message, URL change, or validation error). A Forms test that stops before verifying the outcome is INVALID.
  FORMS ON LINKED PAGES (very important): if the homepage HTML does NOT have a visible form but it has navigation links to pages that likely contain forms (e.g. href contains 'contact', 'search', 'subscribe', 'apply', 'register', 'login', 'careers', 'newsletter'), generate a Forms test that:
    Step 1: Navigate to the homepage
    Step 2: Click the link with href '/contact/' (or whatever the href is)
    Step 3: Verify the page URL contains '/contact/'
    Step 4+: For EVERY visible input and textarea on that page (check the HTML OF LINKED FORM PAGES section above), generate one step that fills it with a realistic value. Use the exact name= or id= attribute from the HTML — do NOT skip any field and do NOT invent field names.
    Last step before submit: Click the submit button
    Final step: Verify the result (confirmation message, URL change, or validation error)
  SEARCH BOX PRIORITY: if the page has a search input (type="search", type="text" inside a form, or an input with id/name/class containing "search", "query", "q"), ALWAYS generate at least one Forms test that types a realistic search query and submits it.
  FORMS DETECTION — scan the HTML carefully for ALL of these: <input>, <textarea>, <form>, <button type="submit">, or any element with id/class/href containing "search", "contact", "subscribe", "newsletter", "query", "email", "apply", "register", "careers". If ANY of these exist in the homepage HTML OR as a navigation href, generate a Forms test.
  NEGATIVE FORMS: generate at least one Forms test where you submit the form with EMPTY required fields and verify a validation error appears — this is a highly valuable negative test.
  Only generate Forms tests if the HTML or navigation links suggest a form exists — never completely invent one.

TEST DIVERSITY — strictly enforced:
- No two Navigation tests may click the same link or verify the same URL fragment.
- No two tests may share the same expected result.
- Navigation tests must target DIFFERENT pages/sections of the site.
- NEVER generate more than 4 Navigation tests regardless of how many nav links exist — pick only the 3-4 most important ones.
- At least 2 tests per run must go DEEPER than a single click: e.g. navigate to a page then verify a specific element on that page, interact with a form, select a dropdown, or verify a content section is populated.

NAVIGATION PATTERN VARIETY — strictly enforced:
Each Navigation test MUST use a DIFFERENT post-navigation pattern. Assign one pattern per test, never repeat the same pattern across two tests. If you have 4 Navigation tests, use 4 different patterns.

  Pattern A — Scroll + lazy content reveal:
    Step 1: Navigate to the homepage
    Step 2: Click the nav link with href '<href>'
    Step 3: Verify the page URL contains '<path>'
    Step 4: Verify the page heading (h1) is visible on the destination page
    Step 5: Verify a content section heading (h2) is visible
    Use when: the destination page has content sections below the fold (blog, careers, services, products pages)
    NOTE: Do NOT add a scroll step before verifying h2. EC.visibility_of_element_located works on elements below the fold — Selenium does not require the element to be in the viewport, only that it is not hidden (display:none / visibility:hidden). A scroll step is ONLY needed when the HTML shows the target element is inside a container with CSS animation classes (animate-in, fade-in, slide-in, scroll-reveal) — in that case the element is literally hidden until an IntersectionObserver fires.

  Pattern B — Secondary link click within page content:
    Step 1: Navigate to the homepage
    Step 2: Click the nav link with href '<href>'
    Step 3: Verify the page URL contains '<path>'
    Step 4: Verify the page heading (h1) is visible
    Step 5: Scroll down to reveal page content
    Step 6: Click a secondary link visible inside the main content area (NOT in the nav or footer) — pick a sub-page link whose href is a deeper path under the current page
    Step 7: Verify the page URL changed to the sub-page path
    Use when: the destination page contains links to sub-pages (about-us with team/csr links, blog with article links, solutions with product sub-pages)

  Pattern C — In-page anchor navigation:
    Step 1: Navigate to the homepage
    Step 2: Click the nav link with href '<href>'
    Step 3: Verify the page URL contains '<path>'
    Step 4: Verify the page heading (h1) is visible
    Step 5: Scroll down to reveal anchor links on the page
    Step 6: Click an anchor link whose href starts with '#' (section jump link, table-of-contents entry, tab)
    Step 7: Verify the URL now contains '#' (anchor fragment appended)
    Use when: the destination page uses section anchors, tabs, or a table of contents (long-form pages, FAQs, feature pages)

  Pattern D — Hover dropdown → sub-page → footer verification:
    Step 1: Navigate to the homepage
    Step 2: Hover over the parent nav item to reveal the dropdown menu
    Step 3: Click the sub-page link with href '<sub-href>' inside the dropdown
    Step 4: Verify the page URL contains '<sub-path>'
    Step 5: Verify the page heading (h1 or h2) is visible
    Step 6: Verify the footer element is visible
    Use when: the nav has a multi-level dropdown with sub-pages (About Us > CSR, Products > Feature X)
    NOTE: Do NOT add a scroll step before verifying the footer. EC.visibility_of_element_located finds footer elements without scrolling — Selenium only requires the element is not display:none, not that it is in the viewport. A scroll step adds dead time with no assertion value.

  FORBIDDEN pattern (never use): navigate → verify URL → stop. Every Navigation test must continue past the URL check.

SUITE DISTRIBUTION — strictly enforced:
- EXACTLY 1 Smoke test.
- MAXIMUM 4 Navigation tests — even if the site has 20 nav links, pick only the 3-4 most important. Do not fill remaining slots with more navigation tests.
- MINIMUM 2 Forms tests if the site has ANY form, input, or search box in the HTML. If the site has multiple forms (search + contact + newsletter), generate one test per form type.
- If you cannot find 2 Forms-worthy elements, use Navigation tests for the remaining slots — but still cap Navigation at 4.
- Target distribution for {num_tests} total tests: 1 Smoke, up to 4 Navigation, and the remaining slots as Forms tests. Example: for 10 tests → 1 Smoke + 4 Navigation + 5 Forms. For 5 tests → 1 Smoke + 2 Navigation + 2 Forms. Always total exactly {num_tests}.

---

NEGATIVE TESTS:
- A negative test deliberately triggers an error response: validation error, 404, server error, empty required field.
- Only write negative tests when the HTML contains inputs, forms, or search fields that can realistically fail.
- If no such functionality exists, set "negative": false on every case.
- Maximum negative tests allowed: {max_negative}

---

Return only valid JSON. No markdown, no explanation, no code fences.
    """)

    if linked_form_htmls:
        linked_pages_section = "HTML OF LINKED FORM PAGES (use the exact field names and attributes you see here when generating Forms test steps — do NOT guess field names):\n"
        for href, html in linked_form_htmls.items():
            linked_pages_section += f"\nPage: {href}\n{html}\n"
    else:
        linked_pages_section = ""

    prompt = template.format_messages(page_html=page_html, num_tests=num_tests,
                                      max_negative=max_negative,
                                      linked_pages_section=linked_pages_section)
    import sys as _sys, time as _t
    print(f"[anthropic] generate_testplan.invoke at {_t.time()} (planner.py:153)", flush=True, file=_sys.stderr)
    response = llm.invoke(prompt)
    plan_json = response.content.strip()
    stop_reason = (response.response_metadata or {}).get("stop_reason", "unknown")
    print(f"LLM stop_reason: {stop_reason}")
    print(f"LLM Output ({len(plan_json)} chars):", plan_json[:2000])

    plan_json = _strip_json(plan_json)

    try:
        parsed = json.loads(plan_json)
    except json.JSONDecodeError as e:
        preview = plan_json[:2000] if plan_json else "(empty)"
        print("❌ Failed JSON parsing. Full LLM output:", plan_json)
        raise ValueError(f"LLM did not return valid JSON (stop_reason={stop_reason}). Raw output preview: {preview}") from e

    cases, suites_list = _parse_cases_from_json(parsed)

    # Fill missing cases (total count short)
    if len(cases) < num_tests:
        missing = num_tests - len(cases)
        existing_ids = [c.id for c in cases]
        fill_template = ChatPromptTemplate.from_template("""
            You are an expert QA engineer.
            Here is the FULL HTML of the target website:
            {page_html}

            A test plan already has these test case IDs: {existing_ids}
            Generate exactly {missing} MORE test cases (do NOT repeat those IDs).
            Suites to use: Smoke, Navigation, Forms.
            Each test case must include: id, suite, steps, expected, priority, negative (boolean).
            Only use elements actually present in the HTML.
            Return only a valid JSON array of test case objects.
        """)
        import sys as _sys, time as _t
        print(f"[anthropic] generate_testplan.fill_invoke at {_t.time()} (planner.py:186)", flush=True, file=_sys.stderr)
        fill_resp = llm.invoke(fill_template.format_messages(
            page_html=page_html, existing_ids=existing_ids, missing=missing))
        try:
            extra = json.loads(_strip_json(fill_resp.content.strip()))
            if isinstance(extra, list):
                for c in extra:
                    c.setdefault("negative", False)
                    cases.append(TestCase(**c))
        except Exception:
            pass

    # Enforce exact total count
    cases = cases[:num_tests]

    if not suites_list:
        suites_list = list(dict.fromkeys(c.suite for c in cases))
    suites_list = list(dict.fromkeys(c.suite for c in cases))

    return TestPlan(website=url, suites=suites_list, cases=cases)


# -----------------------------
# Save Outputs
# -----------------------------

def save_testplan(plan: TestPlan, base_path: str = "./output"):
    # JSON
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/plan.json", "w", encoding="utf-8") as f:
        json.dump(plan.dict(), f, indent=2, ensure_ascii=False)

    # Excel
    data = [
        {
            "ID": c.id,
            "Suite": c.suite,
            "Steps": " | ".join(c.steps),
            "Expected": c.expected,
            "Priority": c.priority
        } for c in plan.cases
    ]
    df = pd.DataFrame(data)
    df.to_excel(f"{base_path}/Plan.xlsx", index=False)




def run_planner(target: str, num_tests: int, depth: int, email: str = "", pm: str = ""):
    parsed = urlparse(target)
    if parsed.scheme == "file":
        local_path = unquote(parsed.path)
        if os.name == "nt" and local_path.startswith("/"):
            local_path = local_path[1:]
        if not os.path.exists(local_path):
            raise ValueError(f"Local file not accessible: {local_path}")
    elif parsed.scheme in ("http", "https"):
        # Pre-flight reachability check. WAF-protected sites (Cloudflare etc.)
        # often reject Python's requests library on TLS fingerprint even with a
        # browser User-Agent, so a non-2xx response is NOT a fatal signal — the
        # real headless Chrome that runs the tests usually still works.
        browser_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            requests.get(target, headers=browser_headers, timeout=10)
            # Don't call raise_for_status(): a 4xx/5xx here usually means the
            # WAF blocked requests but the site itself is up. Let Selenium try.
        except requests.exceptions.ConnectionError as e:
            raise ValueError(f"Site not reachable (network error): {e}") from e
        except requests.exceptions.Timeout as e:
            raise ValueError(f"Site not reachable (timeout): {e}") from e
        except Exception as e:
            # Any other exception is logged but not fatal — proceed and let
            # the real browser-driven flow surface a clearer error if needed.
            print(f"[planner] pre-flight check raised {type(e).__name__}: {e} — proceeding anyway")
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    plan = generate_testplan(target, [], num_tests)
    save_testplan(plan)
    print(f"Test Plan generated for {target}")



if __name__ == "__main__":
    run_planner('https://www.igaming.com/',10,1)