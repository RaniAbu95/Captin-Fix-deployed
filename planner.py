import json
import os
import pandas as pd
import requests
from pydantic import BaseModel
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from config import ANTHROPIC_API_KEY



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


def generate_testplan(url: str, links: List[str], num_tests: int) -> TestPlan:
    page_html = extract_full_html(url)

    max_negative = round(num_tests / 3)

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=8192,
    )

    template = ChatPromptTemplate.from_template("""
You are an expert QA automation engineer. Your job is to generate a structured, logical, and fully verifiable test plan based ONLY on the HTML provided below.

HTML OF THE TARGET PAGE:
{page_html}

---

OUTPUT FORMAT
Return only valid JSON. Each test case must have:
  "id"       — unique string, e.g. "TC001"
  "suite"    — one of: Smoke | Navigation | Interaction
  "steps"    — ordered list of 3–6 steps (see STEP FORMAT below)
  "expected" — a single, concrete, machine-checkable outcome (see EXPECTED FORMAT below)
  "priority" — High | Medium | Low
  "negative" — true only for intentional error/failure tests (see NEGATIVE TESTS)

Generate EXACTLY {num_tests} test cases.

---

STEP FORMAT — each step must be a complete sentence that states:
  1. The ACTION (navigate, click, type, hover, submit)
  2. The TARGET — the exact visible text, label, or HTML attribute that identifies the element
  3. The OUTCOME to observe immediately after (optional but preferred)

Good step examples:
  - "Navigate to the website homepage"
  - "Click the link with visible text 'Sign In'"
  - "Type 'test@example.com' into the email input field"
  - "Click the submit button labelled 'Search'"
  - "Verify the page URL contains the path from the href attribute of the clicked link"
  - "Verify the heading 'Results' is visible on the page"

Bad step examples (too vague — NEVER write these):
  - "Click the menu"                ← which menu? which element?
  - "Check the page"                ← check what exactly?
  - "Navigate somewhere"            ← where?
  - "Verify it works"               ← verify what specifically?

---

EXPECTED FORMAT — must be ONE of these concrete, checkable forms:
  - "The page URL contains '<fragment from href>'"
  - "The element with text '<text>' is visible on the page"
  - "The page title contains '<keyword>'"
  - "A validation error message is displayed"
  - "The form field '<name>' shows an error state"
  - "The dropdown/menu with items [...] is visible"
  DO NOT write vague expectations like "the page loads correctly" or "the user sees the result".

ELEMENT AVAILABILITY — strictly enforced:
  - Only test elements that are ALWAYS present regardless of: login state, geographic region, server IP, or A/B test.
  - NEVER test elements that require the user to be logged in (e.g. account menus, rewards widgets, personalized content).
  - NEVER test elements that are region-specific or only shown to certain IP ranges (e.g. Microsoft Rewards, regional banners, country-specific promotions).
  - NEVER test elements that are shown only on first visit or behind feature flags.
  - NEVER test third-party or ad-injected elements — these include Google Publisher Tag slots (e.g. gpt-*, GPT slots), Outbrain (ob_iframe, ob_holder), Taboola, or any element injected by external ad/analytics scripts. Their presence depends on external services and they will not load in automated test environments.
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

HTML RULES (strictly enforced):
- Base EVERY test case ONLY on elements that actually appear in the provided HTML.
- Before writing a step, confirm the element (button, link, input, heading) exists in the HTML.
- Do NOT use prior knowledge about the website — ignore anything you know from training.
- If an element does not appear in the HTML, do not write a step about it.
- NEVER assume a link opens in a new tab unless the HTML explicitly shows target="_blank".

---

NAVIGATION RULES:
- Every test case that clicks a navigation link MUST end with a URL verification step.
- The URL fragment MUST come from the href attribute in the HTML — never from the link's visible text.
  Example: HTML shows <a href="/about-us">About</a> → step says "Verify the URL contains '/about-us'"
- If the href is an external domain, verify that domain: <a href="https://shop.example.com/"> → "Verify the URL contains 'shop.example.com'"

---

UNIQUENESS RULES (strictly enforced):
- Every test case must test a DIFFERENT feature, interaction, or user flow.
- No two test cases may duplicate steps or expected results, even if worded differently.
- If you run out of distinct features, generate fewer test cases rather than duplicating.

---

SUITE ASSIGNMENT:
- Smoke     — critical page-load and core element visibility checks (does the page open? are key elements present?)
- Navigation — clicking links and menu items, verifying URL changes or new pages
- Interaction — forms, search inputs, buttons that trigger actions, hover menus, dropdowns

---

NEGATIVE TESTS:
- A negative test deliberately triggers an error response: validation error, 404, server error, empty required field.
- Only write negative tests when the HTML contains inputs, forms, or search fields that can realistically fail.
- If no such functionality exists, set "negative": false on every case.
- Maximum negative tests allowed: {max_negative}

---

Return only valid JSON. No markdown, no explanation, no code fences.
    """)

    prompt = template.format_messages(page_html=page_html, num_tests=num_tests,
                                      max_negative=max_negative)
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

# -----------------------------
# Runner using existing user parameters
# -----------------------------
# def run_planner(target: str, num_tests: int = 5, depth: int = 1, email: str = "", pm: str = "jira"):
#
#     process_target_data(target)
#     # Validate URL
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
#         }
#         resp = requests.get(target, headers=headers, timeout=10)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"Site not accessible: {e}")
#         return
#
#     links = sample_links(target, num_tests=num_tests, depth=depth)
#     plan = generate_testplan(target, links)
#     save_testplan(plan)
#
#     print(f"Test Plan generated successfully for {target}!")
#     if email:
#         print(f"Results will be sent to: {email}")
#     print(f"Project Management tool selected: {pm}")


import os
import requests
from urllib.parse import urlparse, unquote

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
    run_planner('https://www.google.com/')