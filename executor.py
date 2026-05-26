import os
import json
import threading
from selenium.webdriver.chrome.options import Options
from langchain_anthropic import ChatAnthropic
import time
from config import ANTHROPIC_API_KEY
from langchain_core.messages import SystemMessage, HumanMessage

PLAN_FILE = "./output/plan.json"
OUTPUT_DIR = "tests"
RESULTS_JSON = "Results.json"
SCREENSHOT_DIR = "./screen/screenshots"

_html_cache = {}
HEADLESS = os.environ.get("HEADLESS", "true").lower() != "false"

# Only one Chrome subprocess at a time — prevents OOM on low-memory hosts
_chrome_lock = threading.Semaphore(1)

def _chrome_options():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--no-first-run")
    opts.add_argument("--disable-background-networking")
    opts.add_argument("--disable-sync")
    opts.add_argument("--disable-default-apps")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36")
    opts.add_argument("--window-size=1440,900")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.page_load_strategy = 'none'
    return opts


# Marked as a cache breakpoint so Anthropic reuses it on every call after the first.
def _make_main_block(website: str) -> str:
    """Return the if __name__ block tailored to the website's locale."""
    israeli = _is_israeli_site(website)
    lang_arg = '        "--lang=he-IL",' if israeli else ""
    lang_pref = (
        '    opts.add_experimental_option("prefs", {"intl.accept_languages": "he,he-IL,en-US,en"})'
        if israeli else
        '    opts.add_experimental_option("prefs", {"intl.accept_languages": "en-US,en"})'
    )
    nav_languages = (
        "['he-IL', 'he', 'en-US', 'en']" if israeli else "['en-US', 'en']"
    )
    geolocation = (
        '''    driver.execute_cdp_cmd("Emulation.setGeolocationOverride", {
        "latitude": 31.7683, "longitude": 35.2137, "accuracy": 100
    })'''
        if israeli else ""
    )
    return '''
if __name__ == "__main__":
    import os
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time

    opts = Options()
    opts.page_load_strategy = "none"
    if os.environ.get("HEADLESS", "false").lower() != "false":
        opts.add_argument("--headless=new")
    for arg in [
        "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        "--disable-extensions", "--no-first-run", "--disable-background-networking",
        "--disable-sync", "--disable-default-apps",
        "--disable-blink-features=AutomationControlled",
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "--window-size=1440,900",
%(lang_arg)s
    ]:
        opts.add_argument(arg)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
%(lang_pref)s

    driver = webdriver.Chrome(options=opts)
    driver.set_script_timeout(8)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        if (!window.chrome) window.chrome = {};
        if (!window.chrome.runtime) window.chrome.runtime = {};
        Object.defineProperty(navigator, 'plugins', {get: () => [
            {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format'},
            {name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:''},
            {name:'Native Client', filename:'internal-nacl-plugin', description:''}
        ]});
        Object.defineProperty(navigator, 'mimeTypes', {get: () => [
            {type:'application/pdf', suffixes:'pdf', description:''}
        ]});
        Object.defineProperty(navigator, 'languages', {get: () => %(nav_languages)s});
        Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
        const _origPerms = navigator.permissions.query.bind(navigator.permissions);
        navigator.permissions.query = (p) =>
            p.name === 'notifications'
                ? Promise.resolve({state: 'default', onchange: null})
                : _origPerms(p);
    """})
%(geolocation)s

    def _dismiss_banner():
        # Native JS click fires the site's own accept handler reliably (a Selenium
        # click can land before the handler is bound). Then strip any leftover
        # Bootstrap modal/backdrop so it cannot intercept later clicks.
        # Returns True only when something was actually handled, so the polling
        # loop keeps watching for a banner/modal that is injected late.
        try:
            return bool(driver.execute_script(r"""
                var handled = false;
                var accSel = ['#cookieConsentModal .btn-accept', '.btn-accept',
                              '#CybotCookiebotDialogBodyButtonClose', '.cky-btn-accept',
                              '[id*="accept-all"]', '[id*="acceptAll"]'];
                var clicked = false;
                for (var i = 0; i < accSel.length; i++) {
                    var el = document.querySelector(accSel[i]);
                    if (el && el.offsetParent !== null) { el.click(); clicked = true; break; }
                }
                if (!clicked) {
                    var labels = ['allow all', 'accept all', 'accept cookies', 'קבל הכל', 'אישור'];
                    var nodes = document.querySelectorAll('button, a, span');
                    for (var j = 0; j < nodes.length; j++) {
                        var t = (nodes[j].innerText || '').trim().toLowerCase();
                        if (labels.indexOf(t) >= 0 && nodes[j].offsetParent !== null) {
                            nodes[j].click(); clicked = true; break;
                        }
                    }
                }
                if (clicked) handled = true;
                // Safety: strip any leftover backdrop/modal so it cannot intercept
                // a click. This does NOT count as "handled" — only an actual accept
                // click grants consent (permanent). Force-removing without granting
                // consent lets the site re-open the modal, so the loop must keep
                // polling until the accept button is clicked.
                document.querySelectorAll('.modal-backdrop').forEach(function(e) { e.remove(); });
                document.querySelectorAll('.modal.show, .modal[style*="display: block"]').forEach(function(m) {
                    m.classList.remove('show'); m.style.display = 'none';
                    m.setAttribute('aria-hidden', 'true');
                });
                document.body.classList.remove('modal-open');
                document.body.style.overflow = ''; document.body.style.paddingRight = '';
                return handled;
            """))
        except Exception:
            return False

    _orig_get = driver.get
    def _patched_get(url):
        _orig_get(url)
        import time as _t
        # Wait up to 4s for page to become interactive, exit early when ready
        _deadline_load = _t.time() + 4
        while _t.time() < _deadline_load:
            try:
                if driver.execute_script("return document.readyState") in ("interactive", "complete"):
                    break
            except Exception:
                pass
            _t.sleep(0.3)
        _t.sleep(1)  # brief settle for banner scripts to inject
        # Only enter the retry loop if a banner element is actually in the DOM
        _has_banner = False
        try:
            _has_banner = bool(driver.execute_script(
                'return document.querySelector(arguments[0])',
                '[id*="cookie"],[class*="cookie"],[id*="consent"],[id*="Cybot"],[class*="cky"],[id*="banner"]'))
        except Exception:
            pass
        if _has_banner:
            _deadline = _t.time() + 12
            while _t.time() < _deadline:
                if _dismiss_banner():
                    _t.sleep(0.5)  # let the close/fade-out settle
                    break
                _t.sleep(0.5)
    driver.get = _patched_get

    try:
        run(driver)
        print("RESULT: Pass")
    finally:
        driver.quit()
''' % dict(lang_arg=lang_arg, lang_pref=lang_pref, nav_languages=nav_languages, geolocation=geolocation)

SYSTEM_PROMPT_TEMPLATE = """You are a senior QA automation engineer writing a Selenium test for a website deployed on a cloud server (Render.com). Pages may be slow to load — use generous timeouts accordingly.

═══════════════════════════════════════
OUTPUT FORMAT (follow exactly)
═══════════════════════════════════════
1. Module-level imports only: selenium, WebDriverWait, By, EC, time. Add ActionChains or Keys only if the test needs them.
2. Wrap ALL test logic in a single function: def run(driver):
3. `driver` is provided by the runner — NEVER call webdriver.Chrome(), driver.quit(), or driver.close().
4. Call driver.get("{website}") exactly ONCE, as the FIRST line inside run(driver).
5. Do NOT add an if __name__ == "__main__": block — it is appended automatically.
Required structure (follow exactly):
   def run(driver):
       try:
           driver.get("{website}")
           # all steps here
       except Exception as e:
           driver.save_screenshot(f"error_{{int(time.time())}}.png")
           raise
       finally:
           pass

═══════════════════════════════════════
STEP FIDELITY — CRITICAL RULE
═══════════════════════════════════════
- IMPORTANT: The steps below are the pre-generated test plan. You MUST implement them EXACTLY as written — word for word. Do not paraphrase, substitute, or reinterpret a step.
- If a step says "element with id 'app'" you MUST use By.ID, "app". If it says "aria-label 'X'" you MUST use that exact aria-label. Never swap one locator type for another.
- Do not skip, merge, reorder, or invent steps. Every step in the plan must appear as a distinct code block.
- If you cannot find a locator in the HTML, use exactly what the step specifies anyway — do NOT silently replace it with a different attribute or selector.

═══════════════════════════════════════
TIMEOUTS — DEPLOYED SITE (important)
═══════════════════════════════════════
- Default timeout for ALL WebDriverWait calls: 10 seconds. Never use less than 10.
- The site runs on a cloud server that may have slow cold starts. 10s gives it room to respond.
- Use ONE wait per page-load check — do NOT chain multiple waits for the same page.

═══════════════════════════════════════
PAGE LOAD CHECK
═══════════════════════════════════════
- After driver.get(), ALWAYS verify the page is ready with this exact JS readyState check — it works on every site regardless of DOM structure:
    WebDriverWait(driver, 5).until(
        lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
    )
- This is the ONLY guaranteed page-load check. NEVER use EC.presence_of_element_located on "header", "nav", "main", or any structural element as the primary page-load signal — these elements may not exist or may be injected late by JavaScript.
- After the readyState check passes, you may add ONE additional wait for a specific element that the test actually needs (e.g. a nav link before clicking it). Do not add redundant waits.

═══════════════════════════════════════
LOCATORS — how to find elements
═══════════════════════════════════════
Priority order (most reliable → least reliable): ID → Name → XPath → CSS Selector
Always use the highest-priority locator that works for the element. Only fall back to a lower-priority strategy when a higher one is not available in the HTML.
IMPORTANT: if a stable, semantic id is present — ALWAYS use By.ID. Do NOT use XPath or CSS when By.ID is available.

BY.ID — always prefer By.ID when a stable, semantic id is present. Only use it when the id is a human-readable, semantic name (e.g. "search-btn", "headerMenu", "flashBell"). NEVER use By.ID for ids that look randomly generated — strings of random alphanumeric characters like "r1w2KWYLVsyGg" or "HJH3YbK84ikMe" are build-time dynamic ids that change on every deployment and will break the test.
    el = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.ID, "search-btn")))
    el.click()

BY.NAME — for form inputs with a name attribute:
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "q")))

BY.TAG_NAME — ONLY for standard HTML tags (div, nav, footer, header, main, section, h1, h2, etc.). NEVER use By.TAG_NAME for custom elements with hyphens (e.g. "footer-2025", "nav-bar", "app-root") — Selenium's tag name strategy does not support hyphenated custom elements. Use By.CSS_SELECTOR instead:
    CORRECT for custom element:   (By.CSS_SELECTOR, "footer-2025")
    INCORRECT:                    (By.TAG_NAME, "footer-2025")

BY.CSS_SELECTOR — use when no stable id or name exists but a stable CSS selector is available (e.g. data-testid, aria-label, class, tag+attribute combos). Prefer attribute selectors over class names when classes may be dynamic:
    (By.CSS_SELECTOR, "[data-testid='search-input']")
    (By.CSS_SELECTOR, "input[name='q']")
    (By.CSS_SELECTOR, "#headerMenu a")
- ARIA-LABEL RULE: when the selector uses aria-label, always wrap the string with single quotes outside and double quotes inside the attribute value:
    CORRECT:   (By.CSS_SELECTOR, '[aria-label="Open main navigation"]')
    INCORRECT: (By.CSS_SELECTOR, "[aria-label='Open main navigation']")
- ARIA-LABEL EXISTENCE RULE: ONLY use aria-label if the element has an aria-label attribute in the HTML. If the element has visible text but NO aria-label attribute (e.g. <span>Ask AI</span>), use XPath text matching instead:
    CORRECT:   (By.XPATH, "//span[normalize-space()='Ask AI']")
    INCORRECT: (By.CSS_SELECTOR, '[aria-label="Ask AI"]')  ← aria-label not in HTML, will always fail

XPATH — use only when ID, Name, and CSS Selector are not suitable (e.g. locating by visible text, contains(@href), or complex DOM traversal). Always scope to a container when possible:
    (By.XPATH, "//div[@id='headerMenu']//a[contains(@href, '/categories.aspx')]")
    (By.XPATH, "//input[@type='submit']")
    (By.XPATH, "//*[@data-testid='search-input']")

LINKS WITH VISIBLE TEXT:
- Use By.PARTIAL_LINK_TEXT with the exact visible text from the HTML.
- For nav links match by @href directly: (By.XPATH, "//a[contains(@href, 'path-stem')]")

NAV SCOPING — do NOT scope nav links to a `<nav>` ancestor by default:
- Many sites do NOT wrap their nav links in a semantic `<nav>` tag — they use `<ul class="nav">`, a header `<div>`, or other containers. `//nav//a[...]` then matches NOTHING and times out, even though the link is plainly visible.
- DEFAULT: match the link by its href anywhere on the page: (By.XPATH, "//a[contains(@href, '/contact/')]"). visibility_of_element_located returns the first DOM match, and the desktop nav link is normally first in source order (mobile-menu and footer duplicates come later), so this resolves to the visible desktop link.
- WRONG: (By.XPATH, "//nav//a[contains(@href, '/contact/')]") ← times out on any site whose nav is not a `<nav>` tag
- WRONG: (By.XPATH, "//nav[@class='nav-2025']//a[contains(@href, '/contact/')]") ← class matching is doubly fragile
- ONLY add a container scope if an unscoped href match resolves to a HIDDEN duplicate earlier in the DOM (rare). In that case scope to a stable visible ancestor you can see in the HTML, e.g. (By.XPATH, "//header//a[contains(@href, '/contact/')]").

HREF MATCHING — NEVER EXACT, ALWAYS SUBSTRING (applies to clicks AND verify-presence checks):
- NEVER locate a link by an EXACT href value — neither CSS a[href='/path/'] nor XPath //a[@href='/path/']. The live DOM frequently rewrites a clean path into an ABSOLUTE URL (e.g. href '/platform-technology/' is rendered as 'https://silk.us/platform-technology/') and appends tracking query params (HubSpot __hstc / __hssc / __hsfp, utm_*, gclid, fbclid). An exact-match selector then matches 0 elements and TIMES OUT, even though the link is plainly visible.
- ALWAYS match a substring of the path stem instead:
    CSS:   (By.CSS_SELECTOR, "header a[href*='/platform-technology/']")
    XPath: (By.XPATH, "//a[contains(@href, '/platform-technology/')]")
- FORBIDDEN: (By.CSS_SELECTOR, "header a[href='/platform-technology/']")  ← exact match, fails on absolute/rewritten/param-appended hrefs
- This is true even when the planner step quotes the href as a clean path like '/platform-technology/' — that is the AUTHORED href, not necessarily what the browser renders. Always treat the quoted href as a substring to match with *= or contains().

NAVIGATION LINKS — <a href> elements MUST be located by href, never by id:
- Many sites (React, Next.js, Angular) generate random ids on <a> elements at build time. These ids look like "r1w2KWYLVsyGg" — they are NOT stable and MUST NOT be used.
- ALWAYS locate <a href> navigation links using EC.visibility_of_element_located, then call .click() directly:
    link = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, "//a[contains(@href, '/economy')]")))
    link.click()
- NEVER use lambda+next+find_elements patterns — they are hard to debug and unnecessary.

NON-ASCII IN HREF (Hebrew, Arabic, etc.):
- XPath @href reads the raw attribute value as-is. Use the original characters: contains(@href, '/about-us')
- NEVER use percent-encoded sequences in XPath @href (e.g. %D7%93) — they never match.

NON-ASCII IN URLS:
- driver.current_url always returns percent-encoded URLs.
- NEVER pass raw non-ASCII text to EC.url_contains. Encode it first:
    from urllib.parse import quote
    EC.url_contains(quote("some-text", safe=""))
- If an ASCII path stem exists in the href, use that instead.

OTHER LOCATOR RULES:
- NEVER use a[title='...'] — title attributes are unreliable in the live DOM.
- NEVER assert button/input value attributes — they vary by locale.
- When asserting any .get_attribute() value: use `in`, never `==`. Attributes may be None or slightly different across environments.

═══════════════════════════════════════
WAITS AND ASSERTIONS
═══════════════════════════════════════
- INTERACTION RULE: before ANY click, send_keys, or clear() — ALWAYS use EC.visibility_of_element_located to get the element, then call .click() / .send_keys() / .clear() on it directly. This confirms the element is rendered on screen before interaction. NEVER use EC.element_to_be_clickable.
    CORRECT pattern:
        el = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.ID, "submit-btn")))
        el.click()
    CORRECT for send_keys:
        inp = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.NAME, "q")))
        inp.send_keys("search text")
    INCORRECT (element_to_be_clickable — do not use):
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "submit-btn"))).click()
    INCORRECT (presence only):
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "submit-btn")))
- MULTI-MATCH CLICK TARGET (icons, toggles, buttons matched by a CLASS): when you click an element located by a CLASS or attribute selector that can match MULTIPLE elements in the DOM (e.g. a search icon `.show-searchbox`, a menu toggle, a CTA button class), the FIRST DOM match is often a hidden responsive duplicate (mobile vs desktop). EC.visibility_of_element_located on the bare selector targets that first match and TIMES OUT waiting for it to become visible. Instead collect all matches and click the first VISIBLE, non-zero-size one:
    WebDriverWait(driver, 10).until(
        lambda d: d.find_elements(By.CSS_SELECTOR, ".show-searchbox")
    )
    target = None
    for _el in driver.find_elements(By.CSS_SELECTOR, ".show-searchbox"):
        if _el.is_displayed() and _el.size["width"] > 0 and _el.size["height"] > 0:
            target = _el
            break
    if target is None:
        raise Exception("No visible match for .show-searchbox found")
    ActionChains(driver).move_to_element(target).click().perform()
  This applies ONLY to class/attribute selectors that match multiple elements. For a unique, stable id (By.ID) use plain EC.visibility_of_element_located — ids are unique so there is no hidden-duplicate problem.
- EC.presence_of_element_located — element is in the DOM (use ONLY for images and read-only checks, never before interaction)
- EC.visibility_of_element_located — use for EVERY interaction (click, send_keys, clear) AND for all assertion steps
- EC.element_to_be_clickable — DO NOT USE. Always use visibility_of_element_located instead.
- NEVER use visibility_of on <img> — images have 0×0 size in headless mode. Use presence_of and check .get_attribute("src") or .get_attribute("alt").
- NEVER use assert element.is_displayed() on images.
- HEADING VERIFICATION (h1 / h2): when a step says "verify the page heading is visible" (or "h1", or "h1 or h2"), do NOT use EC.visibility_of_element_located((By.TAG_NAME, "h1")). Many pages have NO <h1> at all (the hero is an <h2>), or the first <h1> in the DOM is a hidden responsive duplicate — both make that wait TIME OUT. Instead match `h1, h2` and verify the FIRST VISIBLE, non-zero-size heading:
    WebDriverWait(driver, 10).until(
        lambda d: d.find_elements(By.CSS_SELECTOR, "h1, h2")
    )
    heading = None
    for _el in driver.find_elements(By.CSS_SELECTOR, "h1, h2"):
        if _el.is_displayed() and _el.size["height"] > 0 and _el.size["width"] > 0:
            heading = _el
            break
    if heading is None:
        raise Exception("No visible heading (h1/h2) found on the destination page")
  FORBIDDEN: EC.visibility_of_element_located((By.TAG_NAME, "h1")) for a heading check — fails on any page with no h1 or a hidden first h1.
- For title checks: assert "keyword" in driver.title — use ONLY a locale-stable fragment: the brand name in its original script, a TLD like "IL", or a non-translatable abbreviation. NEVER use an English transliteration of a non-English brand (e.g. NEVER "drushim" when the title is in Hebrew "דרושים"). The browser renders the title in the site's locale — English words that are translations will NOT appear.
- WebDriverWait IS the assertion — do not add a redundant assert after an EC condition checks the same thing.
  FORBIDDEN:  el = WebDriverWait(...).until(EC.visibility_of_element_located(...)); assert el.is_displayed()  ← is_displayed() is already guaranteed by visibility_of
  CORRECT:    WebDriverWait(...).until(EC.visibility_of_element_located(...))  ← the wait itself is the assertion
- Use plain assert ONLY for .text or .get_attribute() values not covered by any EC.
- FORBIDDEN: EC.url_changes, EC.url_to_be — use EC.url_contains(fragment) instead.
- FORBIDDEN: WebDriverWait on body, html, or main tag — these add no signal.
- FORBIDDEN: redundant DOM check immediately after EC.url_contains passes.

POWERFUL TESTS — go deeper than just clicking a link:
- For Navigation tests: after verifying the URL, also verify a key element on the destination page is visible (heading, form, button). Example: navigate to /login, verify URL, then also verify the username input is present.
- For Forms tests: always verify the OUTCOME — success message element, URL change, or error text. A form test that stops at send_keys without verifying the result is incomplete.
- For Smoke tests: verify at least 2 distinct visible elements, not just one. Use their IDs or stable text.
- After toggling a checkbox or selecting a dropdown: assert the new state. Example: checkbox.is_selected(), select option value.

═══════════════════════════════════════
NAVIGATION
═══════════════════════════════════════
- After clicking a link, verify arrival with EC.url_contains using a fragment from the href value — not the link's visible text.
- NEVER hard-code the exact URL path (e.g. "/page.aspx") — deployed sites may redirect to a different variant ("/page/"). Use a short stem: contains("page") matches both.
- ONLY wait for a URL change when the href actually points to a different page. If the link points to the current page (e.g. logo → "/"), do not wait for a URL change — verify a page element instead.
- NEVER click or wait for elements that have style="display:none" in the HTML — they are invisible and cannot be interacted with.
- SIGN-IN / AUTH BUTTONS: clicking a sign-in button may open an inline panel OR redirect to an external login page. Always accept both outcomes: wait for either the panel element to become visible OR the URL to contain 'login'/'signin'/'account'. Example:
    WebDriverWait(driver, 5).until(
        lambda d: (d.find_elements(By.ID, "panel_id") and d.find_element(By.ID, "panel_id").is_displayed())
            or any(k in d.current_url.lower() for k in ["login", "signin", "account"])
    )
- NEW TAB: If the HTML shows target="_blank", the click opens a new tab. Capture handles before the click and switch:
    original_handles = driver.window_handles
    link.click()
    WebDriverWait(driver, 5).until(lambda d: len(d.window_handles) > len(original_handles))
    driver.switch_to.window(driver.window_handles[-1])
- If target is "_top", "_self", or absent: link opens in the same tab. Assert len(driver.window_handles) == 1 after clicking.
- NEVER use EC.url_contains("/") — every URL contains "/" and this proves nothing.

═══════════════════════════════════════
HAMBURGER / COLLAPSIBLE NAVIGATION
═══════════════════════════════════════
Many sites (IMDB, BBC, etc.) hide ALL navigation links inside a collapsible drawer behind a hamburger/MENU toggle. These links exist in the DOM but are NOT interactable until the drawer is opened — EC.element_to_be_clickable on the nav link will time out if you skip this step.

MANDATORY 3-STEP PATTERN for any navigation link that lives inside a drawer:
  Step 1 — Open the menu. ALWAYS follow ID > Name > XPATH > CSS priority — check the HTML for an id on the toggle first:
    # STEP 1a — look for an id attribute on the toggle button in the HTML. If found, use By.ID:
    menu_btn = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.ID, "exact-id-from-html"))
    )
    # STEP 1b — only if no id exists, use this multi-selector CSS fallback. NEVER invent a single aria-label:
    menu_btn = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,
            "label[for*='navDrawer'], label[for*='nav-drawer'], "
            "[aria-label*='menu' i], [aria-label*='navigation' i], "
            "[class*='hamburger'], [class*='menu-toggle'], [class*='nav-toggle'], "
            "button[aria-expanded]"
        ))
    )
    menu_btn.click()
    time.sleep(1.5)

  Step 2 — Locate the nav link now that the drawer is open (XPath @href):
    nav_link = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.XPATH, "//a[contains(@href, '/target-path')]"))
    )

  Step 3 — Click it:
    nav_link.click()
    time.sleep(2)

HOW TO DETECT a hidden nav: if the HTML shows a `<label for="...navDrawer...">`, `<button aria-label="Menu">`, `<button aria-label="Open menu">`, or any toggle with `aria-expanded`, the nav links are inside a drawer — always apply the 3-step pattern.
NEVER attempt to click a nav link directly without opening the drawer first on these sites.
NEVER use a single exact aria-label like `button[aria-label='Open main navigation']` for the toggle — different sites use different labels. Always use the multi-selector fallback chain shown above.

═══════════════════════════════════════
SEARCH INPUT INSIDE A DRAWER / PANEL
═══════════════════════════════════════
- Some sites (e.g. BBC, silk.ai) place the search form and input inside a hidden drawer or panel. The form/input exists in the DOM but is not visible until a toggle button is clicked.
- NEVER use EC.presence_of_element_located for a search input you intend to type into — it only checks DOM presence, not interactability.
- TWO failure modes — BOTH require clicking the toggle first:
  1. TimeoutException on visibility_of_element_located for the form or input — the whole container is hidden.
  2. ElementNotInteractableException after finding the input — the input is in the DOM but display:none/visibility:hidden.
- ALWAYS check for a search toggle button BEFORE trying to locate the form or input. Common toggle selectors (try in order):
    button.search-toggle, button.show-searchbox, button[aria-label*="search" i], [class*="search-toggle"], [class*="search-icon"]
- DOMAIN REDIRECT WARNING: A page navigation (e.g. silk.ai → silk.us) can change the form's action attribute. NEVER scope the search input selector to a specific domain (e.g. form[action='https://silk.ai/']). Always use a domain-agnostic selector scoped to the header:
    FORBIDDEN: By.CSS_SELECTOR, "form[action='https://silk.ai/'] input[name='s']"
    CORRECT:   By.CSS_SELECTOR, "header form input[name='s']"
- Pattern:
    # 1. Click the toggle that reveals the search panel
    search_toggle = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,
            "header button.search-toggle, header button.show-searchbox, header button[aria-label*='search' i]"))
    )
    search_toggle.click()
    time.sleep(1)
    # 2. Now wait for the input — domain-agnostic, scoped to header
    search_input = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "header form input[name='s']"))
    )
    # 3. If verifying the result URL, always check for BOTH possible domains:
    #    lambda d: "s=query" in d.current_url or "silk.ai" in d.current_url or "silk.us" in d.current_url

═══════════════════════════════════════
HOVER DROPDOWNS
═══════════════════════════════════════
- To reveal a dropdown without navigating: hover with ActionChains only — NEVER use driver.execute_script to dispatch a mouseover event:
    ActionChains(driver).move_to_element(el).pause(0.5).perform()
    time.sleep(1.5)
    WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[class*='dropdown']")))
- FORBIDDEN: driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('mouseover', ...))") — do not use JS event dispatch for hover. Pure Selenium ActionChains is sufficient and more reliable.
- FORBIDDEN: driver.execute_script("arguments[0].click();", el) — do not use JS to click elements. Use ActionChains instead:
    ActionChains(driver).move_to_element(el).click().perform()
  If the element is not visible, use EC.presence_of_element_located to locate it first, then move_to_element to scroll it into view before clicking.
- Click a nav link only when the step explicitly navigates to a new page.

═══════════════════════════════════════
FORM / SEARCH SUBMIT
═══════════════════════════════════════
- ALWAYS submit search forms using Keys.ENTER on the input field — never by clicking the search button or a search link. In headless Chrome, the autocomplete dropdown covers the search button after typing, making it unclickable. Keys.ENTER works in both headless and non-headless.
  from selenium.webdriver.common.keys import Keys
  search_input.send_keys("query text")
  time.sleep(1.5)
  search_input.send_keys(Keys.ENTER)
- CRITICAL — <a href> search links: if the search submit is an <a> tag (e.g. <a class="menu-search-bar-button-link" href="/en/search?search=">), NEVER click it. Clicking an <a> tag navigates directly to the href URL, bypassing the JavaScript validation and the typed query — the search text is lost and any expected error message will never appear. ALWAYS use Keys.ENTER on the input instead.
  FORBIDDEN: search_link.click()  ← navigates away before JS validation fires; typed text is lost
  CORRECT:   search_input.send_keys(Keys.ENTER)  ← fires the JS submit handler with the input value
- For non-search forms (login, contact, filters): locate the submit button then ALWAYS click it with ActionChains — never use submit_btn.click() directly.
- MULTIPLE SUBMIT BUTTONS — NEVER use bare visibility_of_element_located for a submit button: a page almost always has MORE THAN ONE submit button. The global header search form has its own `input[type='submit']` that comes FIRST in DOM order and is HIDDEN (0x0) behind a search toggle. EC.visibility_of_element_located((By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")) returns/polls that first hidden match and TIMES OUT — it never reaches the real (visible) form submit later in the DOM. Embedded forms (e.g. HubSpot `form.hs-form`) also load ASYNC, so the visible submit may not exist on the first poll. ALWAYS wait for a VISIBLE submit to exist, then pick the first VISIBLE non-zero-size one:
  SUBMIT_SEL = "input[type='submit'], button[type='submit']"
  WebDriverWait(driver, 20).until(
      lambda d: any(b.is_displayed() and b.size.get("height", 0) > 0
                    for b in d.find_elements(By.CSS_SELECTOR, SUBMIT_SEL))
  )
  submit_btn = next(
      b for b in driver.find_elements(By.CSS_SELECTOR, SUBMIT_SEL)
      if b.is_displayed() and b.size.get("height", 0) > 0 and b.size.get("width", 0) > 0
  )
  ActionChains(driver).scroll_to_element(submit_btn).perform()
  time.sleep(0.5)
  ActionChains(driver).move_to_element(submit_btn).click().perform()
  FORBIDDEN: EC.visibility_of_element_located((By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")) — targets the first/hidden header-search submit and times out.
  FORBIDDEN: submit_btn.click() — sticky headers and overlays intercept direct clicks. Always scroll first with ActionChains.scroll_to_element, then click with ActionChains.move_to_element().click().
- Clicking submit WITHOUT input does not navigate. Do NOT use any url_* condition. Wait for the form input to still be visible.
- DISABLED SUBMIT — required consent / anti-bot checkbox: many forms (e.g. Sitecore Forms, forms using ALTCHA/hCaptcha-style widgets) keep the submit button DISABLED until a required consent or anti-bot checkbox is ticked. A click on a disabled submit silently does nothing — no validation fires and any expected error message never appears. If the HTML shows an `input[type='checkbox']` with the `required` attribute inside the form, you MUST check it BEFORE clicking submit:
    checkbox = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "#form-id input[type='checkbox'][required]"))
    )
    ActionChains(driver).scroll_to_element(checkbox).perform()
    time.sleep(0.5)
    if not checkbox.is_selected():
        ActionChains(driver).move_to_element(checkbox).click().perform()
    time.sleep(1)
  Then wait for the submit button to become enabled before clicking it (the widget may run a brief background check):
    WebDriverWait(driver, 15).until(lambda d: not submit_btn.get_attribute("disabled"))
  This applies to NEGATIVE form tests too: to surface "required field" validation errors you must first enable the submit button, otherwise the click is a no-op.
- ALTCHA "I'M NOT A ROBOT" WIDGET — if a step mentions an anti-bot / "I'm not a robot" checkbox, FIRST check whether the HTML contains an `<altcha-widget>` element (ALTCHA). If it does, the anti-bot control is ALTCHA — NOT reCAPTCHA and NOT hCaptcha. Critical facts:
    * ALTCHA renders NO iframe. Do NOT look for `iframe[src*='recaptcha']` / `iframe[title*='reCAPTCHA']` — those never exist on an ALTCHA page and the switch_to.frame branch will silently do nothing.
    * The visible checkbox and the literal text "I'm not a robot" are injected by ALTCHA's JavaScript AFTER it fetches a challenge, so they are ABSENT from the static HTML you are given. Never try to match the "I'm not a robot" text or a `[required]` checkbox id — they are not in the HTML.
    * The real checkbox lives INSIDE the `<altcha-widget>` element, possibly in its shadow DOM. Selenium locators cannot reliably reach into a web component's shadow root, so use a small JS snippet that handles both light and shadow DOM. This is the one sanctioned use of execute_script to click — it is required for shadow-DOM web components.
  Exact pattern to emit for an ALTCHA anti-bot step:
    try:
        altcha = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "altcha-widget, #altcha-widget"))
        )
        ActionChains(driver).scroll_to_element(altcha).perform()
        time.sleep(1)
        driver.execute_script(\"\"\"
            const w = document.querySelector('altcha-widget, #altcha-widget');
            if (!w) return;
            const root = w.shadowRoot || w;
            const cb = root.querySelector("input[type='checkbox']");
            if (cb && !cb.checked) cb.click();
        \"\"\")
        time.sleep(3)  # let ALTCHA's proof-of-work complete and set the solution
    except Exception:
        pass
  If the step says the checkbox is optional ("if present"), wrap it in try/except as above so a missing widget never fails the test.
- CHECKBOX / RADIO INTERACTION — always scroll the box to the viewport centre with ActionChains.scroll_to_element BEFORE clicking it, then click with ActionChains.move_to_element().click(). A bare checkbox.click() scrolls the box to the top edge under a sticky header, which intercepts the click so the box never toggles. Guard with `if not box.is_selected():` so a re-run does not un-tick it.
  NOTE: this manual-click rule is for REAL `<input type='checkbox'>` form fields with a stable id (e.g. a consent checkbox). It does NOT apply to the ALTCHA anti-bot checkbox above — that one is reached via the JS snippet because it lives inside a web component.
- VERIFYING A FORM SUBMISSION OUTCOME — avoid false passes: do NOT assert on bare `[class*='validation']`, `[class*='error']`, `[class*='success']`, or `[class*='invalid']` selectors. Forms (especially Sitecore/ASP.NET) pre-render EMPTY marker spans like `field-validation-valid` that are ALWAYS in the DOM, so find_elements on those returns matches before anything is submitted and the test passes without the form responding. Likewise NEVER use `//*[contains(., 'thank you')]` or `//*[contains(text(), 'success')]` against generic `*` — `.`/string-value aggregates ALL descendant text, so huge ancestor containers (`<body>`, `<main>`) match if the phrase appears anywhere in static page content. Instead require a DISPLAYED element whose OWN text is non-empty, scoped to the real outcome class:
    WebDriverWait(driver, 15).until(
        lambda d: any(e.is_displayed() and (e.text or "").strip()
                      for e in d.find_elements(By.CSS_SELECTOR, ".field-validation-error, [class*='success'], [class*='thank'], [class*='confirmation']"))
    )
  Prefer the most specific outcome locator the HTML actually shows (a named success container, or `.field-validation-error` for validation), or assert a URL change to a confirmation/thank-you path. A real outcome must be VISIBLE with TEXT — presence in the DOM alone proves nothing.


═══════════════════════════════════════
SCROLLING AND VISIBILITY — STRICT RULES
═══════════════════════════════════════
DEFAULT RULE — no scroll needed:
- EC.visibility_of_element_located finds elements anywhere on the page — it only requires the element is not display:none or visibility:hidden. It does NOT require the element to be in the viewport.
- NEVER add a scroll step before a visibility check unless one of the specific exceptions below applies.
- FORBIDDEN pattern (do not use by default):
    el = WebDriverWait(driver, 10).until(EC.presence_of_element_located(...))
    ActionChains(driver).scroll_to_element(el).perform()
    time.sleep(2)
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located(...))
  This 3-step pattern is ONLY valid for CSS-animated elements (see below). Do not use it otherwise.
 

If no visible matching element is initially found, the executor should progressively scroll down using ActionChains(driver).scroll_by_amount(...) and retry locating visible elements before failing the test.
EXCEPTION 1 — CSS animation reveal (rare):
- ONLY use the 3-step presence→scroll→visibility pattern when the HTML shows the target element is inside a container with animation class names like "animate-in", "fade-in", "slide-in", or "scroll-reveal". These elements are visibility:hidden until an IntersectionObserver fires.
- If you are not certain the element uses a CSS scroll animation, use plain EC.visibility_of_element_located — do not add the scroll.

EXCEPTION 2 — footer elements only:
- Many sites do not use a semantic <footer> tag — they use a div/section with a footer class or id. ALWAYS use the broad CSS selector below so the locator works regardless of the site's HTML structure.
- CORRECT footer pattern (use this every time a step verifies the footer):
    # Wait until at least one match exists, then pick the LAST VISIBLE, non-zero-size
    # match — the real bottom-of-page footer. Taking [-1] blindly can grab a hidden
    # 0x0 placeholder, and scroll_to_element on a zero-size element raises
    # ElementNotInteractableException. Iterate in reverse and check .size.
    WebDriverWait(driver, 10).until(
        lambda d: d.find_elements(By.CSS_SELECTOR, "footer, [class*='footer'], #footer, [id*='footer']")
    )
    footer = None
    for _el in reversed(driver.find_elements(By.CSS_SELECTOR, "footer, [class*='footer'], #footer, [id*='footer']")):
        if _el.size.get("height", 0) > 0 and _el.size.get("width", 0) > 0:
            footer = _el
            break
    if footer is None:
        raise Exception("No visible footer element with non-zero size found")
    ActionChains(driver).scroll_to_element(footer).perform()
    time.sleep(1)
    assert footer.is_displayed(), "Footer is not visible"
- WHY reversed + size check: the broad selector matches hidden zero-size placeholders (pre-footer, mobile-footer). Taking `[-1]` alone is unsafe — that element may be 0x0 and scroll_to_element raises ElementNotInteractableException. Iterating in reverse and checking `.size` finds the last REAL footer.
- FORBIDDEN: `find_elements(...)[-1]` without a size check — the last match may be a hidden 0x0 element.
- FORBIDDEN: EC.presence_of_element_located with the broad footer selector — it returns the FIRST match, which may be a hidden element.
- FORBIDDEN: EC.visibility_of_element_located((By.TAG_NAME, "footer")) — fails on sites that use div.footer instead of a <footer> tag.
- FORBIDDEN: scroll_by_amount(0, 3000) for footer — a fixed scroll amount may not reach the footer on long pages. Always use scroll_to_element on the located element.

EXCEPTION 3 — CSS-animated interactive elements (dropdowns, filters, inputs below the fold):
- Some elements (job filters, custom dropdowns, search inputs) are hidden by a parent IntersectionObserver animation. scroll_to_element on the child alone does NOT fire the parent's observer — the element stays hidden.
- CORRECT 4-step pattern:
    # 1. Scroll the viewport by amount to fire the parent's IntersectionObserver
    ActionChains(driver).scroll_by_amount(0, 800).perform()
    time.sleep(3)
    # 2. Wait until the element is truly clickable (visible + enabled)
    el = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, ".target-element"))
    )
    # 3. Scroll precisely to the element so it is centered in the viewport (prevents negative-y click errors)
    ActionChains(driver).scroll_to_element(el).perform()
    time.sleep(0.5)
    # 4. Click
    el.click()
- Use this pattern when: an element is present in the DOM but never becomes visible/clickable via scroll_to_element alone, AND the error is ElementClickInterceptedException with a negative y-coordinate (element above viewport) or ElementNotInteractableException after scroll.
- The scroll_by_amount fires the parent observer; scroll_to_element then precisely centers the now-revealed element before clicking.

═══════════════════════════════════════
PACING — VIDEO CLARITY
═══════════════════════════════════════
- After every user interaction (click, send_keys, submit), add time.sleep(1.5) so the video recording shows each step clearly.
- After a navigation that changes the URL, add time.sleep(2) instead.
- After every EC.url_contains(...) wait (URL verification after navigation), add time.sleep(2) immediately after it — before the except block. This gives the page time to settle after the URL change and makes the final state visible in the video recording.
- Do NOT add sleeps inside WebDriverWait lambdas or polling loops.

═══════════════════════════════════════
COOKIE BANNERS
═══════════════════════════════════════
- The test runner auto-dismisses cookie/consent banners after every driver.get(). Do NOT write code for this.

═══════════════════════════════════════
IMAGES / LAZY-LOADING
═══════════════════════════════════════
- On recorded sessions the runner auto-scrolls the page after every driver.get() and waits for images to finish loading, so the recording/screenshot shows a fully-painted page. Do NOT write your own scroll-to-load or "wait for img.complete / naturalWidth" code — it is handled by the runner.

═══════════════════════════════════════
ERROR HANDLING
═══════════════════════════════════════
- The test body is already inside the try block (see OUTPUT FORMAT).
- In the except block: save a screenshot to "error_{{int(time.time())}}.png", then re-raise. Never swallow exceptions.

═══════════════════════════════════════
HTML EXTRACTION RULES — CRITICAL
═══════════════════════════════════════
- You MUST extract and use the EXACT attributes from the HTML below: id, name, placeholder, value, visible text, class (if unique). Do NOT invent or assume any attribute value.
- Only use locators that exist verbatim in the HTML below. If an attribute is not in the HTML, do not use it.
- NEVER use `[aria-label='...']` unless that exact aria-label string appears in the HTML. Do not guess or infer aria-labels from brand names or common sense.
- For Hebrew or non-Latin text: match the exact visible text from the HTML, preserving spaces, punctuation, and case exactly as they appear.
- Do NOT navigate away from the provided Website URL unless a step explicitly instructs it.


BROWSER MODE: {headless}

Website URL: {website}

Full HTML: {html}"""

# Per-case portion — all steps passed at once so the LLM sees the full picture.
USER_PROMPT_TEMPLATE = """Implement EXACTLY these test steps — one code block per step, in order. Do not add, skip, merge, or reorder any step.

Steps:
{steps}

Expected result: {expected}"""



def _clean_html(html: str, max_chars: int = 30000) -> str:
    import re
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    html = re.sub(r'\s+', ' ', html).strip()
    return html[:max_chars]

def _is_israeli_site(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    return host.endswith(".il")

def extract_full_html(url: str, max_chars: int = 30000) -> str:
    cache_key = (url, max_chars)
    if cache_key in _html_cache:
        return _html_cache[cache_key]
    import requests as _requests
    lang = "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7" if _is_israeli_site(url) else "en-US,en;q=0.9"
    resp = _requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Accept-Language": lang,
    })
    resp.raise_for_status()
    html = _clean_html(resp.text, max_chars=max_chars)
    _html_cache[cache_key] = html
    return html


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("python"):
            text = text[6:]
        text = text.strip()
    return text


def _fix_syntax(code: str, llm, cached_system, case_id: str):
    """Return code if valid, or ask Claude once to fix it.
    Checks both syntax (compile) and structure (def run present).
    Returns None if still broken after one retry."""
    import sys as _sys, time as _t
    from langchain_core.messages import HumanMessage

    def _check(c):
        if not c:
            return "empty code"
        if "def run(" not in c and "def run (" not in c:
            return "missing def run(driver): — code must define a run() function"
        if "if __name__" in c:
            return "unexpected if __name__ block — it is appended automatically, do not include it"
        try:
            compile(c, f"{case_id}.py", "exec")
        except SyntaxError as e:
            return str(e)
        return None

    error = _check(code)
    if error is None:
        return code

    print(f"[syntax] {case_id} invalid ({error}) — asking Claude to fix.")
    fix_prompt = HumanMessage(content=(
        f"The following Python script is invalid ({error}):\n\n```python\n{code}\n```\n\n"
        f"Return ONLY the corrected Python script. Requirements: define a single def run(driver): function "
        f"containing a try/except/finally block; driver is provided by caller; no if __name__ block. "
        f"No markdown fences, no explanation."
    ))
    try:
        print(f"[anthropic] _fix_syntax.invoke case={case_id} at {_t.time()} (executor.py)", flush=True, file=_sys.stderr)
        fixed = _strip_code_fences(llm.invoke([cached_system, fix_prompt]).content.strip())
        error2 = _check(fixed)
        if error2 is None:
            return fixed
        print(f"[syntax] {case_id} still invalid after fix: {error2}")
        return None
    except Exception as e:
        print(f"[syntax] {case_id} fix request failed: {e}")
        return None


def generate_test_files(plan):
    import sys as _sys, time as _t

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_files = []
    website = plan.get("website", "")
    page_html = extract_full_html(website)
    headless_label = "headless Chrome" if HEADLESS else "regular Chrome"

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=ANTHROPIC_API_KEY)

    # Build the system prompt ONCE per run — cached across all cases and steps.
    system_text = SYSTEM_PROMPT_TEMPLATE.format(
        website=website,
        html=page_html,
        headless=headless_label,
        main_block=_make_main_block(website),
    )
    cached_system = SystemMessage(content=[{
        "type": "text",
        "text": system_text,
        "cache_control": {"type": "ephemeral"},
    }])

    for case in plan["cases"]:
        case_id = case["id"]
        steps = case.get("steps", [])
        expected = case.get("expected", "")

        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        user_text = USER_PROMPT_TEMPLATE.format(steps=steps_text, expected=expected)
        messages = [cached_system, HumanMessage(content=user_text)]
        print(f"[anthropic] generate_test_files.invoke case={case_id} at {_t.time()} (executor.py)", flush=True, file=_sys.stderr)
        response = llm.invoke(messages)
        combined = _strip_code_fences(response.content.strip())

        # Strip any __main__ block Claude added despite instructions.
        if "if __name__" in combined:
            combined = combined[:combined.index("if __name__")].rstrip()

        # Validate structure + syntax; ask Claude to fix once if broken.
        combined = _fix_syntax(combined, llm, cached_system, case_id)
        if combined is None:
            print(f"[generate_test_files] Skipping {case_id}: could not produce valid Python after retry.")
            continue

        file_path = os.path.join(OUTPUT_DIR, f"{case_id}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(combined + "\n" + _make_main_block(website))

        test_files.append((case_id, file_path))
        print(f"✅ Generated test file: {file_path}")

    return test_files

# -----------------------------
# 3. Run generated Selenium tests
# -----------------------------
def run_test_file(case_id, file_path, website=""):
    import subprocess, sys, textwrap

    # Pre-compute locale values so f-string expressions below need no backslashes
    # (backslashes inside f-string {} are a SyntaxError on Python < 3.12).
    _israeli = _is_israeli_site(website)
    _nav_langs = "['he-IL', 'he', 'en-US', 'en']" if _israeli else "['en-US', 'en']"

    # Build a self-contained runner script so Chrome runs in its own process,
    # completely isolated from the gunicorn worker (avoids "Chrome instance exited").
    runner = textwrap.dedent(f"""
        import sys
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time

        import os as _os
        _using_lambdatest = bool({repr(os.environ.get("LT_USERNAME", ""))} and {repr(os.environ.get("LT_ACCESS_KEY", ""))})
        _selenoid_url = _os.environ.get("SELENOID_URL", {repr(os.environ.get("SELENOID_URL", ""))})
        opts = Options()
        if not _using_lambdatest and not _selenoid_url and _os.environ.get("HEADLESS", "true").lower() != "false":
            opts.add_argument("--headless=new")
        _israeli = {_israeli}
        _base_args = [
            "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-gpu", "--disable-extensions",
            "--no-first-run", "--disable-background-networking",
            "--disable-sync", "--disable-default-apps",
            "--disable-blink-features=AutomationControlled",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "--window-size=1440,900",
        ]
        if not _using_lambdatest:
            # Selenoid records video for QA review / demo capture, so images MUST
            # load there — blocking them produces recordings full of broken-image
            # placeholders. Keep image-blocking ONLY for headless local/Render runs
            # (memory + speed). The other two flags are memory optimisations and
            # are safe to keep everywhere.
            if not _selenoid_url:
                _base_args.append("--blink-settings=imagesEnabled=false")
            _base_args += ["--disable-software-rasterizer", "--renderer-process-limit=1"]
        if _israeli:
            _base_args.append("--lang=he-IL")
        for arg in _base_args:
            opts.add_argument(arg)
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        _lang_pref = "he,he-IL,en-US,en" if _israeli else "en-US,en"
        opts.add_experimental_option("prefs", {{"intl.accept_languages": _lang_pref}})
        opts.page_load_strategy = 'none'

        _lt_username = {repr(os.environ.get("LT_USERNAME", ""))}
        _lt_access_key = {repr(os.environ.get("LT_ACCESS_KEY", ""))}

        print("[driver] LT_USERNAME=" + ("set" if _lt_username else "MISSING") + " LT_ACCESS_KEY=" + ("set" if _lt_access_key else "MISSING"), flush=True)

        if _lt_username and _lt_access_key:
            print("[driver] Using LambdaTest remote browser", flush=True)
            # Remote browser via LambdaTest — no local Chrome, no OOM on Render.
            opts.set_capability("LT:Options", {{
                "username": _lt_username,
                "accessKey": _lt_access_key,
                "build": "CaptainFix",
                "name": {repr(case_id)},
                "headless": False,
                "w3c": True,
                "platformName": "macOS Sequoia",
                "geoLocation": "IL" if _israeli else "",
            }})
            _lt_endpoint = "https://" + _lt_username + ":" + _lt_access_key + "@hub.lambdatest.com/wd/hub"
            driver = webdriver.Remote(command_executor=_lt_endpoint, options=opts)
            driver.set_script_timeout(30)
        elif _selenoid_url:
            print("[driver] Using Selenoid remote browser at " + _selenoid_url, flush=True)
            opts.set_capability("selenoid:options", {{
                "enableVideo": True,
                "enableVNC": True,
                "name": {repr(case_id)},
            }})
            driver = webdriver.Remote(command_executor=_selenoid_url, options=opts)
            driver.set_script_timeout(30)
        else:
            print("[driver] Using local Chrome (no remote credentials set)", flush=True)
            driver = None
            for attempt in range(3):
                try:
                    driver = webdriver.Chrome(options=opts)
                    driver.set_script_timeout(8)
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        raise

        # Stealth CDP patches — supported by both local Chrome and Browserless.
        _cdp_stealth = '''
            Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
            if (!window.chrome) window.chrome = {{}};
            if (!window.chrome.runtime) window.chrome.runtime = {{}};
            Object.defineProperty(navigator, 'plugins', {{get: () => [
                {{name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format'}},
                {{name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:''}},
                {{name:'Native Client', filename:'internal-nacl-plugin', description:''}}
            ]}});
            Object.defineProperty(navigator, 'mimeTypes', {{get: () => [
                {{type:'application/pdf', suffixes:'pdf', description:''}}
            ]}});
            Object.defineProperty(navigator, 'languages', {{get: () => {_nav_langs}}});
            Object.defineProperty(navigator, 'platform', {{get: () => 'Win32'}});
            const _origPerms = navigator.permissions.query.bind(navigator.permissions);
            navigator.permissions.query = (p) =>
                p.name === 'notifications'
                    ? Promise.resolve({{state: 'default', onchange: null}})
                    : _origPerms(p);
        '''
        try:
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {{"source": _cdp_stealth}})
            if _israeli:
                driver.execute_cdp_cmd("Emulation.setGeolocationOverride", {{
                    "latitude": 31.7683, "longitude": 35.2137, "accuracy": 100
                }})
        except Exception:
            pass  # Remote drivers may not support CDP; stealth is best-effort

        # Auto-dismiss cookie banners after every navigation so tests don't
        # have to know about them. Covers OneTrust, TrustArc, and generic
        # accept-all buttons (English text + Hebrew "אישור"/"אשר").
        _cookie_css = [
            # Accept buttons — major platforms first
            "#onetrust-accept-btn-handler",
            "#truste-consent-button",
            ".cky-btn-accept",
            ".iubenda-cs-accept-btn",
            "#coiOverlay button[data-action='accept-all']",
            "[id*='accept-all']",
            "[id*='acceptAll']",
            "[id*='accept_all']",
            "[class*='accept-all']",
            "[class*='acceptAll']",
            "[data-testid*='accept']",
            "[aria-label*='Accept']",
            "[aria-label*='accept']",
            # Close (X) buttons — scoped to a cookie/consent container first
            "[id*='cookie'] button[aria-label='Close']",
            "[id*='cookie'] [class*='close']",
            "[class*='cookie'] button[aria-label='Close']",
            "[class*='cookie'] [class*='close']",
            "[id*='consent'] [class*='close']",
            "[class*='consent'] [class*='close']",
            "#CybotCookiebotDialogBodyButtonClose",
            ".cky-btn-close",
            ".iubenda-cs-close-btn",
            # Generic X / close buttons — last resort
            "button[aria-label='Close']",
            "button[aria-label='close']",
            "button.btn-close",
            "[role='button'][aria-label='Close']",
            # Castro.com custom cookie/promo popup
            ".idus_popup_widget_cookie_popup button",
            ".idus_popup_wrap button",
            "[class*='idus_popup'] button",
        ]
        _cookie_xpaths = [
            # Accept text (English)
            "//*[(self::button or self::a or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow all')]",
            "//*[(self::button or self::a or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept all')]",
            "//*[(self::button or self::a or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept cookies')]",
            "//*[(self::button or self::a or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'i agree')]",
            "//*[(self::button or self::a or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it')]",
            # Accept text (Hebrew)
            "//*[(self::button or self::a or self::span)][contains(., 'אישור')]",
            "//*[(self::button or self::a or self::span)][contains(., 'אשר')]",
            "//*[(self::button or self::a or self::span)][contains(., 'מאשר')]",
            "//*[(self::button or self::a or self::span)][contains(., 'אני מסכים')]",
            "//*[(self::button or self::a or self::span)][contains(., 'אני מאשר')]",
            "//*[(self::button or self::a or self::span)][contains(., 'קבל הכל')]",
            "//*[(self::button or self::a or self::span)][contains(., 'קבל את כל')]",
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='קבל']",
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='המשך']",
            # Close (X) — unicode multiplication sign, heavy ballot X, and Hebrew close/cancel
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='×']",
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='✕']",
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='✖']",
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='סגור']",
            "//*[(self::button or self::a or self::div or self::span)][normalize-space()='ביטול']",
        ]
        def _dismiss_cookies(d):
            # Single fast JS round-trip: click the accept control and strip any
            # Bootstrap modal/backdrop. Cheap enough to poll on a slow remote
            # browser, where the per-selector Selenium passes below are too slow
            # to catch a modal that is injected several seconds after load.
            def _js_kill_modal():
                try:
                    return bool(d.execute_script(r\"\"\"
                        var handled = false;
                        var accSel = ['#onetrust-accept-btn-handler', '#truste-consent-button',
                                      '.iubenda-cs-accept-btn', '#cookieConsentModal .btn-accept',
                                      '.btn-accept', '#CybotCookiebotDialogBodyButtonClose',
                                      '.cky-btn-accept', '[id*="accept-all"]', '[id*="acceptAll"]'];
                        var clicked = false;
                        for (var i = 0; i < accSel.length; i++) {{
                            var el = document.querySelector(accSel[i]);
                            if (el && el.offsetParent !== null) {{ el.click(); clicked = true; break; }}
                        }}
                        if (!clicked) {{
                            var labels = ['allow all', 'accept all', 'accept cookies', 'i agree', 'got it', 'קבל הכל', 'אישור'];
                            var nodes = document.querySelectorAll('button, a, span');
                            for (var j = 0; j < nodes.length; j++) {{
                                var t = (nodes[j].innerText || '').trim().toLowerCase();
                                if (labels.indexOf(t) >= 0 && nodes[j].offsetParent !== null) {{
                                    nodes[j].click(); clicked = true; break;
                                }}
                            }}
                        }}
                        if (clicked) handled = true;
                        // Safety: strip any leftover backdrop/modal so it cannot
                        // intercept a click. This does NOT count as "handled" — only
                        // an actual accept click grants consent (permanent). Force-
                        // removing without consent lets the site re-open the modal,
                        // so the loop must keep polling until accept is clicked.
                        document.querySelectorAll('.modal-backdrop').forEach(function(e) {{ e.remove(); }});
                        document.querySelectorAll('.modal.show, .modal[style*="display: block"]').forEach(function(m) {{
                            m.classList.remove('show'); m.style.display = 'none';
                            m.setAttribute('aria-hidden', 'true');
                        }});
                        document.body.classList.remove('modal-open');
                        document.body.style.overflow = ''; document.body.style.paddingRight = '';
                        return handled;
                    \"\"\"))
                except Exception:
                    return False
            # Phase 1 — poll the single fast JS dismiss frequently. One round-trip
            # per poll stays responsive even on a slow remote browser, so a modal
            # injected several seconds after load is caught right as it appears.
            end = time.time() + (12 if _using_lambdatest else 9)
            while time.time() < end:
                try:
                    if _js_kill_modal():
                        return
                except Exception:
                    pass
                time.sleep(0.4)
            # Phase 2 — last-resort per-selector Selenium sweep for banners the JS
            # pass cannot reach (e.g. close-only X buttons, exotic markup).
            try:
                for sel in _cookie_css:
                    for el in d.find_elements(By.CSS_SELECTOR, sel):
                        if el.is_displayed() and el.is_enabled():
                            el.click()
                            return
                for xp in _cookie_xpaths:
                    for el in d.find_elements(By.XPATH, xp):
                        if el.is_displayed() and el.is_enabled():
                            el.click()
                            return
            except Exception:
                pass
            # Fallback: JS hide of any small fixed/sticky element whose text
            # mentions cookies — catches custom banners (e.g. castro.com's
            # Hebrew "גולש יקר, אנו משתמשים בקבצי Cookie") with no exposed X.
            try:
                d.execute_script(\"\"\"
                    const phrases = ['גולש יקר', 'בקבצי Cookie', 'cookie', 'cookies', 'עוגיות'];
                    document.querySelectorAll('div, section, aside').forEach(el => {{
                        const cs = getComputedStyle(el);
                        if ((cs.position === 'fixed' || cs.position === 'sticky') && el.offsetHeight < 400) {{
                            const text = (el.innerText || '').toLowerCase();
                            for (const p of phrases) {{
                                if (text.includes(p.toLowerCase())) {{
                                    el.style.display = 'none';
                                    return;
                                }}
                            }}
                        }}
                    }});
                \"\"\")
            except Exception:
                pass
            # Final safety net: JS-hide any remaining visible modal/dialog overlays
            # so they cannot intercept clicks on page elements underneath them.
            try:
                d.execute_script(\"\"\"
                    document.querySelectorAll('[role="dialog"], .modal-popup, .modal-slide._show, [class*="popup_wrap"]').forEach(el => {{
                        if (el.offsetWidth > 0 && el.offsetHeight > 0) {{
                            el.style.display = 'none';
                        }}
                    }});
                    // Bootstrap modal cleanup — the backdrop is a separate element
                    // that intercepts clicks even after the dialog itself is hidden.
                    document.querySelectorAll('.modal-backdrop').forEach(el => {{ el.remove(); }});
                    document.querySelectorAll('.modal.show, .modal[style*="display: block"]').forEach(m => {{
                        m.classList.remove('show'); m.style.display = 'none'; m.setAttribute('aria-hidden', 'true');
                    }});
                    document.body.classList.remove('modal-open');
                    document.body.style.overflow = ''; document.body.style.paddingRight = '';
                \"\"\")
            except Exception:
                pass
        def _wait_for_images(d, scroll_timeout=8, settle_timeout=8):
            # Recorded sessions (Selenoid/LambdaTest) must be fully painted on camera.
            # Many sites lazy-load <img> via IntersectionObserver, so scroll through
            # the page to trigger them, return to the top, then wait until the count
            # of decoded images stabilises. Counting stable (not "all complete")
            # tolerates a few permanently-broken assets like tracking pixels.
            import time as _t
            try:
                _end = _t.time() + scroll_timeout
                _last_h = -1
                while _t.time() < _end:
                    _h = d.execute_script("return document.body.scrollHeight") or 0
                    d.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    _t.sleep(0.4)
                    if _h == _last_h:
                        break
                    _last_h = _h
                d.execute_script("window.scrollTo(0, 0);")
                _end2 = _t.time() + settle_timeout
                _prev = -1
                _stable = 0
                while _t.time() < _end2:
                    _loaded = d.execute_script(
                        "return Array.from(document.images).filter(i => i.complete && i.naturalWidth > 0).length;")
                    if _loaded == _prev:
                        _stable += 1
                        if _stable >= 2:
                            break
                    else:
                        _stable = 0
                    _prev = _loaded
                    _t.sleep(0.5)
            except Exception:
                pass

        _orig_get = driver.get
        def _patched_get(url):
            _orig_get(url)
            # Allow WAF/Cloudflare JS challenges to run before the test starts waiting.
            # LambdaTest remote sessions have extra latency — consent modals load later.
            time.sleep(6 if _using_lambdatest else 3)
            _dismiss_cookies(driver)
            # Images are only enabled on recorded sessions; trigger lazy-load and wait
            # so the video/screenshots show a fully-painted page (no broken images).
            if _selenoid_url or _using_lambdatest:
                _wait_for_images(driver)
            # Progress snapshot — run in a thread so a slow Chrome doesn't block navigation.
            import threading as _threading
            _t = _threading.Thread(target=lambda: driver.save_screenshot(screenshot_path), daemon=True)
            _t.start()
            _t.join(timeout=8)
        driver.get = _patched_get

        screenshot_path = {repr(os.path.join(SCREENSHOT_DIR, case_id + ".png"))}
        import os as _os
        _os.makedirs({repr(SCREENSHOT_DIR)}, exist_ok=True)

        # Background snapshot loop — overwrites the file every 5s so the
        # most-recent page state is always on disk when the subprocess is killed.
        # Each save_screenshot() runs in its own daemon thread with an 8s timeout
        # so a hung Chrome (which makes save_screenshot block forever) never
        # prevents the loop from continuing or from writing a later screenshot.
        import threading as _threading
        _stop_snap = _threading.Event()
        def _safe_snap():
            try:
                driver.save_screenshot(screenshot_path)
            except Exception:
                pass
        def _snapshot_loop():
            while not _stop_snap.is_set():
                _t = _threading.Thread(target=_safe_snap, daemon=True)
                _t.start()
                _t.join(timeout=8)  # abandon if Chrome is hung
                if _stop_snap.wait(5):
                    break
        _snap_thread = _threading.Thread(target=_snapshot_loop, daemon=True)
        _snap_thread.start()

        try:
            code = open({repr(file_path)}).read()
            _ns = {{'__name__': 'captainfix_runner', 'driver': driver}}
            exec(code, _ns)
            _ns['run'](_ns['driver'])
            print("RESULT:Pass")
        except Exception as e:
            try:
                _err_t = _threading.Thread(target=lambda: driver.save_screenshot(screenshot_path), daemon=True)
                _err_t.start()
                _err_t.join(timeout=8)
                import base64 as _b64
                if _os.path.exists(screenshot_path):
                    with open(screenshot_path, "rb") as _sf:
                        _b64data = _b64.b64encode(_sf.read()).decode("ascii")
                    print(f"SCREENSHOT_B64:{{_b64data}}")
            except Exception:
                pass
            err_msg = str(e).replace("\\n", " ").replace("\\r", "").strip()
            first_line = err_msg.splitlines()[0] if err_msg.splitlines() else err_msg
            print(f"RESULT:Fail:{{first_line}}")
        finally:
            _stop_snap.set()
            try:
                _snap_thread.join(timeout=2)
            except Exception:
                pass
            try:
                driver.quit()
            except Exception:
                pass
            # Hard-kill any lingering Chrome processes so they don't accumulate
            # across tests and exhaust the 512MB memory limit.
            import subprocess as _sp
            _sp.run(["pkill", "-9", "-f", "chrome"], capture_output=True)
            time.sleep(1)
    """)

    # Remove any stale screenshot from a previous run with the same case_id
    # so a timed-out test never serves an old site's image.
    stale = os.path.join(SCREENSHOT_DIR, case_id + ".png")
    try:
        os.remove(stale)
    except FileNotFoundError:
        pass

    result = {"id": case_id, "status": "Pass", "error": None, "screenshot": None}
    with _chrome_lock:
        proc = None
        try:
            # start_new_session=True puts the runner in its own process group so
            # killpg() can kill Chrome's child processes too — they otherwise keep
            # the stdout pipe open and block communicate() after timeout.
            proc = subprocess.Popen(
                [sys.executable, "-c", runner],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                import signal as _signal, os as _os
                try:
                    _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
                except Exception:
                    proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""  # orphaned Chrome children keep pipe open — move on
                result["status"] = "Fail"
                result["error"] = "Test timed out after 120 seconds"

            time.sleep(2)
            combined = stdout + "\n" + stderr
            import sys as _sys
            for line in combined.splitlines():
                if not line.startswith("SCREENSHOT_B64:"):
                    print(f"[runner:{case_id}] {line}", flush=True, file=_sys.stderr)
            for line in combined.splitlines():
                if line.startswith("SCREENSHOT_B64:"):
                    result["screenshot_b64"] = line[len("SCREENSHOT_B64:"):]
                elif line.startswith("RESULT:Pass"):
                    result["status"] = "Pass"
                elif line.startswith("RESULT:Fail:"):
                    result["status"] = "Fail"
                    result["error"] = line[len("RESULT:Fail:"):].strip()
            if result["status"] == "Fail" and not result.get("error"):
                result["error"] = (stderr or stdout or "Unknown error").strip()
        except Exception as e:
            result["status"] = "Fail"
            result["error"] = str(e)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
        finally:
            # Always try the disk fallback when no screenshot was captured via
            # stdout — covers normal failures where the exception handler's
            # save_screenshot failed, AND timeout kills where SIGKILL prevented
            # the subprocess from printing SCREENSHOT_B64.
            if not result.get("screenshot_b64"):
                fallback = os.path.join(SCREENSHOT_DIR, case_id + ".png")
                if os.path.exists(fallback):
                    import base64
                    with open(fallback, "rb") as _f:
                        result["screenshot_b64"] = base64.b64encode(_f.read()).decode("ascii")

    return result


# -----------------------------
# 4. Run all tests and save results
# -----------------------------
def main():
    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # 1. Generate Selenium test files from AI
    test_files = generate_test_files(plan)
    website = plan.get("website", "")

    # 2. Execute all test files and collect results
    results = []
    for i, (case_id, file_path) in enumerate(test_files):
        print(f"▶ Running test {case_id} ...")
        result = run_test_file(case_id, file_path, website=website)
        results.append(result)
        print(f"✔ {case_id} → {result['status']}")

    # 3. Save structured results
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Finished! Results saved in {RESULTS_JSON}")


if __name__ == "__main__":
    main()
