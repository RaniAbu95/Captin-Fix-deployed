import os
import json
import threading
import traceback
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from langchain_anthropic import ChatAnthropic
import time
from config import ANTHROPIC_API_KEY

from langchain_core.prompts import ChatPromptTemplate
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
        # Try Selenium click first
        for sel in ["#CybotCookiebotDialogBodyButtonClose", ".cky-btn-accept",
                    "[id*='accept-all']", "[id*='acceptAll']"]:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in els:
                    if el.is_displayed() and el.is_enabled():
                        el.click()
                        return True
            except Exception:
                pass
        for xpath in [
            "//*[(self::button or self::a or self::span)][contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept all')]",
            "//*[(self::button or self::a or self::span)][contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept cookies')]",
            "//*[(self::button or self::a or self::span)][contains(.,'אישור')]",
            "//*[(self::button or self::a or self::span)][contains(.,'קבל הכל')]",
            "//*[(self::button or self::a or self::span)][normalize-space()='\xd7']",
        ]:
            try:
                els = driver.find_elements(By.XPATH, xpath)
                for el in els:
                    if el.is_displayed() and el.is_enabled():
                        el.click()
                        return True
            except Exception:
                pass
        # JS fallback — clicks the button by text even if Selenium can't reach it
        try:
            clicked = driver.execute_script("""
                var buttons = document.querySelectorAll('button, a, span');
                for (var i = 0; i < buttons.length; i++) {
                    var t = buttons[i].innerText.trim().toLowerCase();
                    if (t === 'accept all' || t === 'accept cookies' || t === 'קבל הכל' || t === 'אישור') {
                        buttons[i].click();
                        return true;
                    }
                }
                return false;
            """)
            if clicked:
                return True
        except Exception:
            pass
        return False

    _orig_get = driver.get
    def _patched_get(url):
        _orig_get(url)
        time.sleep(5)
        # Retry dismissal for up to 10 seconds — banner may load after initial paint
        import time as _t
        deadline = _t.time() + 10
        while _t.time() < deadline:
            if _dismiss_banner():
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
- Default timeout for ALL WebDriverWait calls: 30 seconds. Never use less than 30.
- The site runs on a cloud server that may have slow cold starts. 30s gives it room to respond.
- Use ONE wait per page-load check — do NOT chain multiple waits for the same page.

═══════════════════════════════════════
PAGE LOAD CHECK
═══════════════════════════════════════
- After driver.get(), ALWAYS verify the page is ready with this exact JS readyState check — it works on every site regardless of DOM structure:
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
    )
- This is the ONLY guaranteed page-load check. NEVER use EC.presence_of_element_located on "header", "nav", "main", or any structural element as the primary page-load signal — these elements may not exist or may be injected late by JavaScript.
- After the readyState check passes, you may add ONE additional wait for a specific element that the test actually needs (e.g. a nav link before clicking it). Do not add redundant waits.

═══════════════════════════════════════
LOCATORS — how to find elements
═══════════════════════════════════════
Priority order (most reliable → least reliable): ID → Name → CSS Selector → XPath
Always use the highest-priority locator that works for the element. Only fall back to a lower-priority strategy when a higher one is not available in the HTML.

BY.ID — always prefer By.ID when a stable, semantic id is present. Only use it when the id is a human-readable, semantic name (e.g. "search-btn", "headerMenu", "flashBell"). NEVER use By.ID for ids that look randomly generated — strings of random alphanumeric characters like "r1w2KWYLVsyGg" or "HJH3YbK84ikMe" are build-time dynamic ids that change on every deployment and will break the test.
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "search-btn")))

BY.NAME — for form inputs with a name attribute:
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.NAME, "q")))

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
- For nav links use XPath scoped to the nav container with @href: (By.XPATH, "//nav//a[contains(@href, 'path-stem')]")

NAVIGATION LINKS — <a href> elements MUST be located by href, never by id:
- Many sites (React, Next.js, Angular) generate random ids on <a> elements at build time. These ids look like "r1w2KWYLVsyGg" — they are NOT stable and MUST NOT be used.
- ALWAYS locate <a href> navigation links using a simple EC.element_to_be_clickable call. Scope the XPath to the nav container to avoid matching hidden duplicates in mobile nav or footer:
    link = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//nav//a[contains(@href, '/economy')]"))
    )
- NEVER use lambda+next+find_elements patterns — they are hard to debug and unnecessary when the XPath is scoped correctly.

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
- INTERACTION RULE: before ANY click, send_keys, or clear() — ALWAYS use EC.element_to_be_clickable. NEVER use EC.presence_of_element_located or EC.visibility_of_element_located for an element you are about to interact with — presence/visibility does not guarantee the element accepts input.
    CORRECT:   WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "country-picker-search")))
    INCORRECT: WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "country-picker-search")))
- EC.presence_of_element_located — element is in the DOM (use ONLY for images and read-only checks, never before interaction)
- EC.visibility_of_element_located — element is visible (use ONLY for assertion steps that verify something is shown, never before interaction)
- EC.element_to_be_clickable — element is ready to click (use before every click)
- NEVER use visibility_of on <img> — images have 0×0 size in headless mode. Use presence_of and check .get_attribute("src") or .get_attribute("alt").
- NEVER use assert element.is_displayed() on images.
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
    WebDriverWait(driver, 30).until(
        lambda d: (d.find_elements(By.ID, "panel_id") and d.find_element(By.ID, "panel_id").is_displayed())
            or any(k in d.current_url.lower() for k in ["login", "signin", "account"])
    )
- NEW TAB: If the HTML shows target="_blank", the click opens a new tab. Capture handles before the click and switch:
    original_handles = driver.window_handles
    link.click()
    WebDriverWait(driver, 30).until(lambda d: len(d.window_handles) > len(original_handles))
    driver.switch_to.window(driver.window_handles[-1])
- If target is "_top", "_self", or absent: link opens in the same tab. Assert len(driver.window_handles) == 1 after clicking.
- NEVER use EC.url_contains("/") — every URL contains "/" and this proves nothing.

═══════════════════════════════════════
HAMBURGER / COLLAPSIBLE NAVIGATION
═══════════════════════════════════════
Many sites (IMDB, BBC, etc.) hide ALL navigation links inside a collapsible drawer behind a hamburger/MENU toggle. These links exist in the DOM but are NOT interactable until the drawer is opened — EC.element_to_be_clickable on the nav link will time out if you skip this step.

MANDATORY 3-STEP PATTERN for any navigation link that lives inside a drawer:
  Step 1 — Open the menu. ALWAYS follow ID > Name > CSS priority — check the HTML for an id on the toggle first:
    # STEP 1a — look for an id attribute on the toggle button in the HTML. If found, use By.ID:
    menu_btn = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.ID, "exact-id-from-html"))
    )
    # STEP 1b — only if no id exists, use this multi-selector CSS fallback. NEVER invent a single aria-label:
    menu_btn = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,
            "label[for*='navDrawer'], label[for*='nav-drawer'], "
            "[aria-label*='menu' i], [aria-label*='navigation' i], "
            "[class*='hamburger'], [class*='menu-toggle'], [class*='nav-toggle'], "
            "button[aria-expanded]"
        ))
    )
    menu_btn.click()
    time.sleep(1.5)

  Step 2 — Locate the nav link now that the drawer is open (XPath @href):
    nav_link = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/target-path')]"))
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
- Some sites (e.g. BBC) place the search input inside a hidden drawer or slide-out panel. The input exists in the DOM but has disabled="" or is not interactable until the drawer is opened.
- NEVER use EC.presence_of_element_located for a search input you intend to type into — it only checks DOM presence, not interactability.
- If a search input throws ElementNotInteractableException, it is hidden inside a panel. You MUST click the toggle button that opens the panel first, then wait for the input with EC.element_to_be_clickable.
- Pattern:
    # 1. Open the panel/drawer that contains the search input
    menu_toggle = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Open menu']"))
    )
    menu_toggle.click()
    time.sleep(1.5)
    # 2. Now wait for the input to be truly interactable
    search_input = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@data-testid='search-input-field']"))
    )

═══════════════════════════════════════
HOVER DROPDOWNS
═══════════════════════════════════════
- To reveal a dropdown without navigating: hover with ActionChains AND dispatch a JS mouseover event (headless Chrome does not always fire CSS :hover from synthetic moves):
    ActionChains(driver).move_to_element(el).pause(0.5).perform()
    driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));", el)
    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[class*='dropdown']")))
- Click a nav link only when the step explicitly navigates to a new page.

═══════════════════════════════════════
FORM / SEARCH SUBMIT
═══════════════════════════════════════
- ALWAYS submit search forms using Keys.ENTER on the input field — never by clicking the search button. In headless Chrome, the autocomplete dropdown covers the search button after typing, making it unclickable. Keys.ENTER works in both headless and non-headless.
  from selenium.webdriver.common.keys import Keys
  search_input.send_keys("query text")
  time.sleep(1.5)
  search_input.send_keys(Keys.ENTER)
- For non-search forms (login, contact, filters): scroll the submit button into view, then use a JS click to avoid ElementClickInterceptedException from sticky headers or overlays:
  submit_btn = WebDriverWait(driver, 30).until(
      EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit'], button[type='submit']"))
  )
  driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", submit_btn)
  time.sleep(0.5)
  driver.execute_script("arguments[0].click();", submit_btn)
- Clicking submit WITHOUT input does not navigate. Do NOT use any url_* condition. Wait for the form input to still be visible.

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

def extract_full_html(url: str) -> str:
    if url in _html_cache:
        return _html_cache[url]
    import requests as _requests
    lang = "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7" if _is_israeli_site(url) else "en-US,en;q=0.9"
    resp = _requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Accept-Language": lang,
    })
    resp.raise_for_status()
    html = _clean_html(resp.text)
    _html_cache[url] = html
    return html

# def generate_selenium_code(step_text, expected_text, website, page_html):
#     """Generate Python Selenium code for a single step using AI."""
#     llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=ANTHROPIC_API_KEY)
#     messages = prompt_template.format_messages(
#         step=step_text, expected=expected_text, website=website, html=page_html
#     )
#     import sys as _sys, time as _t
#     print(f"[anthropic] generate_selenium_code.invoke at {_t.time()} (executor.py)", flush=True, file=_sys.stderr)
#     response = llm.invoke(messages)
#     return response.content.strip()

# -----------------------------
# 2. Generate test files from plan (old per-step approach, replaced by generate_test_files)
# -----------------------------
# def generate_test_files(plan):
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     test_files = []
#
#     for case in plan["cases"]:
#         case_id = case["id"]
#         steps = case.get("steps", [])
#         expected = case.get("expected", "")
#         all_code = []
#
#         for step in steps:
#             code = generate_selenium_code(step, expected)
#             all_code.append(code)
#
#         file_path = os.path.join(OUTPUT_DIR, f"{case_id}.py")
#         with open(file_path, "w", encoding="utf-8") as f:
#             f.write("\n\n".join(all_code))
#
#         test_files.append((case_id, file_path))
#         print(f"✅ Generated test file: {file_path}")
#
#     return test_files

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
        opts = Options()
        if not _using_lambdatest and _os.environ.get("HEADLESS", "true").lower() != "false":
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
            _base_args += ["--blink-settings=imagesEnabled=false", "--disable-software-rasterizer", "--renderer-process-limit=1"]
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
        else:
            print("[driver] Using local Chrome (LambdaTest credentials not set)", flush=True)
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
            "//*[(self::button or self::a or self::div or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept all')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept cookies')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'i agree')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it')]",
            # Accept text (Hebrew)
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'אישור')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'אשר')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'מאשר')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'אני מסכים')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'אני מאשר')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'קבל הכל')]",
            "//*[(self::button or self::a or self::div or self::span)][contains(., 'קבל את כל')]",
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
            end = time.time() + (8 if _using_lambdatest else 3)
            while time.time() < end:
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
                time.sleep(0.2)
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
                \"\"\")
            except Exception:
                pass
        _orig_get = driver.get
        def _patched_get(url):
            _orig_get(url)
            # Allow WAF/Cloudflare JS challenges to run before the test starts waiting.
            # LambdaTest remote sessions have extra latency — consent modals load later.
            time.sleep(6 if _using_lambdatest else 3)
            _dismiss_cookies(driver)
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
