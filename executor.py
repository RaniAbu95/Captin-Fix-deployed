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

# llm is initialised inside generate_selenium_code() to avoid import-time errors

# prompt_template = ChatPromptTemplate.from_template("""
# You are an expert QA engineer.
# Convert the following test step into **runnable Python Selenium code**.
#
# Rules:
# - Use `driver` for the WebDriver.
# - Use `By` from selenium.
# - Include all necessary imports.
# - Include driver.quit() if this is the last step.
# - Do NOT include markdown or backticks.
# - The step should be executable standalone in a Python file.
#
# Step: {step}
# Expected result: {expected}
# """)

# prompt_template = ChatPromptTemplate.from_template("""
# You are an expert QA engineer.
#
# Convert the following test step into **runnable Python Selenium code** using the provided `driver`.
#
# Rules:
# - Use the most reliable locator: ID > Name > CSS Selector > XPath.
# - Use `WebDriverWait` to wait for elements before interacting.
# - Do NOT re-create the driver; use the existing `driver`.
# - Include necessary imports if they are missing.
# - The code must be executable and independent for this step.
# - Do NOT include markdown or backticks.
# - Capture a screenshot if the element is not found or an action fails.
#
# Step: {step}
# Expected result: {expected}
# """)

# Per-step prompt (kept for reference, no longer used in generate_test_files)
# prompt_template = ChatPromptTemplate.from_template("""
# You are an expert QA engineer.
#
# Convert the following test step into **runnable Python Selenium code** using the provided `driver`.
#
# Rules:
# - If this is the first step in the test case, include:
#     driver.get("{website}")
#   before interacting with the page.
#
# - You are also given the full HTML of the page. You MUST extract and use the **exact attributes** from the HTML (id, name, placeholder, value, visible text, class if unique).
# - Do NOT invent or assume element IDs or names. Only use selectors that actually appear in the HTML.
# - Priority for locators: ID > Name > Placeholder/Text > CSS Selector > XPath.
# - If no clean locator exists, construct an XPath using visible text or hierarchy from the given HTML.
# - Only use locators that appear exactly in the provided HTML.
# - For Hebrew or non‑Latin text, match the exact visible text from the HTML, preserving spaces, punctuation, and case.
# - When navigating to a file:/// URL, assume Chrome is launched with options to allow local file access and disable web security.
# - Never call driver.quit(), driver.close(), or end the browser session in any way.
# - Output only raw Python code, no markdown, no backticks, no explanations.
# - Use `WebDriverWait` for all element interactions.
# - Do NOT re-create the driver; use the existing `driver` passed in.
# - **Never** call driver.quit(), driver.close(), or end the browser session in any way.
# - Do NOT navigate away from the provided Website URL unless explicitly stated in the step.
# - Include necessary imports if they are missing.
# - The code must be executable and independent for this step.
# - Do NOT include markdown or backticks.
# - The step should be executable standalone in a Python file.
# - Capture a screenshot if the element is not found or an action fails, but do not terminate the driver afterwards.
#
# Website URL: {website}
#
# Full HTML: {html}
#
# Step: {step}
# Expected result: {expected}
# """)

# System portion — static across all cases in a run (instructions, website, HTML).
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
    if os.environ.get("HEADLESS", "true").lower() != "false":
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

    _orig_get = driver.get
    def _patched_get(url):
        _orig_get(url)
        time.sleep(5)
        for sel in ["#CybotCookiebotDialogBodyButtonClose", ".cky-btn-accept",
                    "[id*='accept-all']", "[id*='acceptAll']"]:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in els:
                    if el.is_displayed() and el.is_enabled():
                        el.click()
                        return
            except Exception:
                pass
        for xpath in [
            "//*[(self::button or self::a or self::span)][contains(.,'אישור')]",
            "//*[(self::button or self::a or self::span)][contains(.,'קבל הכל')]",
            "//*[(self::button or self::a or self::span)][normalize-space()='\xd7']",
        ]:
            try:
                els = driver.find_elements(By.XPATH, xpath)
                for el in els:
                    if el.is_displayed() and el.is_enabled():
                        el.click()
                        return
            except Exception:
                pass
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
STEP FIDELITY
═══════════════════════════════════════
- Implement EXACTLY the steps listed, in order. Do not skip, merge, reorder, or invent steps.
- One test plan step → one clearly identifiable code block.

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
Priority order: ID → Name → CSS Selector → XPath

LINKS WITH VISIBLE TEXT:
- Use By.PARTIAL_LINK_TEXT with the exact visible text from the HTML. This is the most stable locator.
- NEVER use href, title, or class to locate a clickable link.

NAV BAR LINKS (exception to above):
- UI frameworks often wrap nav links in <span> children with pointer-events:none. PARTIAL_LINK_TEXT on a span always fails.
- For nav links, use XPath scoped to the nav container: (By.XPATH, "//nav//a[contains(@href, 'path-stem')]")
- Use a short path stem so it matches URL variants: contains(@href, 'about') matches /about, /about-us, /about/team.
- XPath @href reads the raw HTML attribute — use it as written, never percent-encoded.

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
- NEVER use CSS [href='...'] or [href*='...'] for clickable links — CSS resolves to absolute URLs, XPath does not.
- NEVER use a[title='...'] — title attributes are unreliable in the live DOM.
- NEVER combine href with class or title in any selector.
- For data-testid: (By.CSS_SELECTOR, "[data-testid='value']")
- NEVER assert button/input value attributes — they vary by locale.
- When asserting any .get_attribute() value: use `in`, never `==`. Attributes may be None or slightly different across environments.

═══════════════════════════════════════
WAITS AND ASSERTIONS
═══════════════════════════════════════
- EC.presence_of_element_located — element is in the DOM (use for initial page load and images)
- EC.visibility_of_element_located — element is visible (use for interactions and read steps)
- EC.element_to_be_clickable — element is ready to click (use before every click)
- NEVER use visibility_of on <img> — images have 0×0 size in headless mode. Use presence_of and check .get_attribute("src") or .get_attribute("alt").
- NEVER use assert element.is_displayed() on images.
- For title checks: assert "keyword" in driver.title — use a short keyword, never the full title string.
- WebDriverWait IS the assertion — do not add a redundant assert after an EC condition checks the same thing.
- Use plain assert ONLY for .text or .get_attribute() values not covered by any EC.
- FORBIDDEN: EC.url_changes, EC.url_to_be — use EC.url_contains(fragment) instead.
- FORBIDDEN: WebDriverWait on body, html, or main tag — these add no signal.
- FORBIDDEN: redundant DOM check immediately after EC.url_contains passes.

═══════════════════════════════════════
NAVIGATION
═══════════════════════════════════════
- After clicking a link, verify arrival with EC.url_contains using a fragment from the href value — not the link's visible text.
- NEVER hard-code the exact URL path (e.g. "/page.aspx") — deployed sites may redirect to a different variant ("/page/"). Use a short stem: contains("page") matches both.
- ONLY wait for a URL change when the href actually points to a different page. If the link points to the current page (e.g. logo → "/"), do not wait for a URL change — verify a page element instead.
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
- If the nav is inside a collapsible panel (hamburger, MENU toggle, aria-expanded button), click the toggle first, wait for the links to appear, then click the target link.
- Pattern:
    toggle = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-controls], [class*='menu-toggle'], [class*='hamburger']")))
    toggle.click()
    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "nav a")))

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
- Clicking submit WITHOUT input does not navigate. Do NOT use any url_* condition. Wait for the form input to still be visible.

═══════════════════════════════════════
COOKIE BANNERS
═══════════════════════════════════════
- The test runner auto-dismisses cookie/consent banners after every driver.get(). Do NOT write code for this.

═══════════════════════════════════════
ERROR HANDLING
═══════════════════════════════════════
- The test body is already inside the try block (see OUTPUT FORMAT).
- In the except block: save a screenshot to "error_{{int(time.time())}}.png", then re-raise. Never swallow exceptions.

BROWSER MODE: {headless}
Only use locators that exist in the HTML below — fetched with {headless}.

Website URL: {website}

Full HTML: {html}"""

# Per-case portion — varies per case, NOT cached.
USER_PROMPT_TEMPLATE = """Steps:
{steps}

Expected result: {expected}"""



# def generate_selenium_code(step_text, expected_text):
#     """Generate Python Selenium code for a single step using AI."""
#     messages = prompt_template.format_messages(step=step_text, expected=expected_text)
#     response = llm.invoke(messages)
#     return response.content.strip()
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

def generate_selenium_code(step_text, expected_text, website, page_html):
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        api_key=ANTHROPIC_API_KEY
    )
    messages = prompt_template.format_messages(
        step=step_text,
        expected=expected_text,
        website=website,
        html=page_html
    )
    import sys as _sys, time as _t
    print(f"[anthropic] generate_selenium_code.invoke at {_t.time()} (executor.py)", flush=True, file=_sys.stderr)
    response = llm.invoke(messages)
    return response.content.strip()

# -----------------------------
# 2. Generate test files from plan
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_files = []
    website = plan.get("website", "")

    page_html = extract_full_html(website)
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=ANTHROPIC_API_KEY)

    # Build the system prompt ONCE per run — same instructions, website, and HTML
    # are reused for every case, so this whole block is the cache prefix.
    system_text = SYSTEM_PROMPT_TEMPLATE.format(
        website=website,
        html=page_html,
        headless="headless Chrome" if HEADLESS else "regular Chrome",
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
        import sys as _sys, time as _t
        print(f"[anthropic] generate_test_files.invoke case={case_id} at {_t.time()} (executor.py)", flush=True, file=_sys.stderr)
        response = llm.invoke(messages)
        code = _strip_code_fences(response.content.strip())

        # Strip any __main__ block Claude added despite instructions (we append our own).
        if "if __name__" in code:
            code = code[:code.index("if __name__")].rstrip()

        # Validate structure + syntax; ask Claude to fix once if broken.
        code = _fix_syntax(code, llm, cached_system, case_id)
        if code is None:
            print(f"[generate_test_files] Skipping {case_id}: could not produce valid Python after retry.")
            continue

        file_path = os.path.join(OUTPUT_DIR, f"{case_id}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code + "\n" + _make_main_block(website))

        test_files.append((case_id, file_path))

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
        opts = Options()
        if _os.environ.get("HEADLESS", "true").lower() != "false":
            opts.add_argument("--headless=new")
        _israeli = {_israeli}
        _base_args = [
            "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-gpu", "--disable-extensions",
            "--no-first-run", "--disable-background-networking",
            "--disable-sync", "--disable-default-apps",
            "--disable-blink-features=AutomationControlled",
            "--blink-settings=imagesEnabled=false",
            "--disable-software-rasterizer",
            "--renderer-process-limit=1",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "--window-size=1440,900",
        ]
        if _israeli:
            _base_args.append("--lang=he-IL")
        for arg in _base_args:
            opts.add_argument(arg)
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        _lang_pref = "he,he-IL,en-US,en" if _israeli else "en-US,en"
        opts.add_experimental_option("prefs", {{"intl.accept_languages": _lang_pref}})
        opts.page_load_strategy = 'none'

        driver = None
        for attempt in range(3):
            try:
                driver = webdriver.Chrome(options=opts)
                # Internal Selenium timeouts intentionally < subprocess
                # wall-clock (60s) so the inner except has room to write
                # the screenshot before SIGKILL.
                driver.set_script_timeout(8)
                # Comprehensive stealth: patch every property sites use to detect headless Chrome.
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {{"source": '''
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
                '''}})
                if _israeli:
                    driver.execute_cdp_cmd("Emulation.setGeolocationOverride", {{
                        "latitude": 31.7683, "longitude": 35.2137, "accuracy": 100
                    }})
                break
            except Exception:
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise

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
            end = time.time() + 3
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
            time.sleep(3)
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
    for case_id, file_path in test_files:
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
