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
    opts.add_argument("--blink-settings=imagesEnabled=false")
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
_MAIN_BLOCK_TEMPLATE = '''
if __name__ == "__main__":
    import os, time
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    opts = Options()
    opts.page_load_strategy = "none"
    if os.environ.get("HEADLESS", "true").lower() != "false":
        opts.add_argument("--headless=new")
    for _arg in [
        "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        "--disable-extensions", "--no-first-run", "--disable-background-networking",
        "--disable-sync", "--disable-default-apps",
        "--blink-settings=imagesEnabled=false",
        "--disable-blink-features=AutomationControlled",
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "--window-size=1440,900",
    ]:
        opts.add_argument(_arg)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    _driver = webdriver.Chrome(options=opts)
    _driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
        Object.defineProperty(navigator, \'webdriver\', {get: () => undefined});
        if (!window.chrome) window.chrome = {};
        if (!window.chrome.runtime) window.chrome.runtime = {};
    """})
    _driver.set_script_timeout(50)
    _orig = _driver.get
    def _pg(url):
        _orig(url)
        time.sleep(3)
    _driver.get = _pg
    try:
        run(_driver)
        print("RESULT: Pass")
    except Exception as _e:
        import traceback; traceback.print_exc()
    finally:
        _driver.quit()
'''

SYSTEM_PROMPT_TEMPLATE = """You are a senior QA automation engineer. Generate ONE Python Selenium script implementing all steps of this test case.

OUTPUT FORMAT — follow exactly:
1. Module-level imports (selenium, WebDriverWait, By, EC, time; add ActionChains or Keys only if needed).
2. A single `def run(driver):` function containing ALL test steps. No try/except inside run() — let exceptions propagate naturally.
3. STOP after the closing line of def run(). Do NOT write an if __name__ == "__main__": block — it is appended automatically by the backend.

The backend calls run(driver) with an already-configured driver. NEVER call webdriver.Chrome(), driver.quit(), or driver.close() inside run().
- Call driver.get("{website}") exactly ONCE at the start of run().

STEP FIDELITY (most important):
- The Steps list is authoritative. Implement EXACTLY those steps, in order, with their stated intent.
- One numbered step → one identifiable code block. Do not skip, merge, reorder, or invent steps.

LOCATORS:
- Use ONLY attributes present in the provided HTML. Never guess.
- Priority: ID > Name > CSS Selector > XPath with visible text.
- For non-Latin text (Hebrew etc.), match the exact visible text from the HTML.
- For href XPath, use the FULL href value from the HTML (e.g. contains(@href, 'myactivity.google.com/privacyadvisor') — not a guessed substring).
- For ANY clickable link that has visible text: ALWAYS use By.PARTIAL_LINK_TEXT. This is NON-NEGOTIABLE. NEVER use href, title, or class attributes to locate a link you intend to click — the rendered href often differs from the HTML snapshot (absolute vs relative, redirects, query params). Visible text is always stable.
- NAVIGATION BAR LINKS — IMPORTANT EXCEPTION: By.PARTIAL_LINK_TEXT only matches <a> anchor tags. Nav items often use UI frameworks (Vuetify, MUI) where the <a> wraps multiple inner <span>s that have `pointer-events: none` in CSS — so element_to_be_clickable on an inner span always fails even though it is visible. The ONLY reliable pattern: target the <a> element itself using XPath with `contains(@href, 'keyword')` scoped to the nav container. XPath @href reads the raw HTML attribute (e.g. "/categories.aspx"), NOT the browser-resolved absolute URL, so it is stable. Use a short path stem that covers all URL variants: `(By.XPATH, "//*[@id='navContainerID']//a[contains(@href, 'path-stem')]")`. Example: `(By.XPATH, "//*[@id='headerMenu']//a[contains(@href, 'categor')]")` matches both /categories.aspx and /categories/. NEVER target inner <span>s inside a nav <a> — they are not independently clickable.
- NEVER use By.LINK_TEXT — it breaks on nested spans or extra whitespace. By.PARTIAL_LINK_TEXT only (or XPath for nav-bar items as above).
- NEVER write CSS selectors like a[href='...'] or a[href*='...'][title='...'] for clickable links. These fail whenever the rendered href differs from the static HTML.
- NEVER use a[title='...'] as a selector for a clickable link — title attributes are unreliable in the rendered DOM. ALWAYS use By.PARTIAL_LINK_TEXT with the visible text instead. BAD: `(By.CSS_SELECTOR, "#headerMenu a[title='כל הקטגוריות']")` — GOOD: `(By.PARTIAL_LINK_TEXT, "כל הקטגוריות")`.
- NEVER use CSS selectors with href (e.g. NEVER `"#headerMenu a[href='/categories.aspx']"`). CSS [href=] resolves to the absolute URL in the rendered DOM even when the HTML shows a relative path. XPath @href is different — it reads the raw attribute — and IS allowed.
- When asserting .get_attribute("alt") or any attribute value: NEVER use exact equality (==). Always use `in` with a short keyword — e.g. `assert "דרושים" in (alt or "")` — because attributes may be None or slightly different across environments.
- Only fall back to CSS_SELECTOR on href when the link has NO visible text at all (e.g. icon-only links).
- NEVER combine href with any other attribute (class, title, etc.) in a selector.
- For data-test / data-testid use By.CSS_SELECTOR, "[data-testid='x']" — there is no By.DATA_TESTID.
- Allowed By strategies only: ID, NAME, CLASS_NAME, TAG_NAME, CSS_SELECTOR, XPATH, LINK_TEXT, PARTIAL_LINK_TEXT.
- NEVER assert button/input value attributes — they vary by locale.

WAITS AND ASSERTIONS:
- Default timeout for ALL WebDriverWait calls is 20 seconds. Never use less than 20.
- For the INITIAL page-load check (verifying the page opened): use EC.presence_of_element_located — the element only needs to be in the DOM, not yet painted.
- For interactions use EC.element_to_be_clickable; for subsequent read/assert steps use EC.visibility_of_element_located.
- NEVER use EC.visibility_of_element_located on <img> elements — images are disabled in headless mode so img tags have 0×0 dimensions and are never "visible". Use EC.presence_of_element_located for images and verify via .get_attribute("alt") or .get_attribute("src") instead.
- NEVER use assert element.is_displayed() on <img> elements for the same reason.
- For page title checks: use `assert "keyword" in driver.title` with a SHORT unique keyword, never the full exact title string (titles vary by locale, A/B test, or dash character encoding).
- Never pair EC.presence_of_element_located with a stronger condition on the same element — use only the strongest.
- FORBIDDEN dead waits: WebDriverWait for By.TAG_NAME 'body', 'html', or 'main' — these tags may not exist and add no signal after a URL check.
- FORBIDDEN redundant DOM check after url_contains: once EC.url_contains() passes, the page navigation is confirmed. Do NOT add another WebDriverWait for a generic element (main, div, section) immediately after — it adds failure surface with no benefit.
- After every click/submit, wait for a SPECIFIC element that reflects what changed (the destination heading, an updated form, the same input still visible if no navigation expected).
- NO REDUNDANT ASSERTS: WebDriverWait IS the assertion. Do not write `assert driver.title == "X"` after `EC.title_is("X")` — same for url_contains, visibility_of, etc.
- Use plain `assert` ONLY for .text / .get_attribute() content that no EC checks.
- FORBIDDEN: capturing old_url and using EC.url_changes / EC.url_to_be. Use EC.url_contains(fragment).

NAVIGATION:
- After clicking a link, verify with EC.url_contains using a fragment TAKEN FROM THE href, not from the link's visible text. (e.g. href="https://mail.google.com/..." → wait for "mail.google.com", not "gmail".)
- NON-ASCII IN URLS: driver.current_url always returns percent-encoded URLs. NEVER pass raw non-ASCII text (Hebrew, Arabic, etc.) to EC.url_contains — it will never match. Always encode it first: `from urllib.parse import quote` then `EC.url_contains(quote("עבודה", safe=""))`. Alternatively, use a path fragment from the href that contains only ASCII characters (e.g. "/jobs/search/" instead of the Hebrew search term).
- If the link has target="_top" or no target attribute, also assert that window_handles length is unchanged. If target="_blank" is set, skip that assertion.

URL VERIFICATION AFTER NAV CLICK:
- After clicking a nav link, NEVER hard-code the exact URL path from the HTML snapshot (e.g. NEVER `EC.url_contains("categories.aspx")`). The deployed server may be in a different region or IP, causing the site to redirect to a different URL variant (e.g. "/categories/" instead of "/categories.aspx").
- ONLY wait for a URL change or new tab when the link's href points to a DIFFERENT page than the current one. If the link points to the same page (e.g. a logo that links to "/" when you are already on "/"), clicking it will NOT change the URL and will NOT open a new tab — waiting for either will always time out. For same-page links, skip the URL check entirely and instead verify a page element is still present after the click.
- NEW TAB WARNING: If the HTML shows `target` other than `_self` or empty (e.g. `target="self"`, `target="_blank"`, `target="main"`), AND the href points to a different page, the click may open a new tab and `driver.current_url` will NOT change in the original window. Capture window handles before the click and switch to the new tab if one appears:
    original_handles = driver.window_handles
    original_url = driver.current_url
    link.click()
    WebDriverWait(driver, 20).until(lambda d: d.current_url != original_url or len(d.window_handles) > len(original_handles))
    if len(driver.window_handles) > len(original_handles):
        driver.switch_to.window(driver.window_handles[-1])
- NEVER use EC.url_contains("/") — every URL contains "/" so this check always passes immediately and proves nothing.
- If a partial keyword check is needed (to confirm you landed on the right section), use a very short stem that covers all URL variants: e.g. `EC.url_contains("categor")` covers both "categories.aspx" and "/categories/".

EMPTY SUBMIT:
- Clicking submit/search WITHOUT entering input does not navigate. Do NOT use any url_* condition. Instead wait that the form input is still visible.

COLLAPSIBLE / HAMBURGER NAVIGATION:
- Before clicking any link inside a navigation menu, inspect the HTML to check whether the nav is inside a collapsible element (hamburger button, "MENU" toggle, aria-expanded, aria-controls, or a button that controls a nav panel).
- If such a toggle exists, ALWAYS click it first to open the nav, then wait for the nav links to become visible, THEN click the target link. Skipping the toggle open step will cause a TimeoutException because the links are hidden.
- Example pattern:
      menu_toggle = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-controls='nav'], .hamburger, [class*='menu-toggle']")))
      menu_toggle.click()
      WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "nav a")))
      # now click the target link

HOVER DROPDOWNS:
- Nav items that are <a> with real hrefs are usually BOTH a link and a hover trigger. Clicking navigates and destroys the dropdown.
- For "verify dropdown / submenu / subcategories" steps, hover with ActionChains — do NOT click. Also dispatch a JS mouseover as a backup, because synthetic mouse moves in headless Chrome don't always trip CSS :hover:
      ActionChains(driver).move_to_element(link).pause(0.5).perform()
      driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));", link)
      WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".dropdown")))
- Click the nav <a> only when the step explicitly says to navigate to that destination page.
- For headless Chrome, call driver.set_window_size(1440, 900) BEFORE driver.get() so the server returns the desktop (hover-menu) layout, not the mobile hamburger.

VERIFY-LOAD ECONOMY:
- "Page loaded" verification needs ONE wait, not many. Pick a single high-signal element (the page header, the main nav, or the hero) and wait for its visibility. Adding redundant waits on header_wrapper, header_content, header_strip, nav_list, page_wrapper etc. multiplies latency without adding signal — each wait can sit near its timeout ceiling, and 9 of them at 10s each will easily exceed the per-test budget.
- BAD (multiplies failure surface):
      WebDriverWait(...).until(EC.visibility_of(...header_wrapper...))
      WebDriverWait(...).until(EC.visibility_of(...header_content...))
      WebDriverWait(...).until(EC.visibility_of(...header_strip...))
      [...7 more...]
- GOOD (one decisive check — presence_of for initial load, 20s timeout):
      WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#page-header-navigation")))

ERROR HANDLING:
- Wrap the entire test body in try/except.
- On any exception, save a screenshot to "error_{{int(time.time())}}.png", then re-raise. Do not swallow exceptions.

BROWSER MODE: {headless}
- HTML below was fetched using {headless}. Only use locators that exist in this HTML.

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

def extract_full_html(url: str) -> str:
    if url in _html_cache:
        return _html_cache[url]
    import requests as _requests
    resp = _requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
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
        if not c or "def run(" not in c and "def run (" not in c:
            return "missing def run(driver) function"
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
        f"Return ONLY the corrected Python script with no explanation and no markdown fences."
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
        main_block=_MAIN_BLOCK_TEMPLATE,
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

        # Strip any __main__ block Claude added despite instructions — we append
        # the correct one below so the standalone runner is always complete.
        if "if __name__" in code:
            code = code[:code.index("if __name__")].rstrip()

        # Always append the canonical __main__ block so the file works both
        # as a standalone script and via exec() in the subprocess runner.
        code = code + "\n\n" + _MAIN_BLOCK_TEMPLATE

        # Validate structure + syntax; ask Claude to fix once if broken.
        code = _fix_syntax(code, llm, cached_system, case_id)
        if code is None:
            print(f"[generate_test_files] Skipping {case_id}: could not produce valid Python after retry.")
            continue

        file_path = os.path.join(OUTPUT_DIR, f"{case_id}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        test_files.append((case_id, file_path))

    return test_files

# -----------------------------
# 3. Run generated Selenium tests
# -----------------------------
def run_test_file(case_id, file_path):
    import subprocess, sys, textwrap

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
        for arg in [
            "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-gpu", "--disable-extensions",
            "--no-first-run", "--disable-background-networking",
            "--disable-sync", "--disable-default-apps",
            "--blink-settings=imagesEnabled=false",
            "--disable-blink-features=AutomationControlled",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "--window-size=1440,900",
        ]:
            opts.add_argument(arg)
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
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
                    Object.defineProperty(navigator, 'languages', {{get: () => ['he-IL', 'he', 'en-US', 'en']}});
                    Object.defineProperty(navigator, 'platform', {{get: () => 'Win32'}});
                    const _origPerms = navigator.permissions.query.bind(navigator.permissions);
                    navigator.permissions.query = (p) =>
                        p.name === 'notifications'
                            ? Promise.resolve({{state: 'default', onchange: null}})
                            : _origPerms(p);
                '''}})
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
            _ns = {{'__name__': 'captainfix_runner'}}
            exec(code, _ns)
            _ns['run'](driver)
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
            time.sleep(3)  # let Chrome fully exit before subprocess ends
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

    # 2. Execute all test files and collect results
    results = []
    for case_id, file_path in test_files:
        print(f"▶ Running test {case_id} ...")
        result = run_test_file(case_id, file_path)
        results.append(result)
        print(f"✔ {case_id} → {result['status']}")

    # 3. Save structured results
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Finished! Results saved in {RESULTS_JSON}")


if __name__ == "__main__":
    main()
