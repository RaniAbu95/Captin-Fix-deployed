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
SYSTEM_PROMPT_TEMPLATE = """You are a senior QA automation engineer. Generate ONE Python Selenium script implementing all steps of this test case using the provided `driver`.

OUTPUT:
- Raw Python only — no markdown, no backticks, no commentary.
- Start with imports (selenium, WebDriverWait, By, EC, time; add ActionChains or Keys only if needed).
- `driver` is already provided in scope. NEVER call webdriver.Chrome(), driver.quit(), or driver.close().
- Call driver.get("{website}") exactly ONCE at the start.

STEP FIDELITY (most important):
- The Steps list is authoritative. Implement EXACTLY those steps, in order, with their stated intent.
- One numbered step → one identifiable code block. Do not skip, merge, reorder, or invent steps.

LOCATORS:
- Use ONLY attributes present in the provided HTML. Never guess.
- Priority: ID > Name > CSS Selector > XPath with visible text.
- For non-Latin text (Hebrew etc.), match the exact visible text from the HTML.
- For href XPath, use the FULL href value from the HTML (e.g. contains(@href, 'myactivity.google.com/privacyadvisor') — not a guessed substring).
- For links, PREFER visible text over href: //a[normalize-space()='Forgot password?'] is more stable than //a[contains(@href,'/recover/initiate/')] because URLs change.
- For data-test / data-testid use By.CSS_SELECTOR, "[data-testid='x']" — there is no By.DATA_TESTID.
- Allowed By strategies only: ID, NAME, CLASS_NAME, TAG_NAME, CSS_SELECTOR, XPATH, LINK_TEXT, PARTIAL_LINK_TEXT.
- NEVER assert button/input value attributes — they vary by locale.

WAITS AND ASSERTIONS:
- For interactions use EC.element_to_be_clickable; for read/assert use EC.visibility_of_element_located.
- Never pair EC.presence_of_element_located with a stronger condition on the same element — use only the strongest.
- FORBIDDEN dead waits: WebDriverWait for By.TAG_NAME 'body' or 'html'. Both always exist.
- After every click/submit, wait for a SPECIFIC element that reflects what changed (the destination heading, an updated form, the same input still visible if no navigation expected).
- NO REDUNDANT ASSERTS: WebDriverWait IS the assertion. Do not write `assert driver.title == "X"` after `EC.title_is("X")` — same for url_contains, visibility_of, etc.
- Use plain `assert` ONLY for .text / .get_attribute() content that no EC checks.
- FORBIDDEN: capturing old_url and using EC.url_changes / EC.url_to_be. Use EC.url_contains(fragment).

NAVIGATION:
- After clicking a link, verify with EC.url_contains using a fragment TAKEN FROM THE href, not from the link's visible text. (e.g. href="https://mail.google.com/..." → wait for "mail.google.com", not "gmail".)
- If the link has target="_top" or no target attribute, also assert that window_handles length is unchanged. If target="_blank" is set, skip that assertion.

EMPTY SUBMIT:
- Clicking submit/search WITHOUT entering input does not navigate. Do NOT use any url_* condition. Instead wait that the form input is still visible.

HOVER DROPDOWNS:
- Nav items that are <a> with real hrefs are usually BOTH a link and a hover trigger. Clicking navigates and destroys the dropdown.
- For "verify dropdown / submenu / subcategories" steps, hover with ActionChains — do NOT click. Also dispatch a JS mouseover as a backup, because synthetic mouse moves in headless Chrome don't always trip CSS :hover:
      ActionChains(driver).move_to_element(link).pause(0.5).perform()
      driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('mouseover', {bubbles: true}));", link)
      WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".dropdown")))
- Click the nav <a> only when the step explicitly says to navigate to that destination page.
- For headless Chrome, call driver.set_window_size(1440, 900) BEFORE driver.get() so the server returns the desktop (hover-menu) layout, not the mobile hamburger.

VERIFY-LOAD ECONOMY:
- "Page loaded" verification needs ONE wait, not many. Pick a single high-signal element (the page header, the main nav, or the hero) and wait for its visibility. Adding redundant waits on header_wrapper, header_content, header_strip, nav_list, page_wrapper etc. multiplies latency without adding signal — each wait can sit near its timeout ceiling, and 9 of them at 10s each will easily exceed the per-test budget.
- BAD (multiplies failure surface):
      WebDriverWait(...).until(EC.visibility_of(...header_wrapper...))
      WebDriverWait(...).until(EC.visibility_of(...header_content...))
      WebDriverWait(...).until(EC.visibility_of(...header_strip...))
      [...7 more...]
- GOOD (one decisive check):
      WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#page-header-navigation")))

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
    driver = webdriver.Chrome(options=_chrome_options())
    driver.get(url)
    time.sleep(2)
    html = _clean_html(driver.page_source)
    driver.quit()
    _html_cache[url] = html
    return html

def generate_selenium_code(step_text, expected_text, website, page_html):
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=ANTHROPIC_API_KEY
    )
    messages = prompt_template.format_messages(
        step=step_text,
        expected=expected_text,
        website=website,
        html=page_html
    )
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
        response = llm.invoke(messages)
        code = response.content.strip()
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            code = code.strip()

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
        ]:
            opts.add_argument(arg)

        driver = None
        for attempt in range(3):
            try:
                driver = webdriver.Chrome(options=opts)
                # Internal Selenium timeouts intentionally < subprocess
                # wall-clock (60s) so the inner except has room to write
                # the screenshot before SIGKILL.
                driver.set_script_timeout(50)
                driver.set_page_load_timeout(50)
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
        _orig_get = driver.get
        def _patched_get(url):
            _orig_get(url)
            _dismiss_cookies(driver)
            # Progress snapshot so the parent has something on disk even
            # if a later wait hangs past the subprocess wall-clock.
            try:
                driver.save_screenshot(screenshot_path)
            except Exception:
                pass
        driver.get = _patched_get

        screenshot_path = {repr(os.path.join(SCREENSHOT_DIR, case_id + ".png"))}
        import os as _os
        _os.makedirs({repr(SCREENSHOT_DIR)}, exist_ok=True)

        # Background snapshot loop — overwrites the file every 2s so the
        # most-recent page state is always on disk by the time the
        # subprocess wall-clock kill fires, regardless of where the test hangs.
        import threading as _threading
        _stop_snap = _threading.Event()
        def _snapshot_loop():
            while not _stop_snap.is_set():
                try:
                    driver.save_screenshot(screenshot_path)
                except Exception:
                    pass
                if _stop_snap.wait(5):
                    break
        _snap_thread = _threading.Thread(target=_snapshot_loop, daemon=True)
        _snap_thread.start()

        try:
            code = open({repr(file_path)}).read()
            exec(code, {{
                "driver": driver, "By": By,
                "WebDriverWait": WebDriverWait, "EC": EC, "time": time,
            }})
            print("RESULT:Pass")
        except Exception as e:
            try:
                driver.save_screenshot(screenshot_path)
                print(f"SCREENSHOT:{{screenshot_path}}")
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

    result = {"id": case_id, "status": "Pass", "error": None, "screenshot": None}
    with _chrome_lock:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                capture_output=True, text=True, timeout=60
            )
            time.sleep(2)  # ensure Chrome OS cleanup finishes before the next test
            combined = proc.stdout + "\n" + proc.stderr
            for line in combined.splitlines():
                if line.startswith("SCREENSHOT:"):
                    result["screenshot"] = line[len("SCREENSHOT:"):]
                elif line.startswith("RESULT:Pass"):
                    result["status"] = "Pass"
                elif line.startswith("RESULT:Fail:"):
                    result["status"] = "Fail"
                    result["error"] = line[len("RESULT:Fail:"):].strip()
            if result["status"] == "Fail" and not result.get("error"):
                result["error"] = (proc.stderr or proc.stdout or "Unknown error").strip()
        except subprocess.TimeoutExpired:
            result["status"] = "Fail"
            result["error"] = "Test timed out after 60 seconds"
            # If the inner except managed to save a screenshot before the
            # SIGKILL, surface its path even though the stdout marker was
            # cut off by the kill.
            fallback = os.path.join(SCREENSHOT_DIR, case_id + ".png")
            if os.path.exists(fallback):
                result["screenshot"] = fallback
        except Exception as e:
            result["status"] = "Fail"
            result["error"] = str(e)

    # Always surface a screenshot file if one exists on disk — covers
    # passing tests (no SCREENSHOT marker printed) and any race where
    # the marker was cut off mid-print.
    if not result.get("screenshot"):
        fallback = os.path.join(SCREENSHOT_DIR, case_id + ".png")
        if os.path.exists(fallback):
            result["screenshot"] = fallback

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
