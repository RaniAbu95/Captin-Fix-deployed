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

full_case_prompt = ChatPromptTemplate.from_template("""
You are a senior QA automation engineer with 10+ years of experience writing production-grade Selenium tests.

Generate a single complete Python Selenium script for ALL steps of this test case using the provided `driver`.

STRUCTURE RULES:
- Start with all necessary imports (selenium, WebDriverWait, By, EC, time).
- CRITICAL: Do NOT create or initialize a WebDriver. The variable `driver` is already
  provided in scope — do NOT write `driver = webdriver.Chrome()` or any similar line.
- Call driver.get("{website}") exactly ONCE at the very beginning.
- Execute each step in order as a logical sequence — do NOT repeat driver.get() or imports.
- Never call driver.quit() or driver.close().
- Output only raw Python code — no markdown, no backticks, no explanations.
- NEVER use EC.presence_of_element_located followed by EC.element_to_be_clickable or
  EC.visibility_of_element_located for the same element. Use only the stronger condition:
  * Use EC.element_to_be_clickable when you need to interact with an element (click, send_keys).
  * Use EC.visibility_of_element_located when you only need to read or assert on an element.
  * EC.presence_of_element_located is redundant whenever either of the above is used — never pair them.
- STRICTLY FORBIDDEN — delete any of these lines before returning code:
    WebDriverWait(driver, ...).until(EC.visibility_of_element_located((By.TAG_NAME, "body")))
    WebDriverWait(driver, ...).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    WebDriverWait(driver, ...).until(EC.visibility_of_element_located((By.TAG_NAME, "html")))
  body and html are always in the DOM — waiting for them asserts nothing and is dead code.
- After every click or submit, wait for a specific meaningful element that reflects what changed:
    * Page navigates away → wait for a unique visible element on the destination page (from the HTML).
    * Page stays the same (empty submit, no input) → wait for an element confirming you are still on the same page (e.g. the form input is still visible).

BROWSER MODE: {headless}
- The HTML provided was fetched using {headless}.
- Only use locators that exist in the provided HTML — do NOT assume elements present in headless
  Chrome (e.g. id="hplogo") exist in regular Chrome or vice versa.

LOCATOR RULES:
- Use ONLY exact attributes that appear in the provided HTML (id, name, placeholder, visible text, class if unique).
- Priority: ID > Name > CSS Selector > XPath with visible text.
- For Hebrew or non-Latin text, match the exact visible text from the HTML including spaces and punctuation.
- Never invent or guess locators — only use what is in the HTML.
- When matching an href, use the FULL href value exactly as it appears in the HTML.
  BAD: contains(@href, '/history/privacyadvisor')  ← guessed path, may match wrong element
  GOOD: contains(@href, 'myactivity.google.com/privacyadvisor')  ← domain from actual HTML href
- NEVER assert the "value" attribute of a button or input. The value attribute text (e.g. "Google Search")
  varies by language and region and must not be hardcoded in tests.
  BAD (never do this):
      assert button.get_attribute("value") == "Google Search"  ← do NOT add this line ever

NAVIGATION VERIFICATION RULES:
- NEVER use old_url, driver.current_url, EC.url_changes, or EC.url_contains to verify navigation.
  You cannot know in advance what URL a click will produce — guessing causes false failures.
- After clicking a link or button, verify navigation happened by waiting for a visible
  element that is known to exist on the destination page (from the HTML), for example:
      WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))
- NEVER write these patterns (strictly forbidden):
      old_url = driver.current_url          ← FORBIDDEN
      EC.url_changes(old_url)               ← FORBIDDEN
      EC.url_contains("...")                ← FORBIDDEN
      EC.url_to_be("...")                   ← FORBIDDEN
      driver.current_url                    ← FORBIDDEN

SAME TAB VERIFICATION RULE:
- Whenever a link has target="_top" or no target attribute in the HTML, ALWAYS add this check after clicking:
      handles_before = driver.window_handles  # capture BEFORE clicking
      # ... click and wait for destination element ...
      assert len(driver.window_handles) == len(handles_before), "Expected link to open in same tab but a new tab was opened"
- If the link has target="_blank", do NOT add this check — a new tab is expected.

GOOGLE SEARCH BUTTONS RULES:
- On the Google homepage, the "Google Search" (btnK) and "I'm Feeling Lucky" (btnI) buttons are
  hidden by default. They only become visible after the user interacts with the search box.
- ALWAYS click the search input first before waiting for the buttons to be visible:
      search_input = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.NAME, "q")))
      search_input.click()
      google_search_button = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, "btnK")))
      lucky_button = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, "btnI")))
- ALWAYS enter a search term in the search box BEFORE clicking the "I'm Feeling Lucky" button.
- Clicking "I'm Feeling Lucky" without a search term does nothing — the URL will not change and the test will fail.
- Correct order: search_input.click() → send_keys(search_term) → click btnI → EC.url_changes(old_url)

SEARCH RESULT RULES:
- When a step says "click the first search result" or "open the first result":
  1. Import Keys: from selenium.webdriver.common.keys import Keys
  2. Submit the search by pressing Keys.RETURN.
  3. Wait for results: WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#search h3")))
  4. Click the first result and verify the destination page loaded by waiting for its heading:
       first_result = driver.find_element(By.CSS_SELECTOR, "div#search h3")
       first_result.click()
       WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))
  NEVER use url_changes, url_contains, or driver.current_url to verify navigation.

- When a step says "I'm Feeling Lucky":
  1. Click the search input, send_keys the search term, then click btnI.
  2. After clicking, wait for the destination page heading:
       lucky_btn.click()
       WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))

ASSERTION RULES (most important):
- After EVERY user action (click, form submit, navigation, input), verify the outcome using WebDriverWait.
- Use WebDriverWait + expected_conditions for every verification — never check stale state.
- STRICT RULE — NO REDUNDANT ASSERTS: WebDriverWait IS the assertion. It raises TimeoutException if
  the condition is not met within the timeout. Therefore:
    * NEVER write an assert statement on the line after a WebDriverWait that already checks the same thing.
    * This applies to ALL EC conditions: url_contains, url_to_be, title_is, title_contains,
      visibility_of_element_located, presence_of_element_located, element_to_be_clickable, etc.
    * BAD (redundant — never do this):
        WebDriverWait(driver, 10).until(EC.title_is("Google"))
        assert driver.title == "Google"   # ← DELETE THIS, it is already checked above
    * BAD (redundant — never do this):
        WebDriverWait(driver, 10).until(EC.url_contains("mail.google.com"))
        assert "mail.google.com" in driver.current_url   # ← DELETE THIS
    * GOOD — use assert ONLY for .text or .get_attribute() content not covered by WebDriverWait:
        el = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "msg")))
        assert "Welcome" in el.text, f"Expected 'Welcome', got: {{el.text}}"

ERROR HANDLING:
- Wrap the entire test body in try/except.
- On any exception: save a screenshot to "error_{{int(time.time())}}.png", then re-raise.
- Do not swallow exceptions silently.

STEP FIDELITY RULES — most important:
- The Steps list below is the authoritative specification. Your Selenium code must implement
  EXACTLY those steps, in EXACTLY that order, with EXACTLY that intent.
- Each numbered step must map to a visible, identifiable block of code.
- Do NOT add extra steps that are not in the list.
- Do NOT skip or merge steps.
- Do NOT change the intent of a step (e.g. if the step says "click the Login button",
  do not click a different button, do not navigate to a different page first).
- Do NOT invent interactions that are not described in the steps.
- The steps are written in plain English — translate each one directly into Selenium code
  and nothing more.

Website URL: {website}

Full HTML: {html}

Steps:
{steps}

Expected result: {expected}
""")



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
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, api_key=ANTHROPIC_API_KEY)

    for case in plan["cases"]:
        case_id = case["id"]
        steps = case.get("steps", [])
        expected = case.get("expected", "")

        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        messages = full_case_prompt.format_messages(
            website=website,
            html=page_html,
            steps=steps_text,
            expected=expected,
            headless="headless Chrome" if HEADLESS else "regular Chrome",
        )
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
                driver.set_script_timeout(60)
                driver.set_page_load_timeout(60)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise
        screenshot_path = {repr(os.path.join(SCREENSHOT_DIR, case_id + ".png"))}
        import os as _os
        _os.makedirs({repr(SCREENSHOT_DIR)}, exist_ok=True)
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
        except Exception as e:
            result["status"] = "Fail"
            result["error"] = str(e)

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
