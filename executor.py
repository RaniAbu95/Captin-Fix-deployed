import os
import json
import traceback
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from langchain_openai import ChatOpenAI
from webdriver_manager.chrome import ChromeDriverManager
import time
from config import OPENAI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

PLAN_FILE = "./output/plan.json"
OUTPUT_DIR = "tests"
RESULTS_JSON = "Results.json"
SCREENSHOT_DIR = "./screen/screenshots"

_html_cache = {}

def _chrome_options():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-plugins")
    opts.add_argument("--disable-background-networking")
    opts.add_argument("--disable-sync")
    opts.add_argument("--no-first-run")
    opts.add_argument("--mute-audio")
    opts.add_argument("--disable-default-apps")
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
prompt_template = ChatPromptTemplate.from_template("""
You are an expert QA engineer.

Convert the following test step into **runnable Python Selenium code** using the provided `driver`.

Rules:
- If this is the first step in the test case, include:
    driver.get("{website}")
  before interacting with the page.

- You are also given the full HTML of the page. You MUST extract and use the **exact attributes** from the HTML (id, name, placeholder, value, visible text, class if unique).
- Do NOT invent or assume element IDs or names. Only use selectors that actually appear in the HTML.
- Priority for locators: ID > Name > Placeholder/Text > CSS Selector > XPath.
- If no clean locator exists, construct an XPath using visible text or hierarchy from the given HTML.
- Only use locators that appear exactly in the provided HTML.
- For Hebrew or non‑Latin text, match the exact visible text from the HTML, preserving spaces, punctuation, and case.
- When navigating to a file:/// URL, assume Chrome is launched with options to allow local file access and disable web security.
- Never call driver.quit(), driver.close(), or end the browser session in any way.
- Output only raw Python code, no markdown, no backticks, no explanations.
- Use `WebDriverWait` for all element interactions.
- Do NOT re-create the driver; use the existing `driver` passed in.
- **Never** call driver.quit(), driver.close(), or end the browser session in any way.
- Do NOT navigate away from the provided Website URL unless explicitly stated in the step.
- Include necessary imports if they are missing.
- The code must be executable and independent for this step.
- Do NOT include markdown or backticks.
- The step should be executable standalone in a Python file.
- Capture a screenshot if the element is not found or an action fails, but do not terminate the driver afterwards.

Website URL: {website}

Full HTML: {html}

Step: {step}
Expected result: {expected}
""")

full_case_prompt = ChatPromptTemplate.from_template("""
You are an expert QA engineer.

Generate a single complete Python Selenium script for ALL steps of this test case using the provided `driver`.

Rules:
- Start with all necessary imports (selenium, WebDriverWait, By, EC, time).
- Call driver.get("{website}") exactly ONCE at the very beginning.
- Execute each step in order as a logical sequence — do NOT repeat driver.get() or imports.
- You are given the full HTML of the page. Use ONLY exact attributes from the HTML (id, name, placeholder, visible text, class if unique).
- Priority for locators: ID > Name > Placeholder/Text > CSS Selector > XPath.
- For Hebrew or non-Latin text, match the exact visible text from the HTML.
- Use WebDriverWait for all element interactions.
- Never call driver.quit() or driver.close().
- Output only raw Python code — no markdown, no backticks, no explanations.
- Capture a screenshot on failure but do not terminate the driver.

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
def extract_full_html(url: str) -> str:
    if url in _html_cache:
        return _html_cache[url]
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=_chrome_options())
    driver.get(url)
    time.sleep(2)
    html = driver.page_source
    driver.quit()
    _html_cache[url] = html
    return html

def generate_selenium_code(step_text, expected_text, website, page_html):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

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
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        import time

        opts = Options()
        for arg in [
            "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-gpu", "--disable-extensions", "--no-first-run",
            "--mute-audio", "--disable-default-apps",
        ]:
            opts.add_argument(arg)

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=opts
        )
        try:
            code = open({repr(file_path)}).read()
            exec(code, {{
                "driver": driver, "By": By,
                "WebDriverWait": WebDriverWait, "EC": EC, "time": time,
            }})
            print("RESULT:Pass")
        except Exception as e:
            print(f"RESULT:Fail:{{e}}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass
    """)

    result = {"id": case_id, "status": "Pass", "error": None, "screenshot": None}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=120
        )
        combined = proc.stdout + "\n" + proc.stderr
        for line in combined.splitlines():
            if line.startswith("RESULT:Pass"):
                result["status"] = "Pass"
                break
            if line.startswith("RESULT:Fail:"):
                result["status"] = "Fail"
                result["error"] = line[len("RESULT:Fail:"):]
                break
        else:
            result["status"] = "Fail"
            result["error"] = (proc.stderr or proc.stdout or "Unknown error").strip()
    except subprocess.TimeoutExpired:
        result["status"] = "Fail"
        result["error"] = "Test timed out after 120 seconds"
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
