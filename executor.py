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



# def generate_selenium_code(step_text, expected_text):
#     """Generate Python Selenium code for a single step using AI."""
#     messages = prompt_template.format_messages(step=step_text, expected=expected_text)
#     response = llm.invoke(messages)
#     return response.content.strip()
def extract_full_html(url: str) -> str:
    """Extract the entire HTML of the given page."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(url)
    time.sleep(2)

    html = driver.page_source
    driver.quit()
    return html

def generate_selenium_code(step_text, expected_text, website):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    page_html = extract_full_html(website)
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

    for case in plan["cases"]:
        case_id = case["id"]
        steps = case.get("steps", [])
        expected = case.get("expected", "")
        all_code = []

        for step in steps:
            code = generate_selenium_code(step, expected, website)
            all_code.append(code)

        file_path = os.path.join(OUTPUT_DIR, f"{case_id}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_code))

        test_files.append((case_id, file_path))
        print(f"✅ Generated test file: {file_path}")

    return test_files

# -----------------------------
# 3. Run generated Selenium tests
# -----------------------------
def run_test_file(case_id, file_path):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    result = {"id": case_id, "status": "Pass", "error": None, "screenshot": None}

    try:
        _opts = Options()
        _opts.add_argument("--headless=new")
        _opts.add_argument("--no-sandbox")
        _opts.add_argument("--disable-dev-shm-usage")
        _opts.add_argument("--disable-gpu")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=_opts)
        # Run the test file dynamically
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, {"driver": driver, "By": By})

    except Exception as e:
        result["status"] = "Fail"
        result["error"] = str(e)

        # Screenshot on failure
        try:
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"{case_id}.png")
            driver.save_screenshot(screenshot_path)
            result["screenshot"] = screenshot_path
        except:
            pass

    finally:
        try:
            driver.quit()
        except:
            pass

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
