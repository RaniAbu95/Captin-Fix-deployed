import os
import json
import traceback
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from selenium.webdriver.chrome.options import Options
import time
from jira_utils import create_jira_issue, attach_file_to_issue
from trello_utils import create_trello_card, attach_file_to_card
from dotenv import load_dotenv

load_dotenv(".env")
PM_TOOL = os.getenv("PM_TOOL", "jira")  # "jira" or "trello"
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "CFQA")
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID", "")

PLAN_FILE = "./output/plan.json"
OUTPUT_DIR = "tests"
RESULTS_JSON = "Results.json"
SCREENSHOT_DIR = "./screen/screenshots"

# -----------------------------
# 1. Setup LangChain AI
# -----------------------------
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY

)

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

# prompt_template = ChatPromptTemplate.from_template("""
# You are an expert QA engineer.
#
# Convert the following test step into **runnable Python Selenium code** using the provided `driver`.
#
# Rules:
# - If this is the first step in the test case, include:
#     driver.get("{website}")
#   before interacting with the page.
# - You are also given the full HTML of the page. You MUST extract and use the **exact attributes** from the HTML (id, name, placeholder, value, visible text, class if unique).
# - Do NOT invent or assume element IDs or names. Only use selectors that actually appear in the HTML.
# - Priority for locators: ID > Name > Placeholder/Text > CSS Selector > XPath.
# - For Hebrew or non‑Latin text, match the exact visible text from the HTML.
# - Never call driver.quit(), driver.close(), or end the browser session.
# - Output only raw Python code, no markdown, no backticks, no explanations.
# - Use WebDriverWait for all element interactions.
# - Do NOT use deprecated methods like `find_element_by_id`, `find_element_by_name`, or `find_element_by_xpath`.
# - Always use `driver.find_element(By.ID, ...)`, `driver.find_element(By.NAME, ...)`, `driver.find_element(By.CSS_SELECTOR, ...)`, or `driver.find_element(By.XPATH, ...)`.
# - Ensure all waits use `WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "...")))` etc.
#
# - Do NOT re-create the driver; use the existing `driver`.
# - Only output raw Python code, no markdown/backticks
# - Include a verification step after performing the action:
#     - Compare the actual page state (text, value, or visibility of an element) with the **Expected result**.
#     - If the verification fails, raise an `AssertionError`.
#     - Capture a screenshot if the assertion fails.
# - The step must be **executable standalone**.
# -  Always include a try-except around waits:
#     try:
#         WebDriverWait(driver, 20).until(...)
#     except TimeoutException:
#         driver.save_screenshot("timeout_error.png")
#         raise AssertionError("Element not found or not visible")
# - Do not call driver.get() multiple times; navigate with clicks or form submissions instead.
#
#
# Website URL: {website}
#
# Full HTML: {html}
#
# Step: {step}
# Expected result: {expected}
# """)

prompt_template = ChatPromptTemplate.from_template("""
You are an expert QA engineer.

Convert the following test step into **runnable Python Selenium code** using the provided `driver`.

Rules:
- Follow the steps **in the same order** as listed in the test plan to convert them into runnable Python Selenium code.
- Each step may include actions such as navigation, form filling, button clicking, message verification, etc.
-If the plan provides input data such as username or password, use these exact values in your Selenium code.
- If login credentials (`username`, `password`) are provided in the plan, locate and fill the correct fields on the login page, then click the login/submit button before proceeding.
-ensure that you use the provided credentials 
- If this is the first step in the test case, include:
    driver.get("{website}")
  before interacting with the page.
- You are also given the full HTML of the page. You MUST extract and use the **exact attributes** from the HTML (id, name, placeholder, value, visible text, class if unique). 
- Do NOT invent or assume element IDs or names. Only use selectors that actually appear in the HTML.
- Priority for locators: ID > Name > Placeholder/Text > CSS Selector > XPath.
- For Hebrew or non‑Latin text, match the exact visible text from the HTML.
- Never call driver.quit(), driver.close(), or end the browser session.
- Output only raw Python code, no markdown or backticks.
- Do NOT use deprecated methods like `find_element_by_id`, `find_element_by_name`, or `find_element_by_xpath`.
- Always use `driver.find_element(By.ID, ...)`, `driver.find_element(By.NAME, ...)`, `driver.find_element(By.CSS_SELECTOR, ...)`, or `driver.find_element(By.XPATH, ...)`.
- Ensure all waits use `WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "...")))` or equivalent.
- Do NOT re-create the driver; use the existing `driver`.
- Only output raw Python code, no markdown/backticks.
- Wrap all element interactions (click, send_keys, get_text) in try-except using WebDriverWait:

- Include a verification step after performing the action.
- Compare the actual page state (text, value, or visibility of an element) with the **Expected result**.
- If the verification fails, raise an `AssertionError`.

- Handle any alert or popup gracefully:

- Include necessary imports at the top if missing:
- If an action (like clicking a button or submitting a form) may trigger a JavaScript alert:
    - Use WebDriverWait(driver, 20).until(EC.alert_is_present()) to wait for the alert.
    - Capture the alert text and compare it to the expected result.
    - Accept the alert after verification.
    - If the alert does not appear or the text is incorrect,  raise AssertionError.
    -Compare the text to the **expected result using assert statements**.

-Detect all errors on the page** after every action:
    - JavaScript alerts
    - Modals/popups
    - Inline form errors
    - Toast or banner notifications



-Always start your generated code by importing all required Selenium modules:
- The step must be **executable standalone**.
- Do not call driver.get() multiple times; navigate with clicks or form submissions instead.
-also generate log file for each selenuim code and store in folder ./logs and specify the log name with date
-at the begging write the test name to the log ant at the end add seperator
Website URL: {website}

Full HTML: {html}

Step: {step}
Expected result: {expected}
""")


def extract_full_html(url: str) -> str:
    """Extract the entire HTML of the given page."""
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    time.sleep(2)

    html = driver.page_source
    driver.quit()
    return html


def generate_selenium_code(step_text, expected_text, website, html_cache, file_path):
    page_html = html_cache
    messages = prompt_template.format_messages(
        step=step_text,
        expected=expected_text,
        website=website,
        html=page_html
    )
    response = llm.invoke(messages)
    # print("response from generate func:", response.content.strip())

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.content.strip())

    return response.content.strip()


def generate_test_files(plan):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # إنشاء اسم المجلد مع التاريخ
    folder_name = f"{OUTPUT_DIR}/{now}"

    # إنشاء المجلد (إن لم يكن موجودًا مسبقًا)
    os.makedirs(folder_name, exist_ok=True)
    test_files = []
    website = plan.get("website", "")
    html_cache = extract_full_html(website)
    PLAN_FILE = "./output/plan.json"

    folder = "tests"
    for case in plan["cases"]:
        case_id = case["id"]
        steps = case.get("steps", [])
        # print("steps: __________",steps)
        expected = case.get("expected", "")
        file_path = os.path.join(folder_name, f"{case_id}.py")
        # --- Generate code for all steps in parallel ---
        with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers
            futures = [executor.submit(generate_selenium_code, steps, expected, website, html_cache, file_path) for step
                       in steps]
            all_code = [f.result() for f in futures]
            # print("all code 2: ", all_code)
        # file_path = os.path.join(folder_name, f"{case_id}.py")
        # with open(file_path, "w", encoding="utf-8") as f:
        #     f.write("\n\n".join(all_code))
        #
        test_files.append((case_id, file_path))
        print(f"✅ Generated test file: {file_path}")

    return test_files


# -----------------------------
# 3. Run generated Selenium tests
# -----------------------------

def report_failure(case_id, screenshot_path, error_msg):
    """Report failed test case to Jira or Trello"""
    summary = f"Test Case Failed: {case_id}"
    #PM_TOOL.lower() = "trello"
    description = f"Error: {error_msg}\nScreenshot attached."
    if PM_TOOL.lower() == "jira":
        issue = create_jira_issue(summary, description, project_key=JIRA_PROJECT_KEY)
        if issue:
            attach_file_to_issue(issue["key"], screenshot_path)
    elif PM_TOOL.lower() == "trello":
        card = create_trello_card(summary, description, list_id=TRELLO_LIST_ID)
        if card:
            attach_file_to_card(card["id"], screenshot_path)


from concurrent.futures import ThreadPoolExecutor


def run_test_file(case_id, file_path):
    """
    Run a generated test file, capture screenshot/console/network logs and duration.
    Places evidence in ./evidence/<case_id>/
    Returns a result dict with id, status, error, screenshot, console_log, network_log, duration.
    """
    case_dir = os.path.join("evidence", case_id)
    os.makedirs(case_dir, exist_ok=True)

    result = {
        "id": case_id,
        "status": "Pass",
        "error": None,
        "screenshot": None,
        "console_log": None,
        "network_log": None,
        "duration": None
    }

    start_time = time.time()
    driver = None

    try:
        # ---------- Setup Chrome options + logging capabilities ----------
        options = Options()
        # Uncomment if you want headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        # Enable browser console and performance (network) logs via capabilities on options
        options.set_capability("goog:loggingPrefs", {"browser": "ALL", "performance": "ALL"})

        # ---------- Create driver ----------
        try:
            # try webdriver_manager if installed (convenient)
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        except Exception:
            # fallback to local chromedriver in PATH
            driver = webdriver.Chrome(options=options)

        # enable Network via CDP to be able to capture network details
        try:
            driver.execute_cdp_cmd("Network.enable", {})
        except Exception:
            # if this fails, we'll still try to read performance logs later
            pass

        # ---------- Execute the generated test file ----------
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # # Execute inside a controlled namespace — give it access to driver and By
        # exec_globals = {"driver": driver, "By": By}
        # exec(code, exec_globals)
        # from selenium.webdriver.support.ui import WebDriverWait
        # from selenium.webdriver.support import expected_conditions as EC
        # from selenium.common.exceptions import TimeoutException
        #
        # exec_globals = {
        #     "driver": driver,
        #     "By": By,
        #     "WebDriverWait": WebDriverWait,
        #     "EC": EC,
        #     "TimeoutException": TimeoutException,
        # }
        #
        # exec(code, exec_globals)
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException

        exec_globals = {
            "driver": driver,
            "By": By,
            "Keys": Keys,
            "WebDriverWait": WebDriverWait,
            "EC": EC,
            "TimeoutException": TimeoutException,
        }
        exec(code, exec_globals)



    except Exception as exc:
        result["status"] = "Fail"
        # store full traceback to help debugging
        result["error"] = traceback.format_exc()

        # Screenshot on failure (if driver exists)
        try:
            if driver:
                screenshot_path = os.path.join(case_dir, f"{case_id}.png")
                driver.save_screenshot(screenshot_path)
                result["screenshot"] = screenshot_path
        except Exception:
            # ignore screenshot failures
            pass

            # ✅ Call report_failure() here
        try:
            report_failure(case_id, screenshot_path, result["error"])
        except Exception as report_err:
            print(f"⚠️ Failed to report issue for {case_id}: {report_err}")

    finally:
        # ---------- Collect logs if driver exists ----------
        if driver:
            # Console logs (browser)
            try:
                console_logs = driver.get_log("browser")
                console_log_path = os.path.join(case_dir, "console.log")
                with open(console_log_path, "w", encoding="utf-8") as f:
                    for entry in console_logs:
                        # each entry is a dict: {'level':..., 'message':..., 'timestamp':...}
                        f.write(f"{entry.get('level')} - {entry.get('message')}\n")
                result["console_log"] = console_log_path
            except Exception:
                # not fatal if logs can't be retrieved
                pass

            # Performance logs (network)
            try:
                perf_logs = driver.get_log("performance")
                network_log_path = os.path.join(case_dir, "network.json")
                with open(network_log_path, "w", encoding="utf-8") as f:
                    json.dump(perf_logs, f, indent=2, ensure_ascii=False)
                result["network_log"] = network_log_path
            except Exception:
                pass

            # Quit the driver
            try:
                driver.quit()
            except Exception:
                pass

        # ---------- Duration ----------
        end_time = time.time()
        result["duration"] = round(end_time - start_time, 3)

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