# test_plan_runner_pm.py

import os
import json
from dotenv import load_dotenv
from sqlalchemy import false

load_dotenv(".env")
import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate



# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #
PLAN_FILE = os.path.join(BASE_DIR, "output", "plan.json")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screen", "screenshots")
RESULTS_JSON = os.path.join(BASE_DIR, "Results.json")
SCREENSHOT_DIR = "./screen/screenshots"
RESULTS_JSON = "Results.json"
load_dotenv(".env")
PM_TOOL = os.getenv("PM_TOOL", "jira")  # "jira" or "trello"
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "CFQA")
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID", "")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
load_dotenv()

from jira_utils import create_jira_issue, attach_file_to_issue
from trello_utils import create_trello_card, attach_file_to_card



# -----------------------------
# Initialize LangChain AI
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

prompt_template = ChatPromptTemplate.from_template("""
You are an expert QA engineer.

Convert the following test step into **runnable Python Selenium code** using the provided `driver`.

Rules:
- Use the exact attributes from the HTML (id, name, placeholder, visible text, class if unique)
- Priority for selectors: ID > Name > Placeholder/Text > CSS Selector > XPath
- Do not invent element IDs/names
- Do not call driver.quit() or driver.close()
- Capture screenshot if action fails
- Only output raw Python code, no markdown/backticks
- Use WebDriverWait for all element interactions

Website URL: {website}
Full HTML: {html}

Step: {step}
Expected result: {expected}
""")

# -----------------------------
# Helper Functions
# -----------------------------
def load_test_plan():
    """Load test plan from JSON file"""
    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return plan.get("cases", [])

def generate_driver(headless=false):
    """Initialize Chrome WebDriver"""
    options = Options()
    if headless:
        options.headless = True
    driver = webdriver.Chrome(options=options)
    return driver

def extract_full_html(url):
    """Extract full HTML of the target page"""
    driver = generate_driver(headless=True)
    driver.get(url)
    time.sleep(2)
    html = driver.page_source
    driver.quit()
    return html

def generate_selenium_code(step_text, expected_text, website):
    """Generate Selenium Python code for a single test step using AI"""
    page_html = extract_full_html(website)
    messages = prompt_template.format_messages(
        step=step_text,
        expected=expected_text,
        website=website,
        html=page_html
    )
    response = llm.invoke(messages)
    return response.content.strip()

def report_failure(case_id, screenshot_path, error_msg):
    """Report failed test case to Jira or Trello"""
    summary = f"Test Case Failed: {case_id}"
    description = f"Error: {error_msg}\nScreenshot attached."

    if PM_TOOL.lower() == "jira":
        issue = create_jira_issue(summary, description, project_key=JIRA_PROJECT_KEY)
        if issue:
            attach_file_to_issue(issue["key"], screenshot_path)
    elif PM_TOOL.lower() == "trello":
        card = create_trello_card(summary, description, list_id=TRELLO_LIST_ID)
        if card:
            attach_file_to_card(card["id"], screenshot_path)

# -----------------------------
# Dynamically Generate Tests
# -----------------------------
test_plan = load_test_plan()
global_website = None
with open(PLAN_FILE, "r", encoding="utf-8") as f:
    plan_json = json.load(f)
    global_website = plan_json.get("website", "")

@pytest.mark.parametrize("case", test_plan)
def test_case_runner(case):
    """Execute each test case from the test plan"""
    case_id = case["id"]
    steps = case.get("steps", [])
    expected = case.get("expected", "")
    website = case.get("website", global_website)

    results = {"id": case_id, "status": "Pass", "error": None, "screenshot": None}

    if not website:
        results["status"] = "Fail"
        results["error"] = "Website URL is empty"
        report_failure(case_id, None, results["error"])
        print(f"[ERROR] Case {case_id} has empty website URL. Skipping...")
        # Save results and skip this case
        if os.path.exists(RESULTS_JSON):
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        else:
            all_results = []
        all_results.append(results)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        return

    driver = generate_driver(headless=True)

    try:
        # Navigate to the website
        print(f"[DEBUG] Case {case_id} opening website: '{website}'")
        driver.get(website)
        time.sleep(2)

        # Execute each step using generated Selenium code
        for step in steps:
            selenium_code = generate_selenium_code(step, expected, website)
            exec(selenium_code, {"driver": driver, "By": By})

    except Exception as e:
        results["status"] = "Fail"
        results["error"] = str(e)
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"{case_id}.png")
        try:
            driver.save_screenshot(screenshot_path)
            results["screenshot"] = screenshot_path
        except:
            pass
        report_failure(case_id, results["screenshot"], results["error"])
        print(f"[ERROR] Exception occurred: {str(e)}")

    finally:
        driver.quit()

    # Save test results
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
