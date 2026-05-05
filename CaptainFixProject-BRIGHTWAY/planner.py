import json
import os
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from pydantic import BaseModel
from typing import List
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from testPlan import process_target_data
from dotenv import load_dotenv
from email_utils import send_results_email
from jira_utils import create_jira_issue, attach_file_to_issue
from trello_utils import create_trello_card, attach_file_to_card

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# -----------------------------
# Data Schemas
# -----------------------------
class TestCase(BaseModel):
    id: str
    suite: str
    steps: List[str]
    expected: str
    priority: str


class TestPlan(BaseModel):
    website: str
    suites: List[str]
    cases: List[TestCase]


# -----------------------------
# Website Sampler
# -----------------------------
def sample_links(url: str, num_tests: int = 5, depth: int = 1) -> List[str]:
    print("numtest: ", num_tests, " depth: ", depth)
    options = Options()
    options.headless = True
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    visited = set()
    to_visit = [(url, 0)]
    links = []

    while to_visit and len(links) < num_tests:
        current_url, current_depth = to_visit.pop(0)
        if current_url in visited or current_depth > depth:
            continue
        try:
            driver.get(current_url)
            time.sleep(2)
        except Exception:
            continue

        visited.add(current_url)
        elements = driver.find_elements(By.TAG_NAME, 'a')
        for elem in elements:
            link = elem.get_attribute('href')
            if link and link.startswith('http') and link not in links:
                links.append(link)
                if current_depth + 1 <= depth:
                    to_visit.append((link, current_depth + 1))
            if len(links) >= num_tests:
                break

    driver.quit()
    return links


# -----------------------------
# LLM Planner
# -----------------------------
# def extract_full_html(url: str) -> str:
#     options = Options()
#     options.headless = True
#     driver = webdriver.Chrome(options=options)
#     driver.get(url)
#     time.sleep(2)
#     html = driver.page_source
#     driver.quit()
#     return html

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    StaleElementReferenceException
)
import time

# def extract_full_html(url: str) -> str:
#     options = Options()
#     options.add_argument("--headless=new")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-gpu")
#     driver = webdriver.Chrome(options=options)
#     driver.set_window_size(1920, 1080)
#     driver.get(url)
#
#     # ✅ Wait for JS to load fully
#     WebDriverWait(driver, 15).until(
#         lambda d: d.execute_script("return document.readyState") == "complete"
#     )
#
#     # ✅ Scroll to the bottom to trigger lazy-loaded items
#     scroll_pause = 1.5
#     last_height = driver.execute_script("return document.body.scrollHeight")
#     while True:
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(scroll_pause)
#         new_height = driver.execute_script("return document.body.scrollHeight")
#         if new_height == last_height:
#             break
#         last_height = new_height
#
#     # ✅ Try clicking visible interactive elements
#     interactive_selectors = "//button | //a | //input[@type='button'] | //input[@type='submit']"
#     elements = driver.find_elements(By.XPATH, interactive_selectors)
#
#     for el in elements:
#         try:
#             driver.execute_script("arguments[0].scrollIntoView(true);", el)
#             ActionChains(driver).move_to_element(el).perform()
#             if el.is_displayed() and el.is_enabled():
#                 el.click()
#                 time.sleep(1.5)
#         except (ElementClickInterceptedException, ElementNotInteractableException, StaleElementReferenceException):
#             continue
#         except Exception:
#             continue
#
#     # ✅ Wait a bit more for dynamic changes
#     time.sleep(2)
#
#     # ✅ Capture final rendered HTML
#     html = driver.page_source
#     driver.quit()
#     return html

from playwright.sync_api import sync_playwright
import time


def extract_full_html(url: str) -> str:
    """
    Extracts the full HTML of a website, including dynamically loaded content,
    modals, buttons, and iframes using Playwright.
    """

    html_snapshots = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')  # Wait until network is idle
        page.set_viewport_size({"width": 1920, "height": 1080})

        # ✅ Scroll to bottom to trigger lazy load
        scroll_pause = 1.2
        previous_height = 0
        while True:
            page.evaluate("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height

        # ✅ Click all visible buttons/links to reveal dynamic content
        selectors = "button, a, input[type='button'], input[type='submit'], div[role='button']"
        elements = page.query_selector_all(selectors)
        for el in elements:
            try:
                if el.is_visible():
                    el.click(timeout=1000)
                    time.sleep(0.5)
                    html_snapshots.append(page.content())
            except:
                continue

        # ✅ Extract iframe content
        for frame in page.frames:
            try:
                html_snapshots.append(frame.content())
            except:
                continue

        # ✅ Final snapshot
        html_snapshots.append(page.content())
        browser.close()

    # Merge all snapshots
    merged_html = "\n<!-- snapshot separator -->\n".join(list(set(html_snapshots)))
    return merged_html


def generate_testplan(url: str, links: List[str], instructions="") -> TestPlan:
    page_html = extract_full_html(url)

    extra_context = instructions
    print(extra_context)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.2
    )

    # template = ChatPromptTemplate.from_template("""
    # You are an expert QA engineer.
    #
    #
    # Your task: Generate a **comprehensive structured test plan in JSON** that:
    #
    # - Covers **all success scenarios** and **all edge/failure cases**.
    # - Includes the following test suites: Smoke, Navigation, Forms.
    # - For each test case, include:
    #     - `id` (unique identifier)
    #     - `suite` (Smoke, Navigation, Forms)
    #     - `steps` (clear, actionable instructions using only elements present in the HTML)
    #     - `expected` (expected result for each step)
    #     - `priority` (High, Medium, Low)
    # - Only use actual elements in the HTML. **Do not invent any links, forms, buttons, or fields.**
    # - Include test credentials:
    #     - **Username:** emilys
    #     - **Password:** emilyspass
    # -explain why this test plan was chosen
    #
    # **Important:** Focus on generating all possible success paths and edge/failure cases. Steps should be actionable and precise (e.g., click, fill, submit).
    # Return **only JSON**.
    #
    # Here is the FULL HTML of the target website:
    # --start of html--
    # {page_html}
    # --end of html--
    # """)
    #
    # template = ChatPromptTemplate.from_template("""
    # You are an expert QA engineer.
    # Here is the FULL HTML of the target website:
    # {page_html}
    #
    # Your task: Generate a **comprehensive and exhaustive test plan in JSON** that:
    #
    # - Covers **all success scenarios** and **all edge/failure cases** for every interactive element in the HTML.
    # - Include these suites: Smoke, Navigation, Forms.
    # - For each test case, include: id, suite, steps, expected, priority.
    # - Only use actual elements in the HTML. Do NOT invent any links, forms, or buttons.
    # - Make steps **clear, actionable, and granular** (clicks, input values, form submissions, navigation steps).
    # - Consider edge cases like:
    #     - Empty or missing inputs
    #     - Invalid inputs
    #     - Max-length inputs
    #     - Multiple or repeated clicks
    #     - Unexpected navigation sequences
    # - Include test credentials:
    #     - Username: emilys
    #     - Password: emilyspass
    #
    # Return **only JSON**.
    # """)

    # template = ChatPromptTemplate.from_template(f"""
    # You are an expert QA engineer with extensive experience in automated test generation.
    # Here is the FULL HTML of the target website:
    # {page_html}
    #
    # {extra_context}
    #
    # ### TASK
    # Generate a **comprehensive and exhaustive test plan in JSON** that includes:
    #
    # 1. **Functional Test Cases** – verifying that each feature works as intended.
    # 2. **Logic Test Cases** – verifying business rules, calculations, validations, and dependencies between fields/actions.
    # 3. **Verification Test Cases** – explicitly check outputs, UI states, form validations, database updates, or expected results after each action.
    # 4. **Edge and Failure Cases** – testing invalid, empty, max-length, or unexpected inputs, repeated clicks, and unusual navigation sequences.
    # 5. **Success Scenarios** – all normal flows for every interactive element.
    # 6. **Sequences and Multi-step Workflows** – interactions across forms, pages, and navigation flows.
    #
    # ### GUIDELINES
    # - Include these suites: Smoke, Navigation, Forms, Functional, Logic, Verification.
    # - For each test case, include: id, suite, steps, expected, priority.
    # - Only use elements present in the HTML. Do NOT invent links, buttons, forms, or fields.
    # - Steps must be **clear, actionable, and granular**, including:
    #     - Clicking buttons or links
    #     -check all the buttons and links are clicking in all the pages
    #     - Entering input values (valid, invalid, empty, special characters, max-length)
    #     - Selecting dropdown options
    #     - Checking/unchecking checkboxes
    #     - Submitting forms
    #     - Navigating across multiple pages or forms
    #     - Verifying UI changes, messages, validations, calculations, or data updates
    #
    # - Include test credentials where applicable:
    #     - Username: emilys
    #     - Password: emilyspass
    # - Assign priority: High (critical), Medium (important), Low (optional).
    # - Ensure **all possible test cases** are generated covering functional, logic, verification, success, edge, and failure scenarios.
    #
    # ### OUTPUT
    # Return **only JSON**.
    # """)
    template = ChatPromptTemplate.from_template("""
    You are an expert QA engineer with extensive experience in automated test generation.  
    Here is the FULL HTML of the target website:  
    {page_html}
    ### Whenever a test case involves **any form of login, authentication, or user identity verification**:
    - You **must explicitly include** these exact data:
    {extra_context}
    ### TASK
    Generate a **comprehensive and exhaustive test plan in JSON** that includes:

    1. **Functional Test Cases** – verifying that each feature works as intended.
    2. **Logic Test Cases** – verifying business rules, calculations, validations, and dependencies between fields/actions.
    3. **Verification Test Cases** – explicitly check outputs, UI states, form validations, database updates, or expected results after each action.
    4. **Edge and Failure Cases** – testing invalid, empty, max-length, or unexpected inputs, repeated clicks, and unusual navigation sequences.
    5. **Success Scenarios** – all normal flows for every interactive element.
    6. **Sequences and Multi-step Workflows** – interactions across forms, pages, and navigation flows.

    ### GUIDELINES
    - Include these suites: Smoke, Navigation, Forms, Functional, Logic, Verification.
    - For each test case, include: **id**, **suite**, **steps**, **expected**, **priority**.
    - Only use elements present in the HTML. Do NOT invent links, buttons, forms, or fields.
    - Steps must be **clear, actionable, and granular**, including:
        - Clicking buttons or links  
        - Entering input values (valid, invalid, empty, special characters, max-length)  
        - Selecting dropdown options  
        - Checking/unchecking checkboxes  
        - Submitting forms  
        - Navigating across multiple pages or forms  
        - Verifying UI changes, messages, validations, calculations, or data updates  


    - Replace every such phrase with the explicit credential entry steps above.
    - Apply this rule in:
      - Forms  
      - Functional tests  
      - Verification tests  
      - Multi-step workflows  
      - Navigation sequences  
    - Even in **edge, failure, or success scenarios**, include explicit credentials if login is required.

    ### ADDITIONAL ENFORCEMENT
    - For any test involving **multi-page flows or sequences**, repeat the credential steps at the start if authentication is required to access the flow.  
    - Do **not** skip credentials even if the LLM thinks it’s implied.  
    - Include **these exact username/password values** in the steps — never omit or anonymize them.  
    - Make the steps **fully copy-paste ready for automation scripts**.

    ### PRIORITY RULES
    - High: Critical login, checkout, or payment flows.
    - Medium: Core functional features.
    - Low: Optional or cosmetic flows.

    ### OUTPUT
    Return **only valid JSON** with this structure:

    - Ensure **every test case involving login explicitly shows username and password values** in its steps.
    """)

    # prompt = template.format_messages(page_html=page_html,test_case_count=test_case_count, depth=depth)
    prompt = template.format(page_html=page_html, extra_context=extra_context)

    print(prompt)
    response = llm.invoke([prompt])
    plan_json = response.content.strip()
    if plan_json.startswith("```json"):
        plan_json = plan_json.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(plan_json)
    except json.JSONDecodeError as e:
        print("❌ Failed JSON parsing. Raw LLM output:", plan_json)
        raise ValueError("LLM did not return valid JSON") from e

    cases = []
    if "testPlan" in parsed:
        suites_dict = parsed["testPlan"].get("suites", {})
    else:
        suites_dict = parsed.get("suites", {})

    for suite_name, suite_cases in suites_dict.items():
        for c in suite_cases:
            cases.append(TestCase(**c))

    return TestPlan(website=url, suites=list(suites_dict.keys()), cases=cases)


# -----------------------------
# Save Outputs
# -----------------------------
def save_testplan(plan: TestPlan, base_path: str = "./output"):
    os.makedirs(base_path, exist_ok=True)
    json_path = f"{base_path}/plan.json"
    excel_path = f"{base_path}/Plan.xlsx"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan.dict(), f, indent=2, ensure_ascii=False)

    data = [
        {
            "ID": c.id,
            "Suite": c.suite,
            "Steps": " | ".join(c.steps),
            "Expected": c.expected,
            "Priority": c.priority
        } for c in plan.cases
    ]
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

    return json_path, excel_path


# -----------------------------
# Planner Runner
# -----------------------------
def run_planner(target: str, num_tests: int = 5, depth: int = 1, email: str = "", pm: str = "jira",
                instructions: str = "", project_key: str = None):
    process_target_data(target)

    links = sample_links(target, num_tests=num_tests, depth=depth)

    plan = generate_testplan(target, links, instructions)

    json_path, excel_path = save_testplan(plan)

    print(f"✅ Test Plan generated successfully for {target}!")

    if email:
        send_results_email(email, attachments=[json_path, excel_path])
        print(f"📧 Results sent to: {email}")

    print(f"📋 Project Management tool selected: {pm}")

    jira_issue_key = None
    trello_card_id = None

    # -----------------------------
    # Jira Integration
    # -----------------------------
    if pm.lower() == "jira" and project_key:
        summary = f"Test Plan Generated for {target}"
        description = f"Files generated:\n- {json_path}\n- {excel_path}"
        issue = create_jira_issue(summary, description, project_key=project_key)
        if issue:
            jira_issue_key = issue.get("key")
            attach_file_to_issue(jira_issue_key, json_path)
            attach_file_to_issue(jira_issue_key, excel_path)

    # -----------------------------
    # Trello Integration
    # -----------------------------
    elif pm.lower() == "trello":
        card_name = f"Test Plan for {target}"
        card_desc = f"Files generated:\n- {json_path}\n- {excel_path}"
        card = create_trello_card(card_name, card_desc)
        if card:
            trello_card_id = card["id"]
            attach_file_to_card(trello_card_id, json_path)
            attach_file_to_card(trello_card_id, excel_path)

    return {
        "json": json_path,
        "excel": excel_path,
        "jira_issue": jira_issue_key,
        "trello_card": trello_card_id
    }
