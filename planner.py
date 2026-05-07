import json
import os
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")



from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from pydantic import BaseModel
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# from langchain.prompts import ChatPromptTemplate
from testPlan import process_target_data
from config import ANTHROPIC_API_KEY



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
# Website Sampler using Selenium with user parameters
# -----------------------------
def sample_links(url: str, num_tests: int, depth: int) -> List[str]:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
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
def extract_full_html(url: str) -> str:
    """Extract the entire HTML of the given page."""
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(2)

    html = driver.page_source
    driver.quit()
    return html


def generate_testplan(url: str, links: List[str], num_tests: int) -> TestPlan:
    # Extract the full HTML from the page
    page_html = extract_full_html(url)

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2
    )

    template = ChatPromptTemplate.from_template("""
        You are an expert QA engineer.
        Here is the FULL HTML of the target website:
        {page_html}

        Generate a structured test plan in JSON with:
        - Suites: Smoke, Navigation, Forms
        - Each test case must include: id, suite, steps, expected, priority.
        - Generate a total of exactly {num_tests} test cases distributed across the suites.
        - Only use elements that are actually present in the HTML.
        - Do NOT invent links, forms, or buttons that are not in the HTML.
        - Make steps clear and actionable (like clicking buttons, filling inputs).

        UNIQUENESS RULES (strictly enforced):
        - Every test case must test a DIFFERENT feature, interaction, or user flow.
        - No two test cases may have the same steps or the same expected result, even if worded differently.
        - Do NOT generate variations of the same action (e.g. two test cases that both search for a term and click Google Search are duplicates — generate only ONE).
        - Before finalising each test case, check it is not already covered by a previous one.
        - If you run out of distinct features to test, reduce the number of test cases rather than creating duplicates.

        Return only valid JSON.
    """)

    prompt = template.format_messages(page_html=page_html, num_tests=num_tests)
    response = llm.invoke(prompt)
    plan_json = response.content.strip()
    print("LLM Output:", plan_json)

    if plan_json.startswith("```json"):
        plan_json = plan_json.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(plan_json)
    except json.JSONDecodeError as e:
        print("❌ Failed JSON parsing. Raw LLM output:", plan_json)
        raise ValueError("LLM did not return valid JSON") from e

    cases = []

    # Handle flat {"cases": [...]} or {"testPlan": {"cases": [...]}}
    raw = parsed.get("testPlan", parsed)
    flat_cases = raw.get("cases") if isinstance(raw, dict) else None
    if flat_cases and isinstance(flat_cases, list):
        for c in flat_cases:
            cases.append(TestCase(**c))
        suites_list = list(dict.fromkeys(c.suite for c in cases))
    else:
        # Support {"suites": {suite_name: [cases]}}
        suites_dict = raw.get("suites", {}) if isinstance(raw, dict) else {}
        if isinstance(suites_dict, dict):
            for suite_name, suite_cases in suites_dict.items():
                for c in suite_cases:
                    cases.append(TestCase(**c))
            suites_list = list(suites_dict.keys())
        else:
            suites_list = suites_dict  # it's already a list of names

    # Fill in any missing cases with a follow-up LLM call
    if len(cases) < num_tests:
        missing = num_tests - len(cases)
        existing_ids = [c.id for c in cases]
        fill_template = ChatPromptTemplate.from_template("""
            You are an expert QA engineer.
            Here is the FULL HTML of the target website:
            {page_html}

            A test plan already has these test case IDs: {existing_ids}
            Generate exactly {missing} MORE test cases (do NOT repeat those IDs).
            Suites to use: Smoke, Navigation, Forms.
            Each test case must include: id, suite, steps, expected, priority.
            Only use elements actually present in the HTML.
            Return only a valid JSON array of test case objects, like:
            [{{"id":"...","suite":"...","steps":[...],"expected":"...","priority":"..."}}]
        """)
        fill_prompt = fill_template.format_messages(
            page_html=page_html,
            existing_ids=existing_ids,
            missing=missing,
        )
        fill_response = llm.invoke(fill_prompt)
        fill_json = fill_response.content.strip()
        if fill_json.startswith("```"):
            fill_json = fill_json.split("```")[1]
            if fill_json.startswith("json"):
                fill_json = fill_json[4:]
            fill_json = fill_json.strip()
        try:
            extra = json.loads(fill_json)
            if isinstance(extra, list):
                for c in extra:
                    cases.append(TestCase(**c))
        except Exception:
            pass  # best-effort; proceed with however many we have

    if not suites_list:
        suites_list = list(dict.fromkeys(c.suite for c in cases))

    return TestPlan(website=url, suites=suites_list, cases=cases)


# -----------------------------
# Save Outputs
# -----------------------------

def save_testplan(plan: TestPlan, base_path: str = "./output"):
    # JSON
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/plan.json", "w", encoding="utf-8") as f:
        json.dump(plan.dict(), f, indent=2, ensure_ascii=False)

    # Excel
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
    df.to_excel(f"{base_path}/Plan.xlsx", index=False)

# -----------------------------
# Runner using existing user parameters
# -----------------------------
# def run_planner(target: str, num_tests: int = 5, depth: int = 1, email: str = "", pm: str = "jira"):
#
#     process_target_data(target)
#     # Validate URL
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
#         }
#         resp = requests.get(target, headers=headers, timeout=10)
#         resp.raise_for_status()
#     except Exception as e:
#         print(f"Site not accessible: {e}")
#         return
#
#     links = sample_links(target, num_tests=num_tests, depth=depth)
#     plan = generate_testplan(target, links)
#     save_testplan(plan)
#
#     print(f"Test Plan generated successfully for {target}!")
#     if email:
#         print(f"Results will be sent to: {email}")
#     print(f"Project Management tool selected: {pm}")


import os
import requests
from urllib.parse import urlparse, unquote

def run_planner(target: str, num_tests: int, depth: int, email: str = "", pm: str = ""):
    parsed = urlparse(target)
    if parsed.scheme == "file":
        local_path = unquote(parsed.path)
        if os.name == "nt" and local_path.startswith("/"):
            local_path = local_path[1:]
        if not os.path.exists(local_path):
            raise ValueError(f"Local file not accessible: {local_path}")
    elif parsed.scheme in ("http", "https"):
        try:
            resp = requests.get(target, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            raise ValueError(f"Site not accessible: {e}") from e
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    plan = generate_testplan(target, [], num_tests)
    save_testplan(plan)
    print(f"Test Plan generated for {target}")



if __name__ == "__main__":
    run_planner('https://www.google.com/')