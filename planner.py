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
    negative: bool = False

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
from executor import extract_full_html


def _parse_cases_from_json(parsed) -> tuple:
    """Return (cases, suites_list) from any Claude JSON format."""
    cases = []
    raw = parsed.get("testPlan", parsed)

    # Format 1: flat {"cases": [...]}
    flat_cases = raw.get("cases") if isinstance(raw, dict) else None
    if flat_cases and isinstance(flat_cases, list) and all(isinstance(c, dict) and "id" in c for c in flat_cases):
        for c in flat_cases:
            cases.append(TestCase(**c))
        return cases, list(dict.fromkeys(c.suite for c in cases))

    suites_raw = raw.get("suites", {}) if isinstance(raw, dict) else {}

    # Format 2: {"suites": [{"name": "Smoke", "testCases": [...]}]}
    if isinstance(suites_raw, list):
        suites_list = []
        for suite_obj in suites_raw:
            if isinstance(suite_obj, dict):
                suite_name = suite_obj.get("name", "")
                suites_list.append(suite_name)
                for c in suite_obj.get("testCases", suite_obj.get("cases", [])):
                    if "suite" not in c:
                        c["suite"] = suite_name
                    cases.append(TestCase(**c))
            elif isinstance(suite_obj, str):
                suites_list.append(suite_obj)
        return cases, suites_list

    # Format 3: {"suites": {"Smoke": [...cases], ...}}
    if isinstance(suites_raw, dict):
        suites_list = []
        for suite_name, suite_cases in suites_raw.items():
            suites_list.append(suite_name)
            for c in suite_cases:
                cases.append(TestCase(**c))
        return cases, suites_list

    return cases, list(dict.fromkeys(c.suite for c in cases))


def _strip_json(text: str) -> str:
    import re as _re
    text = _re.sub(r"^```[a-z]*\s*", "", text, flags=_re.IGNORECASE).strip()
    text = _re.sub(r"```$", "", text).strip()
    m = _re.search(r"(\{.*\}|\[.*\])", text, _re.DOTALL)
    return m.group(0) if m else text


def generate_testplan(url: str, links: List[str], num_tests: int) -> TestPlan:
    page_html = extract_full_html(url)

    num_negative = max(1, round(num_tests / 3))
    num_positive = num_tests - num_negative

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=8192,
    )

    template = ChatPromptTemplate.from_template("""
        You are an expert QA engineer.
        Here is the FULL HTML of the target website:
        {page_html}

        Generate a structured test plan in JSON with:
        - Suites: Smoke, Navigation, Forms
        - Each test case must include: id, suite, steps, expected, priority, negative (boolean).
        - Generate a total of EXACTLY {num_tests} test cases — no more, no fewer.
        - Keep steps concise — maximum 4 steps per test case.
        - Make steps clear and actionable (like clicking buttons, filling inputs).

        HTML RULES (strictly enforced):
        - Base EVERY test case ONLY on elements visible in the provided HTML above.
        - Before writing a step, verify the element (button, link, input, image) exists in the HTML.
        - Do NOT use prior knowledge about the website — ignore anything you know about it from training.
        - Do NOT reference id="hplogo" or any element ID/class/text that does not appear in the HTML.
        - If an element is not in the HTML, do not write a test step about it.
        - NEVER assume a link opens in a new tab unless the HTML explicitly shows target="_blank".
          If the HTML shows target="_top" or no target attribute, the link opens in the SAME tab — say so in the expected result.

        UNIQUENESS RULES (strictly enforced):
        - Every test case must test a DIFFERENT feature, interaction, or user flow.
        - No two test cases may have the same steps or the same expected result, even if worded differently.
        - Do NOT generate variations of the same action (e.g. two test cases that both search for a term and click Google Search are duplicates — generate only ONE).
        - Before finalising each test case, check it is not already covered by a previous one.
        - If you run out of distinct features to test, reduce the number of test cases rather than creating duplicates.

        NEGATIVE TEST RULES — COUNT IS MANDATORY:
        - You MUST include EXACTLY {num_negative} negative test cases (negative: true).
        - You MUST include EXACTLY {num_positive} positive test cases (negative: false).
        - A negative test INTENTIONALLY tests invalid behaviour: submitting an empty form,
          entering wrong/invalid input, searching for something that returns no results.
        - ALL other test cases must have "negative": false.
        - "negative" refers to the TEST INTENT, not the runtime outcome.

        BEFORE RETURNING — verify these three counts:
        1. Total test cases = {num_tests}
        2. Cases with "negative": true = {num_negative}
        3. Cases with "negative": false = {num_positive}
        If any count is wrong, fix the JSON before returning.

        Return only valid JSON.
    """)

    prompt = template.format_messages(page_html=page_html, num_tests=num_tests,
                                      num_negative=num_negative, num_positive=num_positive)
    response = llm.invoke(prompt)
    plan_json = response.content.strip()
    stop_reason = (response.response_metadata or {}).get("stop_reason", "unknown")
    print(f"LLM stop_reason: {stop_reason}")
    print(f"LLM Output ({len(plan_json)} chars):", plan_json[:2000])

    plan_json = _strip_json(plan_json)

    try:
        parsed = json.loads(plan_json)
    except json.JSONDecodeError as e:
        preview = plan_json[:2000] if plan_json else "(empty)"
        print("❌ Failed JSON parsing. Full LLM output:", plan_json)
        raise ValueError(f"LLM did not return valid JSON (stop_reason={stop_reason}). Raw output preview: {preview}") from e

    cases, suites_list = _parse_cases_from_json(parsed)

    # Fill missing cases (total count short)
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
            Each test case must include: id, suite, steps, expected, priority, negative (boolean).
            Only use elements actually present in the HTML.
            Return only a valid JSON array of test case objects.
        """)
        fill_resp = llm.invoke(fill_template.format_messages(
            page_html=page_html, existing_ids=existing_ids, missing=missing))
        try:
            extra = json.loads(_strip_json(fill_resp.content.strip()))
            if isinstance(extra, list):
                for c in extra:
                    c.setdefault("negative", False)
                    cases.append(TestCase(**c))
        except Exception:
            pass

    # Enforce exact total count
    cases = cases[:num_tests]

    # Enforce negative ratio via follow-up call if the LLM under-generated negatives
    neg_count = sum(1 for c in cases if c.negative)
    if neg_count < num_negative:
        shortage = num_negative - neg_count
        positives = [c for c in cases if not c.negative]
        negatives = [c for c in cases if c.negative]
        # Drop the last `shortage` positive cases to make room
        positives = positives[:len(positives) - shortage]
        existing_ids = [c.id for c in cases]

        neg_fill_template = ChatPromptTemplate.from_template("""
            You are an expert QA engineer.
            Here is the FULL HTML of the target website:
            {page_html}

            Generate exactly {shortage} NEGATIVE test cases for this website.
            Do NOT repeat these existing IDs: {existing_ids}
            A negative test intentionally tests invalid behaviour:
            submitting an empty form, entering wrong/invalid input,
            searching for something that returns no results, etc.
            Each test case must include: id, suite, steps, expected, priority.
            Set "negative": true on every case you return.
            Only use elements actually present in the HTML.
            Return only a valid JSON array.
        """)
        neg_resp = llm.invoke(neg_fill_template.format_messages(
            page_html=page_html, shortage=shortage, existing_ids=existing_ids))
        try:
            extra_neg = json.loads(_strip_json(neg_resp.content.strip()))
            if isinstance(extra_neg, list):
                for c in extra_neg[:shortage]:
                    c["negative"] = True
                    c.setdefault("suite", "Forms")
                    negatives.append(TestCase(**c))
        except Exception:
            pass

        cases = positives + negatives

    if not suites_list:
        suites_list = list(dict.fromkeys(c.suite for c in cases))
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