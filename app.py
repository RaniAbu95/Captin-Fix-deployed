import os
import subprocess
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from openai import OpenAI

app = Flask(__name__)


def make_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)


def openai_client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo")
def demo():
    return send_file("ActionChainsEx.Html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    driver = None
    try:
        driver = make_driver()
        driver.get(url)
        page_source = driver.page_source
    except Exception as e:
        return jsonify({"error": f"Failed to load page: {e}"}), 500
    finally:
        if driver:
            driver.quit()

    try:
        client = openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an assistant for Selenium automation."},
                {"role": "user", "content": f"""Given this HTML of a webpage:
{page_source[:8000]}

Identify the active elements (buttons, links, forms, inputs) and describe the possible actions that can be performed on them.
Return a numbered list grouped by element type. Write in English."""}
            ],
            max_tokens=800
        )
        return jsonify({"analysis": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 500


@app.route("/plan", methods=["POST"])
def plan():
    data = request.get_json()
    url = data.get("url", "").strip()
    analysis = data.get("analysis", "").strip()
    num_cases = max(1, int(data.get("num_cases", 5)))
    if not url or not analysis:
        return jsonify({"error": "URL and analysis are required"}), 400

    base = num_cases // 3
    remainder = num_cases % 3
    smoke_count = base + (1 if remainder > 0 else 0)
    nav_count = base + (1 if remainder > 1 else 0)
    form_count = base

    try:
        client = openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an experienced QA engineer creating structured test plans. Return only valid JSON, no markdown."},
                {"role": "user", "content": f"""Given these page elements:
{analysis}

Generate a test plan for: {url}

Return a JSON object in exactly this format:
{{
  "suites": {{
    "SMOKE": [
      {{"id": "S1", "description": "...", "steps": ["step1", "step2"], "expected": "..."}}
    ],
    "NAVIGATION": [...],
    "FORM": [...]
  }}
}}

Rules:
- SMOKE: exactly {smoke_count} test case(s) — page load and key element checks
- NAVIGATION: exactly {nav_count} test case(s) — link clicks and page transitions
- FORM: exactly {form_count} test case(s) — input filling and form submission
- Only include elements that actually exist on the page
- Steps must be clear and actionable"""}
            ],
            max_tokens=1500
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))

        import json
        parsed = json.loads(raw)
        return jsonify({
            "plan": parsed,
            "distribution": {"SMOKE": smoke_count, "NAVIGATION": nav_count, "FORM": form_count}
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate plan: {e}"}), 500


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    url = data.get("url", "").strip()
    plan_data = data.get("plan", {})
    if not url or not plan_data:
        return jsonify({"error": "URL and test plan are required"}), 400

    import json
    plan_str = json.dumps(plan_data, indent=2)

    try:
        client = openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an experienced QA engineer writing Selenium Python automation."},
                {"role": "system", "content": "Return ONLY raw Python code. No explanations. No markdown. No code blocks."},
                {"role": "system", "content": "Configure Chrome with: --headless=new, --no-sandbox, --disable-dev-shm-usage, --disable-gpu"},
                {"role": "user", "content": f"""Implement this test plan as a Python Selenium script for URL: {url}

Test plan:
{plan_str}

For each test case:
- Print [SUITE_NAME] TC-<id>: <description> ... PASS or FAIL
- Wrap each in try/except so failures do not stop the rest
- Follow the steps and expected results from the plan exactly"""}
            ],
            max_tokens=2000
        )
        code = response.choices[0].message.content
        if code.startswith("```"):
            code = "\n".join(l for l in code.split("\n") if not l.startswith("```"))
        return jsonify({"code": code.strip()})
    except Exception as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 500


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json()
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "No code to run"}), 400

    # Ensure Chrome always runs headless when executing generated code on the server
    preamble = """import selenium.webdriver as _wd
from selenium.webdriver.chrome.options import Options as _Opts
_orig_init = _wd.Chrome.__init__
def _headless_init(self, *a, **kw):
    opts = kw.pop('options', None) or _Opts()
    for arg in ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
        opts.add_argument(arg)
    kw['options'] = opts
    _orig_init(self, *a, **kw)
_wd.Chrome.__init__ = _headless_init
"""

    full_code = preamble + "\n" + code

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full_code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["python", tmp],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        return jsonify({"output": output.strip() or "Tests completed with no output."})
    except subprocess.TimeoutExpired:
        return jsonify({"output": "Tests timed out after 60 seconds."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
