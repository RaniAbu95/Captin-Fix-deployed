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


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    url = data.get("url", "").strip()
    analysis = data.get("analysis", "").strip()
    if not url or not analysis:
        return jsonify({"error": "URL and analysis are required"}), 400

    try:
        client = openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an experienced QA engineer writing Selenium Python automation."},
                {"role": "system", "content": "Return ONLY raw Python code. No explanations. No markdown. No code blocks."},
                {"role": "system", "content": "Configure Chrome with these options: --headless=new, --no-sandbox, --disable-dev-shm-usage, --disable-gpu"},
                {"role": "system", "content": f"Use the URL: {url}"},
                {"role": "system", "content": f"Handle the following actions: {analysis}"},
                {"role": "user", "content": "Produce a Python Selenium script that tests each action and prints a result for each one."}
            ],
            max_tokens=1500
        )
        code = response.choices[0].message.content
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(l for l in lines if not l.startswith("```"))
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
