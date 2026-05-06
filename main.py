import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from planner import run_planner

app = Flask(__name__)
app.secret_key = "dev-secret-key-123"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    target = request.form.get('target')
    depth = request.form.get('depth')
    num_cases = request.form.get('num_cases')
    email = request.form.get('email')
    pm_tool = request.form.get('pm_tool')

    if not target:
        flash("⚠️ Please provide a target URL", 'danger')
        return redirect(url_for('index'))
    if not email:
        flash("⚠️ Please provide an email address", 'danger')
        return redirect(url_for('index'))

    try:
        run_planner(target, depth=int(depth), num_tests=int(num_cases), email=email, pm=pm_tool)

        # Pre-warm executor HTML cache so Run Test never needs to launch Chrome again
        from executor import extract_full_html as _warm
        _warm(target)

        plan_path = os.path.join('output', 'plan.json')
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan = json.load(f)

        return render_template('results.html', plan=plan, target=target, email=email, pm_tool=pm_tool)

    except Exception as e:
        flash(f"❌ Error: {str(e)}", 'danger')
        return redirect(url_for('index'))




@app.route('/health', methods=['GET', 'POST'])
def health():
    import traceback
    results = {}

    # Check Chrome
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        driver.get("https://www.google.com")
        results["chrome"] = f"OK — title: {driver.title}"
        driver.quit()
    except Exception:
        results["chrome"] = traceback.format_exc()

    # Check OpenAI key
    from config import OPENAI_API_KEY
    results["openai_key"] = "present" if OPENAI_API_KEY else "MISSING"

    # Check output dir
    try:
        os.makedirs("output", exist_ok=True)
        test_path = os.path.join("output", "_health_check.txt")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        results["output_dir"] = "writable"
    except Exception as e:
        results["output_dir"] = str(e)

    return jsonify(results)


@app.route('/download/<filename>')
def download(filename):
    from flask import send_file, abort
    allowed = {'plan.json': 'output/plan.json', 'Plan.xlsx': 'output/Plan.xlsx'}
    if filename not in allowed:
        abort(404)
    path = os.path.join(os.getcwd(), allowed[filename])
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@app.route('/download/screenshot/<case_id>')
def download_screenshot(case_id):
    from flask import send_file, abort
    import re
    if not re.match(r'^[\w\-]+$', case_id):
        abort(400)
    path = os.path.join(os.getcwd(), 'screen', 'screenshots', f'{case_id}.png')
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=f'{case_id}_failure.png')


@app.route('/generate-code', methods=['POST'])
def generate_code():
    from executor import generate_test_files
    data = request.get_json()
    case = data.get('case')
    website = data.get('website', '')
    if not case or not website:
        return jsonify({"error": "case and website are required"}), 400
    plan = {"cases": [case], "website": website}
    try:
        test_files = generate_test_files(plan)
        if not test_files:
            return jsonify({"error": "Failed to generate code"}), 500
        case_id, file_path = test_files[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return jsonify({"code": code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/run-test', methods=['POST'])
def run_test():
    from executor import generate_test_files, run_test_file
    data = request.get_json()
    case = data.get('case')
    website = data.get('website', '')
    if not case or not website:
        return jsonify({"error": "case and website are required"}), 400

    plan = {"cases": [case], "website": website}
    try:
        test_files = generate_test_files(plan)
        if not test_files:
            return jsonify({"error": "Failed to generate test file"}), 500

        case_id, file_path = test_files[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        result = run_test_file(case_id, file_path)
        return jsonify({"code": code, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
