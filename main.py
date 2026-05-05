import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
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

        plan_path = os.path.join('output', 'plan.json')
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan = json.load(f)

        return render_template('results.html', plan=plan, target=target, email=email, pm_tool=pm_tool)

    except Exception as e:
        flash(f"❌ Error: {str(e)}", 'danger')
        return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
