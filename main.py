import os
from flask import Flask, render_template, request, redirect, url_for, flash
from testPlan import process_target_data
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

    # Basic validation
    errors = []
    if not target:
        errors.append("⚠️ Please provide a target URL")
    if not email:
        errors.append("⚠️ Please provide an email address")

    if errors:
        for e in errors:
            flash(e, 'danger')
        return redirect(url_for('index'))

    # Call the function from the test.py file, passing the 'target' variable
    # process_target_data(target)  # <-- Pass the variable here
    run_planner(target, depth=int(depth), num_tests=int(num_cases), email=email, pm=pm_tool)
    # Success message
    flash(f"✅ Received input: target={target}, depth={depth}, num_cases={num_cases}, email={email}, pm_tool={pm_tool}", "success")

    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)