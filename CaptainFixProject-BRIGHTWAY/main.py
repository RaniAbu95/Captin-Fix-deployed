import os
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, send_file, abort
)
from werkzeug.security import generate_password_hash, check_password_hash

from database import db, User, Result
from testPlan import process_target_data
from planner import run_planner

import subprocess
import sys



app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

# -----------------------------
# Authentication
# -----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash("⚠️ Email already registered", "danger")
        else:
            new_user = User(email=email,
                            password=generate_password_hash(password, method="pbkdf2:sha256", salt_length=16))
            db.session.add(new_user)
            db.session.commit()
            flash("✅ Registration successful, please login.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash("✅ Logged in successfully", "success")
            return redirect(url_for('index'))
        flash("⚠️ Invalid credentials", "danger")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("ℹ️ Logged out", "info")
    return redirect(url_for('login'))

# -----------------------------
# Main Pages
# -----------------------------
@app.route('/', methods=['GET'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    if 'user_id' not in session:
        flash("⚠️ You must login first", "danger")
        return redirect(url_for('login'))

    target = request.form.get('target')
    depth = request.form.get('depth', 1)
    num_cases = request.form.get('num_cases', 5)
    pm_tool = request.form.get('pm_tool')
    instructions = request.form.get('instructions', '')

    user = db.session.get(User, session['user_id'])
    email = user.email

    # Basic validation
    errors = []
    if not target:
        errors.append("⚠️ Please provide a target URL")
    if not email:
        errors.append("⚠️ Please provide an email address")

    try:
        depth = int(depth)
        num_cases = int(num_cases)
    except ValueError:
        errors.append("⚠️ Depth and number of cases must be integers")

    if errors:
        for e in errors:
            flash(e, 'danger')
        return redirect(url_for('index'))

    try:
        process_target_data(target)
        flash(f"📝 Your test plan for {target} is being generated...", "info")



        file_paths = run_planner(
            target, depth=depth, num_tests=num_cases, email=email, pm=pm_tool,instructions=instructions
        )
    except Exception as e:
        flash(f"❌ Error generating test plan: {str(e)}", "danger")
        return redirect(url_for('index'))

    new_result = Result(
        user_id=user.id,
        target_url=target,
        json_path=file_paths.get('json', ''),
        excel_path=file_paths.get('excel', ''),
        created_at=datetime.utcnow()
    )
    db.session.add(new_result)
    db.session.commit()

    flash("✅ Test Plan generated and saved successfully.", "success")
    try:
        # Run executor.py as a separate process
        subprocess.run([sys.executable, "executor.py"], check=True)
        flash("✅ Executor ran successfully.", "success")
    except subprocess.CalledProcessError as e:
        flash(f"⚠️ Error running executor: {e}", "danger")

    return redirect(url_for('index'))


@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("⚠️ You must login first", "danger")
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])
    results = Result.query.filter_by(user_id=user.id)\
                          .order_by(Result.created_at.desc()).all()
    return render_template('history.html', results=results)


@app.route('/download/<int:result_id>/<file_type>')
def download_file(result_id, file_type):
    result = db.session.get(Result, result_id)
    if not result:
        abort(404)

    if result.user_id != session.get('user_id'):
        abort(403)

    if file_type == 'json':
        path = result.json_path
    elif file_type == 'excel':
        path = result.excel_path
    else:
        abort(404)

    if not os.path.exists(path):
        abort(404, description="File not found on server.")
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)