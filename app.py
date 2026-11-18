from pathlib import Path
from functools import wraps
import sqlite3
import joblib

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "jobguard.db"
MODEL_PATH = BASE_DIR / "models" / "fake_job_model.joblib"

app = Flask(__name__)
app.secret_key = "replace-this-with-a-long-random-secret-key-before-deployment"

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_text TEXT NOT NULL,
                result TEXT NOT NULL,
                risk_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or len(password) < 6:
            flash("Enter your name, a valid email, and a password of at least 6 characters.", "danger")
            return render_template("register.html")

        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                    (name, email, generate_password_hash(password))
                )
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "danger")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))

        flash("Incorrect email or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE user_id = ?", (session["user_id"],)
        ).fetchone()[0]
        recent = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
            (session["user_id"],)
        ).fetchall()
    return render_template("dashboard.html", total=total, recent=recent, model_ready=model is not None)

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        job_text = request.form.get("job_text", "").strip()

        if len(job_text) < 30:
            flash("Please enter a more detailed job or internship description (at least 30 characters).", "warning")
            return render_template("analyze.html", job_text=job_text)

        if model is None:
            flash("The ML model is not ready. Add the dataset and run: python train_model.py", "danger")
            return render_template("analyze.html", job_text=job_text)

        probabilities = model.predict_proba([job_text])[0]
        classes = list(model.classes_)
        fake_index = classes.index(1)
        risk_score = float(probabilities[fake_index] * 100)
        prediction = "Potentially Suspicious" if risk_score >= 50 else "Likely Genuine"

        # Simple warning-term scan is supplementary; it is not the ML prediction.
        warning_terms = [
            "registration fee", "pay to apply", "processing fee",
            "security deposit", " whatsapp only", "guaranteed income",
            "no interview", "pay immediately"
        ]
        lower_text = job_text.lower()
        flagged_terms = [term.strip() for term in warning_terms if term.strip() in lower_text]

        with get_db() as conn:
            conn.execute(
                """INSERT INTO analyses (user_id, job_text, result, risk_score)
                   VALUES (?, ?, ?, ?)""",
                (session["user_id"], job_text, prediction, risk_score)
            )

        return render_template(
            "result.html",
            prediction=prediction,
            risk_score=round(risk_score, 2),
            flagged_terms=flagged_terms,
            job_text=job_text
        )

    return render_template("analyze.html", job_text="")

@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC",
            (session["user_id"],)
        ).fetchall()
    return render_template("history.html", rows=rows)

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

init_db()

if __name__ == "__main__":
    app.run(debug=True)
