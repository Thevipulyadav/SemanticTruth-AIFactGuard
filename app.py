import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import database as db
import ml_engine

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "semantictruth_aifactguard_secure_key_2026")

# Context processor for templates (injects auth state)
@app.context_processor
def inject_globals():
    user = None
    if "user_id" in session:
        user = db.get_user_by_id(session["user_id"])
    return {
        "current_user": user
    }

# ================= 1. LANDING PAGE =================
@app.route("/")
def index():
    return render_template("landing.html")

# ================= 2. DETECTOR PAGE =================
@app.route("/detector", methods=["GET", "POST"])
def detector():
    result = None
    input_text = ""
    input_url = ""

    if request.method == "POST":
        input_text = request.form.get("news", "").strip()
        input_url = request.form.get("url", "").strip()

        try:
            result = ml_engine.analyze_news_content(text=input_text, url=input_url)

            # If user is logged in, save to database history
            if "user_id" in session:
                db.save_analysis(
                    user_id=session["user_id"],
                    news_title=result["title"],
                    news_preview=result["preview"],
                    source_url=input_url or (result["domain"] if result.get("domain") else None),
                    model_name="FactGuard Real-Time Engine",
                    prediction=result["status_label"],
                    confidence=result["confidence"]
                )

        except Exception as e:
            flash(str(e), "danger")

    return render_template(
        "detector.html",
        result=result,
        input_text=input_text,
        input_url=input_url
    )

# ================= 3. REAL-TIME LIVE API =================
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or request.form
    input_text = data.get("news", "").strip()
    input_url = data.get("url", "").strip()

    try:
        result = ml_engine.analyze_news_content(text=input_text, url=input_url)

        # If user is logged in, save to database history
        if "user_id" in session:
            db.save_analysis(
                user_id=session["user_id"],
                news_title=result["title"],
                news_preview=result["preview"],
                source_url=input_url or (result["domain"] if result.get("domain") else None),
                model_name="FactGuard Real-Time Engine",
                prediction=result["status_label"],
                confidence=result["confidence"]
            )

        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ================= 4. AUTH: REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("detector"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "warning")
            return render_template("register.html", username=username, email=email)

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html", username=username, email=email)

        success, res = db.register_user(username, email, password)
        if success:
            flash("Account created successfully! Please sign in.", "success")
            return redirect(url_for("login"))
        else:
            flash(res, "danger")
            return render_template("register.html", username=username, email=email)

    return render_template("register.html")

# ================= 5. AUTH: LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("detector"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        if not identifier or not password:
            flash("Please enter your username/email and password.", "warning")
            return render_template("login.html", identifier=identifier)

        success, user_or_err = db.authenticate_user(identifier, password)
        if success:
            session["user_id"] = user_or_err["id"]
            session["username"] = user_or_err["username"]
            flash(f"Welcome back, {user_or_err['username']}!", "success")
            next_url = request.args.get("next") or url_for("detector")
            return redirect(next_url)
        else:
            flash(user_or_err, "danger")
            return render_template("login.html", identifier=identifier)

    return render_template("login.html")

# ================= 6. AUTH: LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("login"))

# ================= 7. USER DASHBOARD / HISTORY =================
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please log in to view your scan history and dashboard.", "warning")
        return redirect(url_for("login", next=url_for("dashboard")))

    history = db.get_user_history(session["user_id"], limit=50)

    total_scans = len(history)
    real_count = sum(1 for h in history if "REAL" in (h["prediction"] or ""))
    fake_count = sum(1 for h in history if "FAKE" in (h["prediction"] or "") or "UNVERIFIED" in (h["prediction"] or ""))
    avg_conf = (sum(h["confidence"] for h in history) / total_scans) if total_scans > 0 else 0.0

    stats = {
        "total_scans": total_scans,
        "real_count": real_count,
        "fake_count": fake_count,
        "avg_confidence": round(avg_conf, 1)
    }

    return render_template("dashboard.html", history=history, stats=stats)

# ================= 8. CLEAR HISTORY =================
@app.route("/clear-history", methods=["POST"])
def clear_history():
    if "user_id" in session:
        db.clear_user_history(session["user_id"])
        flash("Your verification history has been cleared.", "info")
    return redirect(url_for("dashboard"))

# ================= 9. API HEALTH =================
@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "SemanticTruth AI FactGuard Intelligence Real-Time Engine"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
