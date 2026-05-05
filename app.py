from flask import Flask, render_template, request, redirect, session, send_from_directory
import sqlite3
import os
import joblib
import re

# ---------------- LOAD AI MODEL ----------------

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------- APP SETUP ----------------

app = Flask(__name__)

# Secure secret key (for Render)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ---------------- DATABASE ----------------

def get_db():
    return sqlite3.connect("database.db", check_same_thread=False)

# ---------------- AI HELPERS ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_safe_text(text):
    safe_words = ["hello", "hi", "hey", "good morning", "good evening"]
    return any(word in text for word in safe_words)

def is_positive_text(text):
    positive_words = [
        "happy", "joy", "love", "peace", "music",
        "relax", "calm", "motivated", "inspired"
    ]
    return any(word in text for word in positive_words)

def is_negative(text):
    text = clean_text(text)

    if len(text.split()) < 4:
        return False

    if is_safe_text(text) or is_positive_text(text):
        return False

    negative_keywords = [
        "tired of life", "give up", "hopeless",
        "depressed", "worthless", "sad",
        "life is meaningless", "end my life"
    ]

    for kw in negative_keywords:
        if kw in text:
            return True

    vec = vectorizer.transform([text])
    return model.predict(vec)[0] == 1

def add_bot_reply(cur, post_id):
    message = (
        "🤖 If you're feeling emotionally distressed, "
        "please reach out to someone you trust or a professional. "
        "You are not alone. Helpline: 9152987821"
    )

    cur.execute(
        "INSERT INTO comments (post_id, parent_id, username, comment) VALUES (?, ?, ?, ?)",
        (post_id, None, "AI_BOT", message)
    )

# ---------------- STATIC ----------------

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

# ---------------- AUTH ----------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = get_db()
        cur = db.cursor()

        cur.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (request.form["username"], request.form["password"])
        )
        user = cur.fetchone()
        db.close()

        if user:
            session["user"] = request.form["username"]
            return redirect("/home")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        db = get_db()
        cur = db.cursor()

        cur.execute(
            "INSERT INTO users VALUES (?, ?)",
            (request.form["username"], request.form["password"])
        )

        db.commit()
        db.close()
        return redirect("/")

    return render_template("register.html")

# ---------------- HOME ----------------

@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    if request.method == "POST":
        tweet = request.form["tweet"]

        if len(tweet) <= 280:
            cur.execute(
                "INSERT INTO tweets (username, tweet) VALUES (?, ?)",
                (session["user"], tweet)
            )

            post_id = cur.lastrowid

            if is_negative(tweet):
                add_bot_reply(cur, post_id)

            db.commit()

    search = request.args.get("search", "")

    cur.execute("""
        SELECT t.id, t.username, t.tweet, COUNT(l.post_id)
        FROM tweets t
        LEFT JOIN likes l ON t.id = l.post_id
        WHERE t.tweet LIKE ?
        GROUP BY t.id
        ORDER BY t.id DESC
    """, ('%' + search + '%',))

    tweets = cur.fetchall()

    cur.execute("""
        SELECT id, post_id, parent_id, username, comment
        FROM comments
        ORDER BY id ASC
    """)
    comments = cur.fetchall()

    db.close()

    return render_template("home.html", tweets=tweets, comments=comments)

# ---------------- COMMENT ----------------

@app.route("/comment/<int:post_id>", methods=["POST"])
def comment(post_id):
    if "user" not in session:
        return redirect("/")

    text = request.form["comment"]
    parent_id = request.form.get("parent_id")

    if not text.strip():
        return redirect("/home")

    db = get_db()
    cur = db.cursor()

    cur.execute(
        "INSERT INTO comments (post_id, parent_id, username, comment) VALUES (?, ?, ?, ?)",
        (post_id, parent_id, session["user"], text)
    )

    if is_negative(text):
        add_bot_reply(cur, post_id)

    db.commit()
    db.close()

    return redirect("/home")

# ---------------- LIKE ----------------

@app.route("/like/<int:post_id>")
def like(post_id):
    if "user" not in session:
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT * FROM likes WHERE post_id=? AND username=?",
        (post_id, session["user"])
    )

    if cur.fetchone():
        cur.execute(
            "DELETE FROM likes WHERE post_id=? AND username=?",
            (post_id, session["user"])
        )
    else:
        cur.execute(
            "INSERT INTO likes VALUES (?, ?)",
            (post_id, session["user"])
        )

    db.commit()
    db.close()
    return redirect("/home")

# ---------------- DELETE ----------------

@app.route("/delete/<int:post_id>")
def delete(post_id):
    if "user" not in session:
        return redirect("/")

    db = get_db()
    cur = db.cursor()

    if session["user"] == "admin":
        cur.execute("DELETE FROM tweets WHERE id=?", (post_id,))
    else:
        cur.execute(
            "DELETE FROM tweets WHERE id=? AND username=?",
            (post_id, session["user"])
        )

    db.commit()
    db.close()
    return redirect("/home")

# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run()