# web/database.py
# SQLite database layer for users and analysis history.
# Passwords are hashed with bcrypt — never stored as plaintext.

import sqlite3
import os
import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firmnet.db")


def get_db():
    """Return a connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't already exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT    NOT NULL,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL,
            image_filename    TEXT    NOT NULL,
            original_filename TEXT    NOT NULL,
            predicted_class   TEXT    NOT NULL,
            fake_score        REAL    NOT NULL,
            real_score        REAL    NOT NULL,
            confidence        REAL    NOT NULL,
            freq_low          REAL,
            freq_mid          REAL,
            freq_high         REAL,
            fft_image_path    TEXT,
            gradcam_image_path TEXT,
            model_type        TEXT,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Migration: add gradcam_image_path to existing databases
    try:
        conn.execute("ALTER TABLE analyses ADD COLUMN gradcam_image_path TEXT")
        conn.commit()
    except Exception:
        pass  # Column already exists

    conn.commit()
    conn.close()


# ── User operations ──────────────────────────────────────────────────

def create_user(full_name: str, username: str, email: str, password: str):
    """Hash *password* with bcrypt and insert a new user row.

    Returns (success: bool, message: str).
    """
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (full_name, username, email, password_hash) "
            "VALUES (?, ?, ?, ?)",
            (full_name, username, email, password_hash),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as exc:
        msg = str(exc).lower()
        if "username" in msg:
            return False, "That username is already taken."
        if "email" in msg:
            return False, "That email is already registered."
        return False, "Registration failed — please try again."
    finally:
        conn.close()


def verify_user(login_id: str, password: str):
    """Authenticate by *username* or *email*.

    Returns a user dict on success, or ``None``.
    """
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (login_id, login_id),
    ).fetchone()
    conn.close()

    if user and bcrypt.checkpw(
        password.encode("utf-8"), user["password_hash"].encode("utf-8")
    ):
        return dict(user)
    return None


def get_user_by_id(user_id: int):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def username_exists(username: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT 1 FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return row is not None


def email_exists(email: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT 1 FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return row is not None


# ── Analysis operations ──────────────────────────────────────────────

def save_analysis(
    user_id, image_filename, original_filename, predicted_class,
    fake_score, real_score, confidence,
    freq_low=None, freq_mid=None, freq_high=None,
    fft_image_path=None, gradcam_image_path=None, model_type=None,
):
    """Persist one analysis record and return its id."""
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO analyses
           (user_id, image_filename, original_filename, predicted_class,
            fake_score, real_score, confidence,
            freq_low, freq_mid, freq_high, fft_image_path,
            gradcam_image_path, model_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, image_filename, original_filename, predicted_class,
         fake_score, real_score, confidence,
         freq_low, freq_mid, freq_high, fft_image_path,
         gradcam_image_path, model_type),
    )
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id


def get_user_analyses(user_id: int, limit: int = 50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: int, user_id: int):
    """Fetch a single analysis — only if it belongs to *user_id*."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_stats(user_id: int) -> dict:
    conn = get_db()
    total = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    fake = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE user_id = ? AND predicted_class = 'FAKE'",
        (user_id,),
    ).fetchone()[0]
    real = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE user_id = ? AND predicted_class = 'REAL'",
        (user_id,),
    ).fetchone()[0]
    conn.close()
    return {"total": total, "fake": fake, "real": real}
