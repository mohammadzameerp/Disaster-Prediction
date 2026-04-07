import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password_hash TEXT, created_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, path TEXT, user_id INTEGER, uploaded_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, kind TEXT, output TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

def create_user(email, password_hash):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO users (email, password_hash, created_at) VALUES (?,?,?)", (email, password_hash, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_user(email):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row

def log_upload(filename, path, user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO uploads (filename, path, user_id, uploaded_at) VALUES (?,?,?,?)", (filename, path, user_id, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def log_prediction(user_id, kind, output_json):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO predictions (user_id, kind, output, created_at) VALUES (?,?,?,?)", (user_id, kind, output_json, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
