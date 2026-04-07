import os
from flask import Flask

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
TEMPLATES_FOLDER = os.path.join(os.path.dirname(__file__), "templates")
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")
STATIC_UPLOADS = os.path.join(STATIC_FOLDER, "uploads")

app = Flask(__name__, template_folder=TEMPLATES_FOLDER, static_folder=STATIC_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("APP_SECRET", "dev-secret")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_UPLOADS, exist_ok=True)

from app import routes
