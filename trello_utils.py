# trello_utils.py
import os
import requests
from dotenv import load_dotenv

# -----------------------------
# Load Trello credentials from .env
# -----------------------------
load_dotenv(".env")
TRELLO_KEY = os.getenv("TRELLO_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID")
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID")

if not all([TRELLO_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID, TRELLO_LIST_ID]):
    raise ValueError("❌ Trello credentials not set in .env")


# -----------------------------
# Create Trello Card
# -----------------------------
def create_trello_card(name: str, desc: str, list_id: str):
    """
    Create a Trello card in the specified list.
    """
    url = f"https://api.trello.com/1/cards"
    params = {
        "key": TRELLO_KEY,
        "token": TRELLO_TOKEN,
        "idList": TRELLO_LIST_ID,
        "name": name,
        "desc": desc
    }
    response = requests.post(url, params=params)
    if response.status_code in (200, 201):
        print(f"✅ Trello card created: {name}")
        return response.json()
    else:
        print(f"❌ Failed to create Trello card: {response.status_code} {response.text}")
        return None


# -----------------------------
# Attach file to Trello card (optional)
# -----------------------------
def attach_file_to_card(card_id: str, file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    url = f"https://api.trello.com/1/cards/{card_id}/attachments"
    params = {"key": TRELLO_KEY, "token": TRELLO_TOKEN}
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, params=params, files=files)

    if response.status_code in (200, 201):
        print(f"✅ File attached: {file_path} to card {card_id}")
    else:
        print(f"❌ Failed to attach file: {response.status_code} {response.text}")
