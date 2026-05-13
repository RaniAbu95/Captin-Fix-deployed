# jira_utils.py
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import json

# -----------------------------
# Load Jira credentials from .env
# -----------------------------
load_dotenv(".env")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")
JIRA_URL = os.getenv("JIRA_URL")  # Base URL, e.g., "https://yourdomain.atlassian.net"

if not all([JIRA_EMAIL, JIRA_TOKEN, JIRA_URL]):
    raise ValueError("❌ Jira credentials not set in .env")

# -----------------------------
# Helper: Convert plain text to Atlassian Document Format (ADF)
# -----------------------------
def to_adf(text: str):
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
    }

# -----------------------------
# Create Jira Issue
# -----------------------------
def create_jira_issue(summary: str, description: str, project_key: str, issue_type: str = "Bug"):
    """
    Creates a Jira issue using ADF for the description.

    Args:
        summary (str): Short title for the issue.
        description (str): Detailed description (plain text is fine).
        project_key (str): Jira project key.
        issue_type (str): Type of issue (default: "Bug").
    Returns:
        dict: Response JSON from Jira API.
    """
    url = f"{JIRA_URL}/rest/api/3/issue"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": to_adf(description),  # <-- convert to ADF
            "issuetype": {"name": issue_type}
        }
    }

    response = requests.post(url, json=payload, headers=headers, auth=auth)
    if response.status_code in (200, 201):
        print(f"✅ Jira issue created: {summary}")
        return response.json()
    else:
        print(f"❌ Failed to create Jira issue: {response.status_code} {response.text}")
        return None

# -----------------------------
# Optional: Attach file to Jira issue
# -----------------------------
def attach_file_to_issue(issue_key: str, file_path: str):
    """
    Attach a file to an existing Jira issue.

    Args:
        issue_key (str): Jira issue key (e.g., "PROJ-123").
        file_path (str): Path to the file to attach.
    """
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/attachments"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
    headers = {
        "X-Atlassian-Token": "no-check"
    }

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        response = requests.post(url, headers=headers, auth=auth, files=files)

    if response.status_code in (200, 201):
        print(f"✅ File attached: {file_path} to {issue_key}")
    else:
        print(f"❌ Failed to attach file: {response.status_code} {response.text}")
