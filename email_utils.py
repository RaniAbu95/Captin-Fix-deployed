import os
import base64
import requests


def send_results_email(to_email: str, attachments: list, subject="Test Plan Results"):
    """
    Send an email with attachments via the Brevo HTTP API.

    Brevo is used (rather than SMTP) because most PaaS providers block
    outbound SMTP on ports 25/465/587. The HTTP API bypasses that block.

    Free tier: 300 emails/day. Sender email must be verified in Brevo's
    dashboard (Senders & IP → Senders → Add a sender).

    :param to_email: Recipient email address
    :param attachments: List of file paths to attach
    :param subject: Email subject
    """
    api_key = os.getenv("BREVO_API_KEY")
    if not api_key:
        raise RuntimeError("BREVO_API_KEY not set in environment")

    from_email = os.getenv("FROM_EMAIL", "raniab25@icloud.com")
    from_name = os.getenv("FROM_NAME", "Captain Fix")

    body_text = "Dear user,\n\nPlease find attached the generated Test Plan results.\n\nBest regards."

    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": body_text,
        "attachment": [],
    }

    for file_path in attachments:
        try:
            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            # Brevo rejects .json attachments (not on their extension whitelist).
            # Rename to .txt — content is identical, recipient can rename back.
            name = os.path.basename(file_path)
            if name.lower().endswith(".json"):
                name = name[:-5] + ".txt"
            payload["attachment"].append({
                "name": name,
                "content": encoded,
            })
        except Exception as e:
            print(f"❌ Failed to attach file {file_path}: {e}")

    resp = requests.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        raise RuntimeError(f"Brevo API error {resp.status_code}: {resp.text}")

    print(f"📧 Email sent successfully to {to_email}")
