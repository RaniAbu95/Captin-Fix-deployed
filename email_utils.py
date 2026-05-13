import os
import base64
import requests


def send_results_email(to_email: str, attachments: list, subject="Test Plan Results"):
    """
    Send an email with attachments via the Resend HTTP API.

    Resend is used (rather than SMTP) because most PaaS providers block
    outbound SMTP on ports 25/465/587. The HTTP API bypasses that block.

    :param to_email: Recipient email address
    :param attachments: List of file paths to attach
    :param subject: Email subject
    """
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise RuntimeError("RESEND_API_KEY not set in environment")

    from_email = os.getenv("FROM_EMAIL", "Captain Fix <onboarding@resend.dev>")

    body_text = "Dear user,\n\nPlease find attached the generated Test Plan results.\n\nBest regards."

    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": body_text,
        "attachments": [],
    }

    for file_path in attachments:
        try:
            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            payload["attachments"].append({
                "filename": os.path.basename(file_path),
                "content": encoded,
            })
        except Exception as e:
            print(f"❌ Failed to attach file {file_path}: {e}")

    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        raise RuntimeError(f"Resend API error {resp.status_code}: {resp.text}")

    print(f"📧 Email sent successfully to {to_email}")
