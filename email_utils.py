import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

def send_results_email(to_email: str, attachments: list, subject="Test Plan Results"):
    """
    Send an email with attachments.

    :param to_email: Recipient email address
    :param attachments: List of file paths to attach
    :param subject: Email subject
    """
    from_email = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")  # iCloud app-specific password

    if not from_email or not email_password:
        raise RuntimeError("EMAIL_ADDRESS / EMAIL_PASSWORD not set in environment")

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Email body
    body = "Dear user,\n\nPlease find attached the generated Test Plan results.\n\nBest regards."
    msg.attach(MIMEText(body, 'plain'))

    # Attach files
    for file_path in attachments:
        try:
            with open(file_path, "rb") as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
            msg.attach(part)
        except Exception as e:
            print(f"❌ Failed to attach file {file_path}: {e}")

    # Send email
    server = smtplib.SMTP('smtp.mail.me.com', 587)  # iCloud Mail
    try:
        server.starttls()
        server.login(from_email, email_password)
        server.send_message(msg)
        print(f"📧 Email sent successfully to {to_email}")
    finally:
        server.quit()
