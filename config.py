import os
from dotenv import load_dotenv  # deploy-cost check 2026-05-17

# טוען מהקובץ המקומי (לא נמצא ב-GitHub)
load_dotenv("properties.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BROWSERLESS_TOKEN = os.getenv("BROWSERLESS_TOKEN", "")
BROWSERLESS_URL = os.getenv("BROWSERLESS_URL", "https://chrome.browserless.io")