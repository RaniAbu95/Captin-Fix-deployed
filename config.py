import os
from dotenv import load_dotenv  # deploy-cost check 2026-05-17

# טוען מהקובץ המקומי (לא נמצא ב-GitHub)
load_dotenv("properties.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LT_USERNAME = os.getenv("LT_USERNAME", "")
LT_ACCESS_KEY = os.getenv("LT_ACCESS_KEY", "")