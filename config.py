import os
from dotenv import load_dotenv

# טוען מהקובץ המקומי (לא נמצא ב-GitHub)
load_dotenv("properties.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")