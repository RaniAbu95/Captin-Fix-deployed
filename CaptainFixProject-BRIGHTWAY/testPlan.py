import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI



# Process target URL using Selenium
def process_target_data(target_url: str):
    print(f"The URL '{target_url}' has been received and will be opened in Chrome.")

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(target_url)
        time.sleep(2)
        # page_source = driver.page_source
    finally:
        driver.quit()
