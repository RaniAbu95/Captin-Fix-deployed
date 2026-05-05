from langchain_openai import ChatOpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
from dotenv import load_dotenv
import time

#from Page_sourse import page_source, llm_suggestions

load_dotenv(dotenv_path="properties.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#create function to anaylayz the page using selenium
# def analyz_html_with_llm(html:str)->str:
#     templete=f""" i want to give you a code html for page internet {html} identify the elements"""
#     prompt=ChatPromptTemplate.from_template(templete)
#     chain=prompt|llm
#     response=chain.invoke({'html',html})
#     return response.content
#
# def generate_selenium_script(actions:list)->str:
#
#     """get a llm and you can know selenium"""
#
#     template=f"""you are an expert in qa and generating script with 20 years experience,requerd to test a webpage with selenium return only a valid python script (no explenation and mark)use this url:"file:///C:/Users/Dell/Downloads/ActionChainsEx.Html"
#      locate elemants by their id ,class name or css selector ,etc...,and perform the following actions:{actions} the script must:
#      import selenium moudoles
#      open chrome using driver=webdriver.Chrome()
#      navigate to the url
#      locate the elements correctly
#      perform all the actions"""
#     prompt=ChatPromptTemplate.from_template(template)
#     chain=prompt|llm
#     response=chain.invoke({'html',actions})
#     return response.content


def process_target_data(target_url):
    """
    This function processes the target URL received from main.py.
    """
    print(f"The URL '{target_url}' has been received by the test.py file.")
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(target_url)
    driver.quit()


def run_test(url):
    print("RUN TEST STARTED")

    try:
        import os
        print("chromium exists:", os.path.exists("/usr/bin/chromium"))

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.binary_location = "/usr/bin/chromium"

        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)

        driver.get(url)
        print("TITLE:", driver.title)

    except Exception as e:
        print("❌ SELENIUM ERROR:", str(e))

#producion script automation






