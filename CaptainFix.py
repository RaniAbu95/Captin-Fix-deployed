from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from config import OPENAI_API_KEY

# ---- הגדרת OpenAI ----
Ai_agent = OpenAI(api_key=OPENAI_API_KEY)

def analyze_html_with_llm(html):
    prompt = f"""
    אני נותן לך קוד HTML של דף אינטרנט:
    {html}

    זהה את האלמנטים הפעילים (כפתורים, קישורים, טפסים)
    ותאר את הפעולות האפשריות שניתן לבצע עליהם.
    החזר רשימה מסודרת לפי סוג האלמנט.
    """

    response = Ai_agent.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "אתה עוזר לאוטומציה עם Selenium."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


# ---- Selenium ----
driver = webdriver.Chrome()
# the path of the file
url = "file:///Users/raniaburaia/PycharmProjects/Captain-Fix/ActionChainsEx.Html"
driver.get(url)
driver.maximize_window()

page_source = driver.page_source
llm_suggestions = analyze_html_with_llm(page_source)

driver.quit()

print("\nה-LLM מציע את הפעולות הבאות:\n")
print(llm_suggestions)

confirm = input("\nרוצה לבצע את הפעולות? (y/n): ")

if confirm.lower() == "y":
    print("\nמייצר קוד אוטומציה....\n")


    response = Ai_agent.chat.completions.create (
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an experienced QA engineer writing Selenium Python automation."},
            {"role": "system", "content": "Return ONLY raw Python code. No explanations. No markdown."},
            {"role": "system", "content": "the operation system is macOS."},
            {"role": "system", "content": "Use the URL: "+url},
            {"role": "system", "content": "Handle the following actions: " + llm_suggestions},
            {"role": "user", "content": "Produce a Python Selenium script and maximize the window."}
        ],
        max_tokens=500
    )

    script_code = response.choices[0].message.content

    print("\nהקוד שנוצר:\n")
    print(script_code)

    exec(script_code)

else:
    print("\nביצוע הפעולות בוטל.")