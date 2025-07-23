import requests
import re
from bs4 import BeautifulSoup, NavigableString, element
import pandas as pd

def extract_questions(url, faq):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    all_info = soup.find("div", class_="item-page")

    questions = re.findall(r'(?:[a-z]|\d+\.*\d*)(?:\.|-|\))* .+\?', all_info.get_text())
    answers = re.split(r'(?:[a-z]|\d+\.*\d*)(?:\.|-|\))* .+\?', all_info.get_text())[1:]

    subject = soup.find("h2", itemprop="name").get_text()

    for i in range(len(questions)):
        faq["Subject"].append(subject)
        faq["Question"].append(questions[i])
        faq["Answer"].append(answers[i])

    return faq

URL = "https://www.deg.unb.br"
page = requests.get(URL + "/perguntas")
soup = BeautifulSoup(page.content, "html.parser")

urls = [url.findChild("a" , recursive=False) for url in soup.findAll("div", class_="box2")]

faq = {"Subject": [], "Question": [], "Answer": []}

for url in urls:
    # TODO if it redirects to an external website 
    if not str(url["href"]).startswith("/"):
        continue

    faq = extract_questions(URL + url["href"], faq)

dataframe = pd.DataFrame(data=faq)
dataframe.to_csv("Planilhas/FAQ DEG.csv", index=False)