import requests
import re
from bs4 import BeautifulSoup, NavigableString
import pandas as pd

def get_previous_sibling(element):
    sibling = element.previous_sibling
    if sibling == "\n":
        return get_previous_sibling(sibling)
    else:
        return sibling

def get_subject(info):
    res = get_previous_sibling(get_previous_sibling(info.parent.parent.parent))
    if res.name == "p":
        return res.get_text().strip()
    # TODO: get_subject doesn't work for this specific subject
    else:
        return "Lato Sensu"

def remove_empty_tags(soup):
    for x in soup.find_all():
        if len(x.get_text(strip=True)) == 0 and x.name not in ["br", "img"]:
            x.extract()
    
    return soup

faq = {"Subject": [], "Question": [], "Answer": []}
URL = "https://www.saa.unb.br/faq-geral"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")

soup = BeautifulSoup(str(soup).replace("Este endereço de email está sendo protegido de spambots. Você precisa do JavaScript ativado para vê-lo.", "saaatendimento@unb.br"), "html.parser")

remove_empty_tags(soup)

all_info = soup.find_all("div", class_="accordion-inner panel-body")

for info in all_info:
    subject = get_subject(info)
    question = ""
    subtopic = ""

    for section in info.children:
        if section.name == "h2":
            subtopic = section.get_text()
        elif section.name == "ul":
            for c in section.children:
                if not isinstance(c, NavigableString) and c.has_attr("data-listid"):
                    question = c.get_text()
        elif section.name:
            answer = section.get_text()
            
            if section.name == "br":
                answer = ""
                for s in section.next_siblings:
                    answer += s.get_text()

            if (len(faq["Question"]) == 0 or faq["Question"][-1] != subtopic + " " + question):
                faq["Subject"].append(subject)
                faq["Question"].append(subtopic + " " + question)
                faq["Answer"].append(answer)
                print(question)
            elif question != "":
                faq["Answer"][-1] += answer

dataframe = pd.DataFrame(data=faq)
# remove last line
dataframe = dataframe.iloc[:-1 , :]
dataframe.to_csv("Planilhas/FAQ SAA.csv", index=False)