import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_previous_sibling(element):
    sibling = element.previous_sibling
    if sibling == "\n":
        return get_previous_sibling(sibling)
    else:
        return sibling

def page_data(soup, news, URL):
    info = soup.find_all("a", itemprop="url")
    summaries = soup.find_all(itemprop="introtext")
    dates = soup.find_all("div", class_="items-row")

    # end webscrapping
    if (len(info) == 0):
        exit()

    for text in dates:
        date = get_previous_sibling(text).get_text()
        if text == "":
            news["Date"].append("")    
        else:
            news["Date"].append(date)

    for title in info:
        news["Title"].append(title.get_text().strip())

    for summary in summaries:
        news["Summary"].append(summary.get_text().strip())

    for link in info:
        news["Link"].append(URL + link.get("href"))

        sub_page = requests.get(URL + link.get("href"))
        sub_soup = BeautifulSoup(sub_page.content, "html.parser")

        body = sub_soup.find("div", class_="item-page")

        news["Text"].append(body.get_text().strip())        

if __name__ == "__main__":
    news = { "Date": [], "Title": [], "Summary": [], "Text": [], "Link": [] }
    URL = "https://www.cic.unb.br/informacoes/noticias"

    news_number = 0
    dataframe = pd.DataFrame(data=news)
    while (True):
        try:
            page = requests.get(URL + "/?start=" + str(news_number))
            news_number += 10
            soup = BeautifulSoup(page.content, "html.parser")
            page_data(soup, news, URL)
            print("Processing page ", news_number // 10)

            dataframe = pd.DataFrame(data=news)
            dataframe.to_csv("Planilhas/Notícias CIC.csv", index=False)
        except Exception as err:
            print(err)
            break