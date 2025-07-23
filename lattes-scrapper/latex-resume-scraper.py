from bs4 import BeautifulSoup
import csv
import os

CSV_DELIMITER = ','
INSIDE_SEPARATOR = ' $<->$ '
INPUT_DIRECTORY = './input'

wanted_fields = [
    'Nome',
    'Lattes ID',
    'Resumo',
    'Formação acadêmica/titulação',
    'Áreas de atuação',
    'Idiomas',
    'Prêmios e títulos',
    'Produções',
    'Orientações',
]

def scraper(path):
    with open(path, encoding='utf8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')

    # first matching is enough
    name = soup.select_one('body > div.page.min-width > div.content-wrapper > div > div > div > div.infpessoa > h2').get_text(strip=True)
    lattes_id = soup.select_one('body > div.page.min-width > div.content-wrapper > div > div > div > div.infpessoa > ul > li:nth-child(2) > span:nth-child(2)').get_text(strip=True)
    summary = soup.select_one('body > div.page.min-width > div.content-wrapper > div > div > div > div:nth-child(4) > p').get_text(strip=True)

    # get other sections info
    sections = soup.find_all('div', { 'class': 'title-wrapper' })
    
    professor_data = {}

    professor_data['Nome'] = name
    professor_data['Lattes ID'] = lattes_id
    professor_data['Resumo'] = summary

    for section in sections:
        section = section.find_all(recursive=False)
        section_title = section[0].get_text(strip=True)

        if section_title == '':
            continue
        
        section_data = section[2].find_all(recursive=False)
        section_texts = []

        if section_title == 'Identificação':
            pass
        elif section_title == 'Endereço':
            pass
        elif section_title == 'Formação acadêmica/titulação':
            for i in range(0, len(section_data), 2):
                date = section_data[i].get_text(strip=True)
                description = section_data[i + 1].get_text(strip=True)

                section_texts.append(f'Data: {date}. {description}')
        elif section_title == 'Atuação Profissional':
            pass
        elif section_title == 'Projetos de pesquisa':
            pass
        elif section_title == 'Projetos de extensão':
            pass
        elif section_title == 'Projetos de ensino':
            pass
        elif section_title == 'Revisor de periódico':
            pass
        elif section_title == 'Áreas de atuação':
            for i in range(0, len(section_data), 2):
                description = section_data[i + 1].get_text(strip=True)
                section_texts.append(f'{description}')
        elif section_title == 'Idiomas':
            for i in range(0, len(section_data), 2):
                language = section_data[i].get_text(strip=True)
                description = section_data[i + 1].get_text(strip=True)

                section_texts.append(f'{language}: {description}')
        elif section_title == 'Prêmios e títulos':
            for i in range(0, len(section_data), 2):
                date = section_data[i].get_text(strip=True)
                description = section_data[i + 1].get_text(strip=True)

                section_texts.append(f'{date}: {description}')
        elif section_title == 'Produções':
            papers = soup.find_all('div', { 'class': 'artigo-completo' })

            for paper in papers:
                section_texts.append(paper.find_all(recursive=False)[1].get_text(strip=True))
        elif section_title == 'Bancas':
            pass
        elif section_title == 'Eventos':
            pass
        elif section_title == 'Orientações':
            orientations = section[2].find_all('div', { 'class': 'layout-cell layout-cell-11' })

            for orientation in orientations:
                section_texts.append(orientation.get_text(strip=True))
        elif section_title == 'Inovação':
            pass
        elif section_title == 'Educação e Popularização de C & T':
            pass
        elif section_title == 'Outras informações relevantes':
            pass
        else:
            # print(f'Unknown section: {section_title}')
            pass

        if len(section_title) > 0 and len(section_texts) > 0:
            section_text = INSIDE_SEPARATOR.join(section_texts)
            professor_data[section_title] = section_text

    with open(f'./output/{name} [{lattes_id}].csv', 'w', newline='', encoding='utf8') as csvfile:
        data = [wanted_fields, [professor_data.get(wanted_field, "") for wanted_field in wanted_fields]]

        writer = csv.writer(csvfile, delimiter=CSV_DELIMITER)
        writer.writerows(data)

for filename in os.listdir(INPUT_DIRECTORY):
    file_path = os.path.join(INPUT_DIRECTORY, filename)
    
    if os.path.isfile(file_path):
        scraper(file_path)
