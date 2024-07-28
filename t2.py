import asyncio
import aiohttp
from bs4 import BeautifulSoup
import sqlite3


# Define a function to collect data from the specified websites
async def collect_data(websites):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for website in websites.split(','):
            task = asyncio.create_task(fetch(session, website))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
    return results
        

# Define a function to fetch data from a single website
async def fetch(session, website):
    async with session.get(website) as response:
        html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').text
        content = soup.find('div', class_='term-page--news-article--item--descr').text
        url = website
        return title, content, url


websites = input('Enter the websites you want to receive updates from (separated by commas)')



articles = asyncio.run(collect_data(websites))
info_list = []
for article in articles:
    info = {'title': article[1], 'content': article[2], 'url': article[3]}
    info_list.append(info)

# Define a function to summarize the extracted information
def summarize_info(info):
    # Use summarization algorithms to condense the extracted information
    # Return a concise summary
    return info['title'] + ': ' + info['content'][:100]

summaries = []
for info in info_list:
   print(info)