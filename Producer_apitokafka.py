import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from kafka import KafkaProducer
import requests
from bs4 import BeautifulSoup
import time
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  
)


categories = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
    "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
    "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
    "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
    "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

print("Kafka producer started...")


def safe_get_text(tag):
    return tag.text.strip() if tag and tag.text else "N/A"


def fetch_arxiv_papers():
    for category in categories:
        print(f"\nFetching {category}...")

        feed_url = f"https://rss.arxiv.org/rss/{category}"
        response = requests.get(feed_url)
        soup = BeautifulSoup(response.content, "xml")  
        items = soup.find_all('item')

        for item in items:
            categories_list = [cat.text for cat in item.find_all("category")]

            authors_raw = item.find('dc:creator')
            if authors_raw:
                authors_list = [author.strip() for author in authors_raw.text.split(",")]
            else:
                authors_list = []

            paper = {
                "title": safe_get_text(item.title),
                "authors": authors_list,        
                "summary": safe_get_text(item.description),
                "link": safe_get_text(item.link),
                "published": safe_get_text(item.pubDate),
                "categories": categories_list
}


            producer.send('arxiv-papers', value=paper)
            print(f"Sent: {paper['title']}")

fetch_arxiv_papers()

batch_count = 0  


batch_count += 1
print(f"All papers fetched - batch number: {batch_count}")



while True:
    fetch_arxiv_papers()
    print("Sleeping for 300 seconds...\n")
    time.sleep(300)
