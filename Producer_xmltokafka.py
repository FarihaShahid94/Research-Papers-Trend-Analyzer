import warnings
import time
import json
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from kafka import KafkaProducer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def safe_get_text(tag):
    return tag.text.strip() if tag and tag.text else "N/A"

xml_file_path = "C:/Users/DELL/OneDrive/Desktop/Desktop/University/BigData/Project/data/arxiv_combined_items.xml"
window_size = 10000
kafka_topic = "arxiv-papers"

def parse_arxiv_items_from_file():
    with open(xml_file_path, "rb") as f:
        soup = BeautifulSoup(f, "xml")
        return soup.find_all("item")

def main():
    print(" Loading items from local XML file...")
    items = parse_arxiv_items_from_file()
    total = len(items)
    print(f" Total items found: {total}")

    index = 0
    batch_count = 1

    while True:
        print(f"\n Sending batch {batch_count} [{index}:{index+window_size}]")

        batch = items[index: index + window_size]

        if not batch:
            print(" Reached end of file. Restarting from beginning...")
            index = 0
            continue

        for item in batch:
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

            producer.send(kafka_topic, value=paper)

        print(f" Batch {batch_count} sent with {len(batch)} items")
        batch_count += 1
        index += window_size

        print(" Sleeping for 300 seconds...\n")
        time.sleep(300)

if __name__ == "__main__":
    main()
