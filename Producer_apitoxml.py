import os
import requests
from datetime import datetime
from bs4 import BeautifulSoup

categories = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
    "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
    "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
    "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
    "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

output_file = "data/arxiv_combined_items.xml"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def write_header():
    now_rfc = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<rss xmlns:arxiv="http://arxiv.org/schemas/atom" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" '
                'xmlns:atom="http://www.w3.org/2005/Atom" '
                'xmlns:content="http://purl.org/rss/1.0/modules/content/" '
                'version="2.0">\n')
        f.write('  <channel>\n')
        f.write('    <title>Combined arXiv updates</title>\n')
        f.write('    <link>http://arxiv.org/</link>\n')
        f.write('    <description>Combined RSS feed for multiple arXiv categories</description>\n')
        f.write('    <language>en-us</language>\n')
        f.write(f'    <lastBuildDate>{now_rfc}</lastBuildDate>\n')
        f.write('    <managingEditor>rss-help@arxiv.org</managingEditor>\n')
        f.write(f'    <pubDate>{now_rfc}</pubDate>\n')

def write_footer():
    with open(output_file, "a", encoding="utf-8") as f:
        f.write('  </channel>\n</rss>\n')

def fetch_and_append_items():
    with open(output_file, "a", encoding="utf-8") as f:
        for category in categories:
            print(f"Fetching {category}...")
            url = f"https://rss.arxiv.org/rss/{category}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "xml")
                    items = soup.find_all("item")
                    f.write(f"\n<!-- CATEGORY: {category} | TIMESTAMP: {datetime.utcnow().isoformat()} -->\n")
                    for item in items:
                        f.write(str(item))
                        f.write("\n")
                else:
                    print(f"Failed: {category} (HTTP {response.status_code})")
            except Exception as e:
                print(f"Error fetching {category}: {e}")
    print(" Done appending items.")


write_header()
fetch_and_append_items()
for _ in range(1000):
    fetch_and_append_items()
write_footer()
