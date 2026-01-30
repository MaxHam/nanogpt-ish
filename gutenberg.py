import csv
import gzip
from pathlib import Path
from io import StringIO
import requests

# 14MB CSV of all Gutenberg metadata
CSV_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
r = requests.get(CSV_URL)
text = gzip.decompress(r.content).decode("utf-8")
data = list(csv.DictReader(StringIO(text)))

print(f"Total books in catalog: {len(data)}")

out = Path("gutenberg_txts")

if not out.exists():
    out.mkdir(exist_ok=True)
    for book in data[:3]:  
        book_id = book["Text#"]
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        r = requests.get(url)
        if r.status_code == 200:
            (out / f"{book_id}.txt").write_text(r.text)
            print(f"Saved {book['Title']}")


# Concatenate all text files into one big corpus called corpus.txt
corpus_path = out / "corpus.txt"
with corpus_path.open("w", encoding="utf-8") as fout:
    for txt_file in out.glob("*.txt"):
        if txt_file.name == "corpus.txt":
            continue
        fout.write(txt_file.read_text(encoding="utf-8"))
        fout.write("\n")
print(f"Wrote corpus to {corpus_path}")