import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt tokenizer data is already downloaded.")
    except LookupError:
        print("Punkt tokenizer data not found. Downloading it is necessary.")
        nltk.download('punkt_tab')

def num_tokens_from_string(string: str) -> int:
    tokens = word_tokenize(string)
    return len(tokens)

def load_news_data():
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )
    news["tokens"] = [
        num_tokens_from_string(f"{row['title']} {row['text']}")
        for i, row in news.iterrows()
    ]
    return news

