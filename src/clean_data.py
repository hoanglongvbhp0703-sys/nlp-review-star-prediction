import pandas as pd
import re
import html

TEXT_COL = "Text"

HTML_TAG_RE = re.compile(r'<[^>]*>')

def clean_text(text: str) -> str:
    text = str(text)
    text = html.unescape(text)          
    text = re.sub(HTML_TAG_RE, '', text)  
    text = text.lower()                


    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataframe(csv_path: str, text_col: str = TEXT_COL) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found. Available: {list(df.columns)}"
        )
    df[text_col] = df[text_col].apply(clean_text)
    return df
