# src/data_preprocessing/preprocess.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Ensure required NLTK data packages are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_text(text):
    # Remove URLs
    text = re.sub(r'httpS+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@w+|#w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-zs]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def expand_contractions(text):
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def normalize_text(text):
    text = expand_contractions(text)
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    normalized_text = ' '.join(tokens)
    return normalized_text

def preprocess_dataframe(df):
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['normalized_text'] = df['cleaned_text'].apply(normalize_text)
    
    # Check if 'intent' column exists
    if 'intent' not in df.columns:
        raise ValueError("Intent labels are missing. Please label your data with an 'intent' column.")
    
    return df[['normalized_text', 'intent']]

def main():
    raw_data_path = '../../data/raw/customer_support_on_twitter.csv'
    processed_train_path = '../../data/processed/train.csv'
    processed_test_path = '../../data/processed/test.csv'

    # Load and label data if not already done
    try:
        df = load_data('../../data/processed/labeled_customer_support.csv')
    except FileNotFoundError:
        print("Labeled data not found. Please label the data first.")
        return

    df = preprocess_dataframe(df)

    # Encode intents if necessary
    # Here, assuming 'intent' is already categorical

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['intent'])
    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)

    print("Data preprocessing completed. Train and test datasets are saved.")

if __name__ == "__main__":
    main()
