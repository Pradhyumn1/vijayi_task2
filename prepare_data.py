import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def prepare_and_index_data():
    """
    Downloads, cleans, and indexes the dataset from Hugging Face.
    """
  
    print("Step 1: Loading Data from Hugging Face...")
    try:
        ds = load_dataset("Abirate/english_quotes", split='train')
        df = ds.to_pandas()
    except Exception as e:
        print(f"Failed to load dataset from Hugging Face. Error: {e}")
        print("Please check your network connection.")
        return

    print("\nStep 2: Preprocessing and Cleaning Data...")
    df.dropna(subset=['quote'], inplace=True)
    df['author'] = df['author'].fillna('Unknown').str.rstrip(',').str.strip()
    df['quote'] = df['quote'].str.strip(' “"”').str.strip()
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    
    noise_tag = 'attributed-no-source'
    df['tags'] = df['tags'].apply(lambda tag_list: [tag for tag in tag_list if tag != noise_tag])
    
    df['tags_str'] = df['tags'].apply(lambda x: ', '.join(x) if x else 'none')
    df['combined'] = "Quote: \"" + df['quote'] + "\" by " + df['author'] + ". Tags: " + df['tags_str']

    print(f"Data prepared. Total quotes: {len(df)}")

  
    print("\nStep 3: Encoding Quotes with 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)
    print(f"Embeddings created with shape: {embeddings.shape}")


    print("\nStep 4: Building and Saving FAISS Index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))

    if not os.path.exists('rag_assets'):
        os.makedirs('rag_assets')

    faiss.write_index(index, 'rag_assets/quotes.index')
    df.to_csv('rag_assets/quotes_data.csv', index=False)

    print("\n✅ Data preparation and indexing complete.")

if __name__ == '__main__':
    prepare_and_index_data()