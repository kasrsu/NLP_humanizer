import ollama
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import re
import os

nltk.download('punkt_tab')

def split_into_sentences(text):
    """Split text into clean sentences."""
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    # Split into sentences
    sentences = sent_tokenize(text)
    # Clean each sentence
    return [s.strip() for s in sentences if s.strip()]

def human_to_ai_text(human_text, doc_id, output_csv, debug=False):
    try:
        if debug:
            print(f"\nProcessing document ID: {doc_id}")
            
        paragraphs = [p.strip() for p in human_text.split('\n\n') if p.strip()]
        
        # Create or check if file exists
        if not os.path.exists(output_csv):
            pd.DataFrame(columns=['doc_id', 'paragraph_id', 'sentence', 'ai_sentence']).to_csv(output_csv, index=False)
        
        for para_idx, paragraph in enumerate(paragraphs, 1):
            sentences = split_into_sentences(paragraph)
            
            if debug:
                print(f"\nParagraph {para_idx}: {len(sentences)} sentences")
            
            for sentence in sentences:
                if debug:
                    print(f"\nProcessing sentence: {sentence}")
                
                prompt = f"""Rewrite this sentence in AI style without changing the original structure . Do not try to provide defferent formats. just the rewrite sentence only no any other things:
                {sentence}"""
                
                try:
                    response = ollama.generate(
                        model='mistral:7b',
                        prompt=prompt
                    )
                    rewritten = response['response'].strip()
                    
                    # Create single record
                    record = pd.DataFrame([{
                        'doc_id': doc_id,
                        'paragraph_id': para_idx,
                        'sentence': sentence,
                        'ai_sentence': rewritten
                    }])
                    
                    # Append to CSV immediately
                    record.to_csv(output_csv, mode='a', header=False, index=False)
                    
                    if debug:
                        print(f"Saved: {sentence[:50]}... → {rewritten[:50]}...")
                        
                except Exception as e:
                    if debug:
                        print(f"Error processing sentence: {str(e)}")
                    
        return True
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {str(e)}")
        return False

def process_csv(input_csv='hello.csv', output_csv='sentences_pairs.csv', 
                row_limit=None, debug=False):
    try:
        df = pd.read_csv(input_csv)
        
        if debug:
            print(f"Processing {len(df)} documents")
        
        if row_limit:
            df = df.head(row_limit)
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if debug:
                print(f"\nProcessing document {idx+1}/{len(df)}")
            
            success = human_to_ai_text(
                row['Extracted Paragraphs'],
                idx + 1,  # doc_id
                output_csv,
                debug
            )
            
            if debug and success:
                print(f"Successfully processed document {idx+1}")
                
    except Exception as e:
        print(f"Error in process_csv: {str(e)}")

if __name__ == "__main__":
    process_csv(debug=True)