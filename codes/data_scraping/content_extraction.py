import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# Load the CSV file
csv_file = "data_extraction/filtered_data.csv"  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Add a new column to store paragraphs
df['Extracted Paragraphs'] = ''

# Function to extract paragraphs from a URL
def extract_paragraphs(url):
    try:
        print(f"\nProcessing URL: {url}")
        start_time = time.time()
        
        response = requests.get(url)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)
        
        processing_time = time.time() - start_time
        print(f"Found {len(paragraphs)} paragraphs")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return content
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}")
        return f"Error: {e}"
    except Exception as e:
        print(f"General Error: {e}")
        return f"Error: {e}"

# Iterate through the URLs and extract paragraphs
total_urls = len(df)
for index, row in df.iterrows():
    url = row['link']  # Adjust the column name if different
    print(f"\nProcessing {index + 1}/{total_urls}")
    df.at[index, 'Extracted Paragraphs'] = extract_paragraphs(url)

# Save the updated CSV
output_file = "output7.csv"  
df.to_csv(output_file, index=False)

print(f"Paragraphs extracted and saved to {output_file}")
