import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the CSV file
csv_file = "../links3.csv"  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Add a new column to store paragraphs
df['Extracted Paragraphs'] = ''

# Function to extract paragraphs from a URL
def extract_paragraphs(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')  # Find all paragraph tags
        return ' '.join(p.get_text(strip=True) for p in paragraphs)
    except Exception as e:
        return f"Error: {e}"

# Iterate through the URLs and extract paragraphs
for index, row in df.iterrows():
    url = row['urls']  # Adjust the column name if different
    df.at[index, 'Extracted Paragraphs'] = extract_paragraphs(url)

# Save the updated CSV
output_file = "output3.csv"  # Replace with your desired output file name
df.to_csv(output_file, index=False)

print(f"Paragraphs extracted and saved to {output_file}")
