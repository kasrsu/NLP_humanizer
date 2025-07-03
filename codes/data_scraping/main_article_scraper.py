from playwright.sync_api import sync_playwright
import re
import pandas as pd
import csv
import json
import os

def save_checkpoint(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def scrape_medium_links_with_playwright(url):
    with sync_playwright() as p:
        print("Launching browser")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Medium page
        print(f"Navigating to {url}")
        page.goto(url)

        # Wait for the page to load (optional)
        print("Waiting for the page to load")
        page.wait_for_timeout(5000)

        # Extract all links
        print("Extracting all links")
        all_links = page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
        print(f"Found {len(all_links)} links")

        browser.close()
        print("Browser closed")
        return all_links

# Read URLs from the CSV file, skipping first 10 rows
input_file = 'links4.csv'
output_file = 'links5.csv'
urls_checkpoint = 'urls_checkpoint.json'
links_checkpoint = 'links_checkpoint.json'

# Try to load URLs from checkpoint
urls = load_checkpoint(urls_checkpoint)
if urls is None:
    print(f"Reading URLs from {input_file}")
    with open(input_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for _ in range(10):
            next(reader, None)
        urls = [row[0] for row in reader if len(row) > 0]
    save_checkpoint(urls, urls_checkpoint)

# Try to load existing scraped links from checkpoint
all_scraped_links = load_checkpoint(links_checkpoint) or []
start_index = len(all_scraped_links)

# Scrape links from each URL
for i, url in enumerate(urls[start_index:], start_index):
    try:
        print(f"Scraping links from {url} ({i+1}/{len(urls)})")
        links = scrape_medium_links_with_playwright(url)
        if len(links) > 20:
            all_scraped_links.extend(links[20:])
        print(f"Added {len(links) - 20 if len(links) > 20 else 0} links after skipping first 20")
        
        # Save checkpoint after each URL is processed
        save_checkpoint(all_scraped_links, links_checkpoint)
        
        # Also save to CSV periodically
        if i % 5 == 0:  # Save every 5 URLs
            temp_df = pd.DataFrame(all_scraped_links, columns=["Links"])
            temp_df.to_csv(f"{output_file}.temp", index=False)
            
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        continue

# Save final results
df = pd.DataFrame(all_scraped_links, columns=["Links"])
df.to_csv(output_file, index=False)
print(f"Saved filtered links to '{output_file}'")

# Clean up checkpoint files
os.remove(urls_checkpoint)
os.remove(links_checkpoint)
if os.path.exists(f"{output_file}.temp"):
    os.remove(f"{output_file}.temp")
