from playwright.sync_api import sync_playwright
import re
import pandas as pd

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

# URL of the page to scrape
url = "https://tetw.org/Greats/"

# Scrape links
print(f"Scraping links from {url}")
links = scrape_medium_links_with_playwright(url)
print("Scraped Medium Links published before 2019:")
for link in links:
    print(link)

# Save links to a CSV file
df = pd.DataFrame(links, columns=["Links"])
df.to_csv("links4.csv", index=False)
print("Saved links to 'links4.csv'")
