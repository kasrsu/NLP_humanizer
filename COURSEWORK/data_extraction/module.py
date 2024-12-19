import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
from fake_useragent import UserAgent
import csv  # Import csv module
import random  # Import random module for jitter
from datetime import datetime  # Import datetime module

# List of preferred domains
PREFERRED_DOMAINS = [
    'medium.com',
]

def is_preferred_domain(url):
    return any(domain in url.lower() for domain in PREFERRED_DOMAINS)

def extract_date(result, engine):
    if engine == 'google':
        date_tag = result.find('span', class_='f')
    elif engine == 'bing':
        date_tag = result.find('span', class_='news_dt')
    
    if date_tag:
        try:
            date_str = date_tag.text
            date = datetime.strptime(date_str, '%b %d, %Y')
            return date
        except ValueError:
            pass
    return None

def search(query, num_results=50, engine='google'):
    links = []
    preferred_links = []
    ua = UserAgent()
    page = 0
    retries = 5  # Number of retries for handling 429 error
    
    while len(links) + len(preferred_links) < num_results:
        start = page * 10
        encoded_query = urllib.parse.quote(query)
        
        if engine == 'google':
            url = f"https://www.google.com/search?q={encoded_query}&start={start}"
        elif engine == 'bing':
            url = f"https://www.bing.com/search?q={encoded_query}&first={start + 1}"
        else:
            raise ValueError("Unsupported search engine. Use 'google' or 'bing'.")
        
        headers = {"User-Agent": ua.random}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            if engine == 'google':
                search_results = soup.find_all('div', class_='yuRUbf')
            elif engine == 'bing':
                search_results = soup.find_all('li', class_='b_algo')
            
            for result in search_results:
                link = result.find('a').get('href')
                date = extract_date(result, engine)
                if link and link.startswith('http') and (date is None or date.year < 2020):
                    if is_preferred_domain(link):
                        preferred_links.append(link)
                    else:
                        links.append(link)
                if len(links) + len(preferred_links) >= num_results:
                    break
            
            page += 1
            time.sleep(2)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and retries > 0:
                retries -= 1
                wait_time = (2 ** (5 - retries)) + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f"429 Too Many Requests. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error occurred: {e}")
                break
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    final_results = preferred_links + links
    return final_results[:num_results]

def save_links_to_csv(links, filename='links3.csv'):
    file_exists = False
    try:
        with open(filename, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['urls'])  # Write header if file does not exist
        for link in links:
            writer.writerow([link])  # Write each link

def main():
    search_query = input("Enter your search query: ")
    search_engine = input("Enter search engine (google/bing): ").strip().lower()
    results = search(search_query, engine=search_engine)
    
    print(f"\nFound {len(results)} links:")
    for i, link in enumerate(results, 1):
        print(f"{i}. {link}")
    
    save_links_to_csv(results)  # Save results to CSV

if __name__ == "__main__":
    main()