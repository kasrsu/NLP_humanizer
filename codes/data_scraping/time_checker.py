import csv
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
import os
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import socket

def check_internet(timeout=5, test_url="8.8.8.8"):
    """Test if we have internet connectivity"""
    try:
        socket.create_connection((test_url, 53), timeout=timeout)
        return True
    except OSError:
        return False

def wait_for_internet():
    """Wait until internet connection is available"""
    wait_time = 30  # Start with 30 seconds
    attempt = 1
    
    while not check_internet():
        print(f"\nNo internet connection detected. Waiting {wait_time} seconds...")
        print("Attempt", attempt)
        time.sleep(wait_time)
        attempt += 1
        if attempt % 3 == 0:  # Increase wait time every 3 attempts
            wait_time = min(300, wait_time * 2)  # Cap at 5 minutes
    
    print("Internet connection restored!")
    time.sleep(5)  # Give connection time to stabilize

def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504]  # status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def extract_all_dates_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    time_data = []
    
    print("Extracting time tags...")
    time_tags = soup.find_all('time')
    print(f"Found {len(time_tags)} time tags")
    
    for time_tag in time_tags:
        content = time_tag.get('datetime') or time_tag.get('content') or time_tag.text.strip()
        if content:
            time_data.append(content)
            print(f"Found time content: {content}")
    
    return time_data

def save_progress(url, times, mode='a', success_only=True):
    """Save time data to CSV, optionally only saving successful fetches"""
    if not times and success_only:
        print(f"Skipping failed URL: {url}")
        return
        
    with open('time.csv', mode, newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if mode == 'w':
            writer.writerow(['link', 'time'])
        if times:
            for time_content in times:
                writer.writerow([url, time_content])
            print(f"Saved time data for {url}")

def process_single_url(url, session):
    print(f"\nProcessing URL: {url}")
    max_retries = 1
    current_retry = 0
    
    while current_retry < max_retries:
        # Check internet connection before attempting request
        if not check_internet():
            print("Network disconnected. Waiting for connection...")
            wait_for_internet()
            continue
            
        try:
            print(f"Attempt {current_retry + 1} of {max_retries}")
            response = session.get(url, timeout=30)
            
            if response.status_code == 200:
                print("Successfully fetched webpage")
                time_data = extract_all_dates_from_html(response.text)
                if time_data:
                    save_progress(url, time_data)
                    time.sleep(1)  # Add delay between successful requests
                    return True
                print("No time data found in webpage")
                return False
                
            elif response.status_code == 429:  # Too Many Requests
                wait_time = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                current_retry += 1
                continue
                
            else:
                print(f"Failed with status code: {response.status_code}")
                current_retry += 1
                time.sleep(5)  # Wait before retry
                
        except requests.exceptions.Timeout:
            print("Request timed out. Waiting before retry...")
            time.sleep(10)  # Longer wait for timeouts
            current_retry += 1
            
        except requests.exceptions.ConnectionError:
            print("Connection lost. Waiting for network to return...")
            wait_for_internet()
            current_retry += 1
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            if not check_internet():
                wait_for_internet()
            current_retry += 1
            time.sleep(5)
            
    print(f"Failed to process {url} after {max_retries} attempts")
    return False

def read_urls_from_csv(file_path):
    print(f"Reading URLs from {file_path}")
    urls = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        urls = [row[0] for row in csv_reader]
    print(f"Found {len(urls)} URLs to process")
    return urls

if __name__ == "__main__":
    csv_file_path = 'c_links6.csv'
    session = create_session()
    
    # Check internet before starting
    if not check_internet():
        print("No internet connection detected. Waiting for connection...")
        wait_for_internet()
    
    # Initialize new time.csv with headers only
    save_progress('link', 'time', mode='w', success_only=False)
    
    # Process URLs
    urls = read_urls_from_csv(csv_file_path)
    total_urls = len(urls)
    successful = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\nProcessing URL {i}/{total_urls}")
        if process_single_url(url, session):
            successful += 1
        
        if i % 10 == 0:  # Add longer pause every 10 requests
            print("Taking a short break...")
            time.sleep(5)
            
        # Progress report
        print(f"Progress: {i}/{total_urls} URLs processed ({successful} successful)")

    print(f"\nFinal Summary:")
    print(f"Total URLs processed: {total_urls}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {total_urls - successful}")
    print("Results saved to time.csv")
