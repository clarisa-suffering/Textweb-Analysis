from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# --- SET YOUR REVIEW LIMIT HERE ---
REVIEW_LIMIT = 5500

# The URL of the product page
url = "https://global.oliveyoung.com/product/detail?prdtNo=GA220615265&dataSource=top_orders"

# Setup and open Chrome using Selenium
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)
driver.maximize_window() # Maximize window to ensure elements are clickable

# --- 1. SCRAPE GLOBAL REVIEWS ---
print("--- Starting Global Reviews ---")
print("Waiting for initial page load...")
time.sleep(5) # Wait for the initial page elements to be ready

# Loop to click the 'More' button until it's gone or the limit is reached
while True:
    try:
        more_button = driver.find_element(By.CLASS_NAME, 'review-list-more-btn')
        driver.execute_script("arguments[0].click();", more_button)
        print("Clicked 'More' button... waiting for new reviews to load.")
        time.sleep(2)

        # --- ADDED CHECK FOR REVIEW LIMIT ---
        # Count how many reviews are currently loaded on the page
        loaded_reviews_count = len(driver.find_elements(By.CLASS_NAME, 'product-review-unit'))
        print(f"Loaded {loaded_reviews_count} reviews so far...")
        
        # If the count reaches the limit, stop clicking 'More'
        if loaded_reviews_count >= REVIEW_LIMIT:
            print(f"Review limit of {REVIEW_LIMIT} reached. Stopping.")
            break

    except NoSuchElementException:
        print("'More' button not found. All global reviews are loaded.")
        break

# Get the page source after all global reviews are loaded
html_global = driver.page_source
driver.quit() # Close the browser now that we have the global reviews HTML

# --- 2. PARSE AND SAVE GLOBAL DATA ---
print("\n--- Parsing all collected global reviews ---")
soup = BeautifulSoup(html_global, 'html.parser')
all_review_elements = soup.find_all('div', class_='product-review-unit')

# Use a set to store unique review IDs to prevent duplicates
scraped_ids = set()
scraped_reviews = []

for review in all_review_elements:
    writer_element = review.find('span', class_='review-write-info-writer')
    date_element = review.find('span', class_='review-write-info-date')
    content_element = review.find('div', class_='review-unit-cont-comment')
    
    # Create a unique ID from essential elements to handle duplicates
    if not writer_element or not date_element or not content_element:
        continue
        
    username = writer_element.text.strip()
    date = date_element.text.strip()
    review_text = content_element.text.strip()
    unique_id = f"{username}-{date}-{len(review_text)}"

    if unique_id not in scraped_ids:
        scraped_ids.add(unique_id)

        rating_element = review.find('div', class_='review-star-rating')
        rating = len(rating_element.find_all('div', class_='icon-star coral-50 left filled')) if rating_element else 'N/A'
        
        option_element = review.find('div', class_='review-unit-option')
        product_option = option_element.text.strip() if option_element else 'N/A'

        scraped_reviews.append({
            'username': username,
            'date': date,
            'rating': rating,
            'product_option': product_option,
            'review_text': review_text
        })

# Convert the list to a pandas DataFrame and save to Excel
if scraped_reviews:
    df = pd.DataFrame(scraped_reviews)
    df.to_excel('olive_young_global_reviews.xlsx', index=False)
    print(f"\nSuccess! Scraped a total of {len(scraped_reviews)} unique global reviews and saved to olive_young_global_reviews.xlsx")
else:
    print("\nNo reviews were scraped in the end.")