import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager  # Ensure this is installed

# Initialize browser options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Initialize WebDriver using webdriver-manager and Service
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Target URL
base_url = 'https://exjobb.liu.se/en-US/'

def normalize_link(link):
    """Ensure the link is a complete URL and remove the trailing slash"""
    if not link.startswith('http'):
        link = urljoin(base_url, link)
    return link.rstrip('/')

def parse_page(driver, seen_links):
    """
    Parse the current page and extract project data.
    """
    projects = []
    try:
        # Wait for the table to load and ensure there are data rows in <tbody>
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.table.table-striped.table-fluid tbody tr'))
        )

        # Find the table
        table = driver.find_element(By.CSS_SELECTOR, 'table.table.table-striped.table-fluid')
        print("Table found, starting to parse projects.")

        # Find <tbody>
        tbody = table.find_element(By.TAG_NAME, 'tbody')

        # Find all data rows
        rows = tbody.find_elements(By.TAG_NAME, 'tr')
        print(f"Found {len(rows)} rows.")

        for idx, row in enumerate(rows, start=1):
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) != 4:
                print(f"Number of columns in row {idx} does not match the expected (4 columns), skipping this row.")
                continue

            # Extract title and link
            try:
                headline_tag = cols[0].find_element(By.CLASS_NAME, 'details-link')
                title = headline_tag.text.strip()
                relative_link = headline_tag.get_attribute('href')
                link = normalize_link(relative_link) if relative_link else 'N/A'
                print(f"Extracted project link: {link}")
            except NoSuchElementException:
                title = 'N/A'
                link = 'N/A'
                print(f"Project link does not exist, skipping.")
                continue  # Skip the project if the link does not exist

            # Check if the link already exists
            if link in seen_links:
                print(f"Existing link, skipping: {link}")
                continue
            seen_links.add(link)

            # Extract organization
            organization = cols[1].text.strip() if cols[1] else 'N/A'

            # Extract main research field
            main_field = cols[2].text.strip() if cols[2] else 'N/A'

            # Extract application deadline
            try:
                deadline_tag = cols[3].find_element(By.TAG_NAME, 'time')
                deadline_iso = deadline_tag.get_attribute('datetime')
                deadline_display = deadline_tag.text.strip()
            except NoSuchElementException:
                deadline_iso = 'N/A'
                deadline_display = 'N/A'

            projects.append({
                'Title': title,
                'Link': link,
                'Organization': organization,
                'Main Research Field': main_field,
                'Application Deadline (ISO)': deadline_iso,
                'Application Deadline (Display)': deadline_display
            })

    except TimeoutException:
        print("Waiting for the table to load timed out.")
    except NoSuchElementException:
        print("Target table or other elements not found.")

    return projects

def get_total_pages(driver):
    """
    Get the total number of pages.
    """
    try:
        pagination = driver.find_element(By.CSS_SELECTOR, 'ul.pagination')
        pages = pagination.find_elements(By.TAG_NAME, 'li')
        # Filter possible "Previous" and "Next" buttons, keeping only numeric page numbers
        page_numbers = [li.text for li in pages if li.text.isdigit()]
        last_page = int(page_numbers[-1]) if page_numbers else 1
        print(f"Total pages parsed: {last_page}")
        return last_page
    except (NoSuchElementException, IndexError, ValueError):
        return 1

def crawl_exjobb_selenium():
    """
    Main crawler function, using Selenium to scrape projects from all pages.
    """
    all_projects = []
    seen_links = set()  # Used to record already scraped links

    driver.get(base_url)

    try:
        # Wait for the page to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.table.table-striped.table-fluid'))
        )
    except TimeoutException:
        print("Page load timed out.")
        driver.quit()
        return

    # Scrape the homepage
    print(f"Scraping page: {base_url}")
    projects = parse_page(driver, seen_links)
    all_projects.extend(projects)

    # Get the total number of pages
    last_page = get_total_pages(driver)
    print(f"Total pages found: {last_page}")

    for page in range(2, last_page + 1):
        # Construct the logic to click pagination, as the URL does not change
        try:
            # Find the pagination button
            pagination = driver.find_element(By.CSS_SELECTOR, 'ul.pagination')
            page_button = pagination.find_element(By.XPATH, f".//a[@data-page='{page}']")
            # Click the pagination button
            driver.execute_script("arguments[0].click();", page_button)
            print(f"Clicked page {page} button.")

            # Wait for the page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'table.table.table-striped.table-fluid tbody tr'))
            )

            # Ensure the page is fully loaded, wait for 1 second
            time.sleep(1)

            # Scrape the current page's projects
            print(f"Scraping page {page}.")
            projects = parse_page(driver, seen_links)
            all_projects.extend(projects)

        except NoSuchElementException:
            print(f"Page {page} pagination button not found, skipping.")
            continue
        except TimeoutException:
            print(f"Page {page} load timed out, skipping.")
            continue

    if all_projects:
        # Ensure the data folder exists
        os.makedirs('data', exist_ok=True)
        # Save the data to a CSV file
        df = pd.DataFrame(all_projects)
        df.to_csv('data/liu_exjobb_projects.csv', index=False, encoding='utf-8-sig')
        print("Data saved to data/liu_exjobb_projects.csv")
    else:
        print("No project information collected. Please check the crawler logic.")

    driver.quit()

if __name__ == "__main__":
    crawl_exjobb_selenium()
