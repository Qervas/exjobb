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
from webdriver_manager.chrome import ChromeDriverManager
import logging

# Initialize logging
logging.basicConfig(
    filename='kth_crawler.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize browser options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Optional: Set a user-agent to mimic a real browser
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36")

# Initialize WebDriver using webdriver-manager and Service
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Target URL
base_url = 'https://kth-exjobb.powerappsportals.com/en-US/'

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
        # Wait for the project rows to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'tr[data-entity="aca_job"]'))
        )
        logging.info("Project rows found, starting to parse projects.")

        # Find all project rows
        project_rows = driver.find_elements(By.CSS_SELECTOR, 'tr[data-entity="aca_job"]')
        logging.info(f"Found {len(project_rows)} project rows on the current page.")

        for idx, row in enumerate(project_rows, start=1):
            try:
                # Extract all <td> elements in the row
                cols = row.find_elements(By.TAG_NAME, 'td')
                if len(cols) < 6:
                    logging.warning(f"Row {idx} does not contain enough columns, skipping.")
                    continue

                # Extract Title and Link from the first <td>
                title_td = cols[0]
                title_link = title_td.find_element(By.TAG_NAME, 'a')
                title = title_link.text.strip()
                relative_link = title_link.get_attribute('href')
                link = normalize_link(relative_link) if relative_link else 'N/A'
                logging.info(f"Extracted project link: {link}")

                # Check for duplicates
                if link in seen_links:
                    logging.info(f"Duplicate project found, skipping: {link}")
                    continue
                seen_links.add(link)

                # Extract Organization from the second <td>
                organization = cols[1].text.strip() if cols[1].text else 'N/A'

                # Extract Location from the third <td>
                location = cols[2].text.strip() if cols[2].text else 'N/A'

                # Extract Assignment Type from the fourth <td>
                assignment_type = cols[3].text.strip() if cols[3].text else 'N/A'

                # Extract Publish Date from the fifth <td>, which contains a <time> tag
                publish_date_td = cols[4]
                publish_date_time = publish_date_td.find_element(By.TAG_NAME, 'time')
                publish_date_iso = publish_date_time.get_attribute('datetime')
                publish_date_display = publish_date_time.text.strip()

                # Extract Deadline for Application from the sixth <td>, which contains a <time> tag
                deadline_td = cols[5]
                deadline_time = deadline_td.find_element(By.TAG_NAME, 'time')
                deadline_iso = deadline_time.get_attribute('datetime')
                deadline_display = deadline_time.text.strip()

                # Append the project data to the list
                projects.append({
                    'Title': title,
                    'Link': link,
                    'Organization': organization,
                    'Location': location,
                    'Assignment Type': assignment_type,
                    'Publish Date (ISO)': publish_date_iso,
                    'Publish Date': publish_date_display,
                    'Deadline for Application (ISO)': deadline_iso,
                    'Deadline for Application': deadline_display
                })

            except NoSuchElementException as e:
                logging.error(f"Error parsing row {idx}: {e}")
                continue

    except TimeoutException:
        logging.error("Timeout waiting for project rows to load.")
    except NoSuchElementException:
        logging.error("Project rows not found on the page.")

    return projects

def get_total_pages(driver):
    """
    Get the total number of pages.
    """
    try:
        pagination = driver.find_element(By.CSS_SELECTOR, 'ul.pagination')
        pages = pagination.find_elements(By.TAG_NAME, 'li')
        # Extract page numbers, ignoring 'Previous' and 'Next'
        page_numbers = [li.text for li in pages if li.text.isdigit()]
        last_page = int(page_numbers[-1]) if page_numbers else 1
        logging.info(f"Total pages found: {last_page}")
        return last_page
    except (NoSuchElementException, IndexError, ValueError) as e:
        logging.error(f"Error determining total pages: {e}")
        return 1

def crawl_kth_exjobb():
    """
    Main crawler function, using Selenium to scrape KTH Exjobb projects from all pages.
    """
    all_projects = []
    seen_links = set()  # Used to record already scraped links

    logging.info(f"Accessing {base_url}")
    driver.get(base_url)

    try:
        # Wait for the page to load by checking the presence of project rows
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'tr[data-entity="aca_job"]'))
        )
        logging.info("Page loaded successfully.")
    except TimeoutException:
        logging.error("Page load timed out.")
        driver.quit()
        return

    # Scrape the first page
    logging.info("Scraping the first page.")
    projects = parse_page(driver, seen_links)
    all_projects.extend(projects)

    # Get the total number of pages
    last_page = get_total_pages(driver)

    # Scrape remaining pages
    for page in range(2, last_page + 1):
        try:
            logging.info(f"Scraping page {page}.")

            # Find the pagination button for the desired page
            pagination = driver.find_element(By.CSS_SELECTOR, 'ul.pagination')
            # Use the data-page attribute to find the correct page button
            page_button = pagination.find_element(By.XPATH, f".//a[@data-page='{page}']")

            # Click the pagination button using JavaScript to avoid issues with headless mode
            driver.execute_script("arguments[0].click();", page_button)
            logging.info(f"Clicked page {page} button.")

            # Wait for the new page to load by checking the presence of project rows
            WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'tr[data-entity="aca_job"]'))
            )
            logging.info(f"Page {page} loaded successfully.")

            # Optional: Wait for additional time to ensure all elements load
            time.sleep(2)

            # Scrape the current page
            projects = parse_page(driver, seen_links)
            all_projects.extend(projects)

        except NoSuchElementException:
            logging.warning(f"Pagination button for page {page} not found, skipping.")
            continue
        except TimeoutException:
            logging.warning(f"Timeout while loading page {page}, skipping.")
            continue

    if all_projects:
        # Ensure the data folder exists
        os.makedirs('data', exist_ok=True)

        # Save the data to a CSV file
        df = pd.DataFrame(all_projects)
        csv_path = os.path.join('data', 'kth_exjobb_projects.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Data saved to {csv_path}")
    else:
        logging.warning("No project information collected. Please check the crawler logic.")

    driver.quit()
    logging.info("Crawler finished successfully.")

if __name__ == "__main__":
    crawl_kth_exjobb()