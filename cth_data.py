import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException
)
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager  # Ensure this is installed
import re


# Initialize browser options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Optional: Set a custom user-agent to identify your crawler responsibly
chrome_options.add_argument('user-agent=Mozilla/5.0 (compatible; YourCrawlerName/1.0)')

# Disable image loading to speed up crawling
chrome_prefs = {
    "profile.default_content_settings": {"images": 2},
    "profile.managed_default_content_settings": {"images": 2}
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

# Initialize WebDriver using webdriver-manager and Service
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Target URL
base_url = 'https://annonsportal.chalmers.se/CareerServices/en/Ads/Index'

def normalize_link(link):
    """Ensure the link is a complete URL and remove the trailing slash."""
    if not link.startswith('http'):
        link = urljoin('https://annonsportal.chalmers.se', link)
    return link.rstrip('/')

def is_valid_date(date_str):
    """Validate date format YYYY-MM-DD."""
    if not date_str:
        return False
    # Simple regex for YYYY-MM-DD format
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, date_str))

def parse_detail_page(driver, link):
    """
    Navigate to the project's detail page and extract additional information.
    Returns a dictionary with the extracted data.
    """
    detail_data = {
        'Apply By': 'N/A',
        'Apply By Detail': 'N/A',
        'Subject Area': 'N/A',
        'Educational Area': 'N/A',
        'Way to Apply': 'N/A',  # Will be dropped later
        'Contact Info': 'N/A'
    }
    try:
        # Open the link in a new tab
        driver.execute_script("window.open(arguments[0]);", link)
        # Switch to the new tab
        driver.switch_to.window(driver.window_handles[-1])

        # Wait for the detail page to load specific elements
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.mira-html-editor-output'))
        )

        # Extract all form-group divs
        form_groups = driver.find_elements(By.CSS_SELECTOR, 'div.form-group')
        for idx, group in enumerate(form_groups):
            try:
                # Attempt to find the label
                label_element = group.find_element(By.TAG_NAME, 'label')
                label = label_element.text.strip().lower()
            except NoSuchElementException:
                print(f"No label found in form-group {idx + 1}, skipping.")
                continue  # Skip this form-group if label is missing

            try:
                # Attempt to find the value element
                value_element = group.find_element(By.CSS_SELECTOR, 'div.view-text')
                value = value_element.text.strip()
            except NoSuchElementException:
                print(f"No value found for label '{label}' in form-group {idx + 1}.")
                value = 'N/A'

            # Map label to corresponding field
            if 'last apply date' in label or 'sista ansökningsdatum' in label:
                detail_data['Apply By'] = value
            elif 'subject area' in label:
                detail_data['Subject Area'] = value
            elif 'educational area' in label:
                detail_data['Educational Area'] = value
            elif 'information on how to apply' in label:
                detail_data['Way to Apply'] = value  # Will be dropped later
            elif 'contact' in label:
                # Handle Contact Info separately
                contact_name = value
                try:
                    # Look for the next form-group for email
                    email_group = group.find_element(By.XPATH, "./following-sibling::div[contains(@class, 'form-group')]")
                    email_link = email_group.find_element(By.TAG_NAME, 'a').get_attribute('href').replace('mailto:', '').strip()
                    if contact_name and email_link:
                        detail_data['Contact Info'] = f"{contact_name} - {email_link}"
                    elif email_link:
                        detail_data['Contact Info'] = email_link
                    elif contact_name:
                        detail_data['Contact Info'] = contact_name
                    else:
                        detail_data['Contact Info'] = 'N/A'
                except NoSuchElementException:
                    # If email is not found, assign only the contact name
                    detail_data['Contact Info'] = contact_name if contact_name else 'N/A'

        # Extract 'Detailed Description' from the detail page
        try:
            description_element = driver.find_element(By.CSS_SELECTOR, 'div.mira-html-editor-output')
            description_text = description_element.text.strip()
            detail_data['Apply By Detail'] = description_text  # Assign detailed description
        except NoSuchElementException:
            print(f"'Detailed Description' not found on detail page: {link}")

        # Drop 'Way to Apply' as per user instruction
        detail_data.pop('Way to Apply', None)

    except Exception as e:
        print(f"Error parsing detail page {link}: {e}")
    finally:
        # Close the detail tab and switch back to the main tab
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    return detail_data


def extract_apply_by_date(driver, link):
    """
    Navigate to the project's detail page and extract the "Last date of application".
    Returns the date as a string, or 'N/A' if not found.
    """
    apply_by_date = 'N/A'
    try:
        # Open the link in a new tab
        driver.execute_script("window.open(arguments[0]);", link)
        # Switch to the new tab
        driver.switch_to.window(driver.window_handles[-1])

        # Wait for the detail page to load specific elements
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.form-group'))
        )

        # Find all form-group divs
        form_groups = driver.find_elements(By.CSS_SELECTOR, 'div.form-group')

        for group in form_groups:
            try:
                label = group.find_element(By.TAG_NAME, 'label').text.strip().lower()
                value_element = group.find_element(By.CSS_SELECTOR, 'div.view-text')
                value = value_element.text.strip()

                if 'sista ansökningsdatum' in label:
                    apply_by_date = value
                    break  # Exit loop once found
            except NoSuchElementException as e:
                print(f"NoSuchElementException: {e}")
                continue  # Skip if label or value not found

    except Exception as e:
        print(f"Error extracting 'Apply By' date: {e}")
    finally:
        # Close the detail tab and switch back to the main tab
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    return apply_by_date

def parse_page(driver, seen_links, max_retries=3):
    """
    Parse the current page and extract project data.
    """
    projects = []
    try:
        # Wait until all project elements are present
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]'))
        )

        # Repeatedly check until the number of projects stabilizes
        initial_count = len(driver.find_elements(By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]'))
        time.sleep(2)  # Wait for potential dynamic loading
        final_count = len(driver.find_elements(By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]'))

        if initial_count != final_count:
            print("Project elements are still loading. Waiting...")
            WebDriverWait(driver, 10).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]')) == initial_count
            )

        project_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]')
        print(f"Found {len(project_elements)} project elements.")

        for idx in range(len(project_elements)):
            retries = 0
            while retries < max_retries:
                try:
                    # Re-locate the project element to avoid StaleElementReferenceException
                    project_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]')
                    elem = project_elements[idx]

                    # Extract the link and normalize it
                    relative_link = elem.get_attribute('href')
                    link = normalize_link(relative_link) if relative_link else 'N/A'
                    print(f"Extracted project link: {link}")

                    # Check if the link already exists
                    if link in seen_links:
                        print(f"Existing link, skipping: {link}")
                        break  # Exit the retry loop for this element
                    seen_links.add(link)

                    # Extract title
                    try:
                        title = elem.find_element(By.CSS_SELECTOR, 'span[data-field$="Title_Display"]').text.strip()
                    except NoSuchElementException:
                        title = 'N/A'

                    # Extract organization
                    try:
                        organization = elem.find_element(By.CSS_SELECTOR, 'span[data-field$="OrganizationName_Display"]').text.strip()
                    except NoSuchElementException:
                        organization = 'N/A'

                    # Extract location and country
                    try:
                        location = elem.find_element(By.CSS_SELECTOR, 'span[data-field$="Location_Display"]').text.strip()
                    except NoSuchElementException:
                        location = 'N/A'

                    try:
                        country = elem.find_element(By.CSS_SELECTOR, 'span[data-field$="CountryName_Display"]').text.strip()
                    except NoSuchElementException:
                        country = 'N/A'

                    # Extract additional data from the detail page
                    detail_data = parse_detail_page(driver, link)

                    # Validate 'Apply By' date format
                    apply_by_date = detail_data['Apply By']
                    if not is_valid_date(apply_by_date):
                        print(f"Invalid 'Apply By' date format for project {idx + 1}: {apply_by_date}")
                        apply_by_date = 'N/A'

                    # Combine all data
                    project_data = {
                        'Title': title,
                        'Link': link,
                        'Organization': organization,
                        'Location': location,
                        'Country': country,
                        'Apply By': apply_by_date,  # Correctly assign 'Apply By' date
                        'Apply By Detail': detail_data['Apply By Detail'],
                        'Subject Area': detail_data['Subject Area'],
                        'Educational Area': detail_data['Educational Area'],
                        'Contact Info': detail_data['Contact Info']
                    }

                    projects.append(project_data)

                    print(f"Successfully parsed project {idx + 1}")
                    break  # Exit the retry loop on success

                except StaleElementReferenceException:
                    retries += 1
                    print(f"StaleElementReferenceException encountered for project {idx + 1}. Retry {retries}/{max_retries}.")
                    time.sleep(1)  # Wait before retrying
                except Exception as e:
                    print(f"Error parsing project {idx + 1}: {e}")
                    break  # Skip to the next project on other exceptions

            else:
                print(f"Failed to parse project {idx + 1} after {max_retries} retries.")

    except TimeoutException:
        print("Waiting for the project listings to load timed out.")
    except NoSuchElementException:
        print("Project listings not found on the page.")
    except Exception as e:
        print(f"Unexpected error during parse_page: {e}")

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
    except (NoSuchElementException, IndexError, ValueError) as e:
        print(f"Could not determine total pages: {e}. Defaulting to 1.")
        return 1

def crawl_exjobb_selenium():
    """
    Main crawler function, using Selenium to scrape projects from all pages.
    """
    all_projects = []
    seen_links = set()  # Used to record already scraped links

    try:
        driver.get(base_url)
        print(f"Navigated to {base_url}")

        try:
            # Wait for the page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]'))
            )
        except TimeoutException:
            print("Initial page load timed out.")
            return

        # Scrape the homepage
        print(f"Scraping page: {base_url}")
        projects = parse_page(driver, seen_links)
        all_projects.extend(projects)

        # Get the total number of pages
        last_page = get_total_pages(driver)
        print(f"Total pages found: {last_page}")

        # Iterate through all pages starting from page 2
        for page in range(2, last_page + 1):
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    # Find the pagination container
                    pagination = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.pagination'))
                    )
                    # Locate the specific page button
                    page_button = pagination.find_element(By.XPATH, f".//a[text()='{page}']")

                    # Click the pagination button using JavaScript to avoid issues with hidden elements
                    driver.execute_script("arguments[0].click();", page_button)
                    print(f"Clicked page {page} button.")

                    # Wait for the new page's project listings to load
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href^="/CareerServices/en/Ads/Details/"]'))
                    )

                    # Optional: Wait for a short duration to ensure all elements are loaded
                    time.sleep(1)

                    # Scrape the current page's projects
                    print(f"Scraping page {page}.")
                    projects = parse_page(driver, seen_links)
                    all_projects.extend(projects)

                    break  # Exit the retry loop on success

                except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as e:
                    retries += 1
                    print(f"Error navigating to page {page}: {e}. Retry {retries}/{max_retries}.")
                    time.sleep(2)  # Wait before retrying

            else:
                print(f"Failed to navigate to page {page} after {max_retries} retries.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        if all_projects:
            # Ensure the data folder exists
            os.makedirs('data', exist_ok=True)
            # Save the data to a CSV file
            df = pd.DataFrame(all_projects)
            # Define the desired column order
            columns_order = [
                'Title', 'Link', 'Organization', 'Location', 'Country',
                'Apply By', 'Apply By Detail',
                'Subject Area', 'Educational Area',
                'Contact Info'
            ]
            # Reorder columns if all are present
            df = df.reindex(columns=columns_order)
            # Exclude rows where 'Link' contains unwanted footer links (if any)
            unwanted_texts = ['www.chalmers.se', 'Handling of personal data']
            df = df[~df['Link'].str.contains('|'.join(unwanted_texts), na=False)]
            # Save to CSV
            df.to_csv('data/cth_exjobb_projects.csv', index=False, encoding='utf-8-sig')
            print("Data saved to data/cth_exjobb_projects.csv")
        else:
            print("No project information collected. Please check the crawler logic.")

        driver.quit()
        print("WebDriver session ended.")

if __name__ == "__main__":
    crawl_exjobb_selenium()