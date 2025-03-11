import json
import time
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup


# def scrape_page(driver, url):
#     driver.get(url)
#     # Wait until the table loads (adjust timeout as needed)
#     WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.CSS_SELECTOR, "table.rd-table"))
#     )
#     time.sleep(1)  # additional wait for any dynamic content

#     # Locate all table rows for items
#     rows = driver.find_elements(By.CSS_SELECTOR, "tr.rd-filter__search-item")
#     mappings = []
#     for row in rows:
#         try:
#             # Block name: second <td> element with class "rd-table__cell--fluid" containing an <a>
#             name_elem = row.find_element(By.CSS_SELECTOR, "td.rd-table__cell--fluid a")
#             block_name = name_elem.text.strip()

#             # Block id: cell with data-show-type="item_id", find its <span> with the item id text
#             id_elem = row.find_element(By.XPATH, ".//td[@data-show-type='item_id']//span")
#             item_id = id_elem.text.strip()  # e.g., "minecraft:stone"
#             # Remove the "minecraft:" prefix if present
#             block_id = item_id.replace("minecraft:", "")
#             mappings.append((block_name, block_id))
#         except Exception as e:
#             print("Error parsing row:", e)
#     return mappings

def scrape_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()  # Raise error for failed requests

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="rd-table")

    mappings = []
    for row in table.find_all("tr", class_="rd-filter__search-item"):
        name = row.find_all("td")[1].text.strip()
        block_id = row.find("td", {"data-show-type": "item_id"}).text.strip().replace("minecraft:", "")
        mappings.append((name, block_id))

    return mappings

def main():
    base_url = "https://minecraftitemids.com/"
    total_pages = 27

    # Configure Selenium to use headless Chrome
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--disable-gpu")
    # chrome_options.add_argument("--no-sandbox")

    # driver = webdriver.Chrome(options=chrome_options)

    all_mappings = []
    for page in range(1, total_pages + 1):
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}{page}"
        print(f"Scraping page: {url}")
        # page_mappings = scrape_page(driver, url)
        page_mappings = scrape_page(url)
        all_mappings.extend(page_mappings)
        time.sleep(1)  # brief pause between pages

    # driver.quit()

    # Build two dictionaries:
    # block_name2id: mapping block name -> block id
    # block_id2name: mapping block id -> block name
    block_name2id = {}
    block_id2name = {}
    for name, bid in all_mappings:
        block_name2id[name] = bid
        block_id2name[bid] = name

    # Write the mappings to files so that the scraper only needs to be run once.
    with open("block_name2id.json", "w") as f:
        json.dump(block_name2id, f, indent=2)
    with open("block_id2name.json", "w") as f:
        json.dump(block_id2name, f, indent=2)

    print("Scraping complete. Mappings saved to 'block_name2id.json' and 'block_id2name.json'.")

if __name__ == "__main__":
    main()
