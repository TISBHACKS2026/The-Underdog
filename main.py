from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import os
import pdfplumber
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import tempfile

code = input("Enter IGCSE subject code: ")

igcse_subject_codes = {
    "Accounting": "0452",
    "Mathematics": "0606",
    "Afrikaans": "0548",
    "Agriculture": "0600",
    "Art and Design": "0400",
    "Bahasa Indonesia": "0538",
    "Biology": "0610",
    "Business Studies": "0450",
    "Chemistry": "0620",
    "Chinese": "0509",
    "Computer Science": "0478",
    "Economics": "0455",
    "English": "0500",
    "Environmental Management": "0680",
    "French": "0520",
    "Geography": "0460",
    "German": "0525",
    "History": "0470",
    "Latin": "0480",
    "Physics": "0625",
    "Sociology": "0495",
    "Spanish": "0530",
    "Travel and Tourism": "0471",
    "World Literature": "0408",
}

def get_igcse_subject_name(subject_code):
    for name, code in igcse_subject_codes.items():
        if code == subject_code:
            return name
    return "Code not found"
def fetch_past_papers(subject_code):
    subject_name = get_igcse_subject_name(subject_code)
    if subject_name == "Code not found":
        print("Invalid subject code.")
        return
    base_url = "https://pastpapers.papacambridge.com/"
    slug = subject_name.lower().replace(' ', '-')
    url = urljoin(base_url, f"papers/caie/igcse-{slug}-{subject_code}")
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        print("Failed to retrieve data.")
        return
    ugly_soup = BeautifulSoup(response.content, 'html.parser')
    ugly_soup.prettify() #Ugly? I don't think so...
    papers = ugly_soup.find_all('a', class_='kt-widget4__title kt-nav__link-text cursor colorgrey stylefont fonthover')#interesting formatting rule...
    paper_urls = [urljoin(base_url, paper.get('href', '')) for paper in papers]
    if not paper_urls:
        print("No papers found for this subject.")
        return
    download_links = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_fetch_paper_downloads, url, base_url) for url in paper_urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching papers", unit="paper"):
            try:
                download_links.extend(future.result())
            except Exception:
                pass
    if not download_links:
        print("No download links found for this subject.")
        return
    return download_links


def _fetch_paper_downloads(url, base_url):
    uglier_soup = BeautifulSoup(requests.get(url, timeout=15).content, 'html.parser')
    uglier_soup.prettify() #Uglier? Still don't think so...
    return [
        urljoin(base_url, a.get('href', ''))
        for a in uglier_soup.find_all('a', class_='badge badge-info')
    ]

download_links = fetch_past_papers(code)
def download_file(url, filename):
    """
    Downloads a file from a given URL and saves it to the current working directory.

    Args:
        url (str): The URL of the file to download.
        filename (str): The name to use for the saved file.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status() # Check if the download was successful

        # Open the file in binary write mode ('wb') in the current working directory
        with open(filename, 'wb') as f:
            f.write(response.content)
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except IOError as e:
        print(f"Error writing to file: {e}")

def delete_file(filename):
    """
    Deletes a file from the current working directory.

    Args:
        filename (str): The name of the file to delete.
    """
    try:
        os.remove(filename)
    except OSError as e:
        print(f"Error deleting file: {e}")



def _download_to_file(url, filename):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)


def _parse_single_pdf(link, timeout_seconds=120):
    def _timeout_handler(signum, frame):
        raise TimeoutError("PDF parse timed out")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    filename = tmp.name
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        _download_to_file(link, filename)
        text_parts = []
        with pdfplumber.open(filename) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- {os.path.basename(filename)} | Page {page_num} ---\n{text}\n\n")
        return ''.join(text_parts)
    except Exception:
        return ''
    finally:
        signal.alarm(0)
        try:
            os.remove(filename)
        except OSError:
            pass


def parse_pdfs(download_links, max_workers=None, timeout_seconds=120, max_pdfs=100):
    if not download_links:
        return ''
    download_links = download_links[:max_pdfs]
    all_text_parts = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_parse_single_pdf, link, timeout_seconds) for link in download_links]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs", unit="pdf"):
            try:
                all_text_parts.append(future.result())
            except Exception:
                all_text_parts.append('')
    return ''.join(all_text_parts)
if download_links:
    all_extracted_text = parse_pdfs(download_links)
    with open('extracted_igcse_papers.txt', 'w', encoding='utf-8') as f:
        f.write(all_extracted_text)
    print("All text extracted and saved to 'extracted_igcse_papers.txt'.")
else:
    print("No download links to process.")
