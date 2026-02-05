from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import os
import pdfplumber
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import tempfile
import json
import re
import random
from wordsegment import load, segment

load()



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

structure = {
    "Accounting": [("Section A", "multiple", 35), ("Section B", "text", 5)],
    "Mathematics": [("Paper 1", "text", 20), ("Paper 2", "text", 20)],
    "Afrikaans": [("Reading and Writing", "text", 30), ("Listening", "text", 20)],
    "Agriculture": [("Section 1", "text", 35), ("Section 2", "text", 5)],
    "Art and Design": [("Practical", "text", 100)],
    "Bahasa Indonesia": [("Reading and Writing", "text", 4), ("Listening", "text", 2)],
    "Biology": [("Paper 2 - MCQ", "multiple", 40), ("Paper 4 - Text", "text", 7), ("Paper 6 - Text", "text", 3)], 
    "Business Studies": [("Paper 1", "text", 4), ("Paper 2", "text", 4)],
    "Chemistry": [("Paper 2 - MCQ", "multiple", 40), ("Paper 4 - Text", "text", 7), ("Paper 6 - Text", "text", 3)],
    "Chinese": [("Reading and Writing", "text", 4), ("Listening", "multiple", 35)],
    "Computer Science": [("Computer Systems", "text", 12), ("Algorithms, Programming and Logic", "text", 9)],
    "Economics": [("Paper 1", "multiple", 30), ("Paper 2", "text", 13)],
    "English": [("Paper 1 - Reading", "text", 20), ("Paper 2 - Writing", "text", 20)],
    "Environmental Management": [("Paper 1", "text", 40)],
    "French": [("Reading and Writing", "text", 4), ("Listening", "multiple", 35)],
    "Geography": [("Paper 1", "text", 40)],
    "German": [("Reading and Writing", "text", 4), ("Listening", "multiple", 35)],
    "History": [("Paper 1", "text", 40)],
    "Latin": [("Paper 1", "text", 40)],
    "Physics": [("Paper 2 - MCQ", "multiple", 40), ("Paper 4 - Text", "text", 7), ("Paper 6 - Text", "text", 3)],
    "Sociology": [("Paper 1", "text", 40)],
    "Spanish": [("Reading and Writing", "text", 4), ("Listening", "multiple", 35)],
    "Travel and Tourism": [("Paper 1", "text", 40)],
    "World Literature": [("Paper 1", "text", 40)],
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
    if subject_code == "0500" or subject_code == "0457":
        filtered_links = [link for link in download_links if "qp" in link.lower() or "in" in link.lower()]
    else: 
        filtered_links = [link for link in download_links if "qp" in link.lower()]
    return filtered_links
    


def _fetch_paper_downloads(url, base_url):
    uglier_soup = BeautifulSoup(requests.get(url, timeout=15).content, 'html.parser')
    uglier_soup.prettify() #Uglier? Still don't think so...
    return [
        urljoin(base_url, a.get('href', ''))
        for a in uglier_soup.find_all('a', class_='badge badge-info')
    ]


def download_file(url, filename):
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


def parse_pdfs(download_links, max_workers=None, timeout_seconds=120, max_pdfs=10):
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

def _clean_text_for_questions(text):
    # Remove page header markers like: --- filename | Page N ---
    cleaned = re.sub(r"(?m)^--- .*? ---\s*$", "", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    boilerplate_line_patterns = [
        r"^Section\s+[AB]$",
        r"^Answer\s+Question\s+\d+$",
        r"^Answer\s+any\s+.*questions.*$",
        r"^Answer\s+all\s+parts\s+of\s+Question.*$",
        r"^Read\s+the\s+source\s+material.*$",
        r"^Source\s+material:.*$",
        r"^INSTRUCTIONS$",
        r"^INFORMATION$",
        r"^READ\s+THESE\s+INSTRUCTIONS\s+FIRST$",
        r"^Additional\s+Materials:.*$",
        r"^You\s+must\s+answer.*$",
        r"^You\s+will\s+need:.*$",
        r"^You\s+may\s+use.*$",
        r"^The\s+total\s+mark.*$",
        r"^The\s+number\s+of\s+marks.*$",
        r"^Write\s+your.*$",
        r"^Do\s+not\s+use.*$",
        r"^Choose\s+the\s+one.*$",
        r"^Each\s+correct\s+answer.*$",
        r"^Any\s+rough\s+working.*$",
        r"^Soft\s+clean\s+eraser$",
        r"^Soft\s+pencil.*$",
        r"^\[Turn\s+over.*$",
    ]
    boilerplate_line_re = re.compile("|".join(boilerplate_line_patterns), re.IGNORECASE)
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Drop standalone page numbers and boilerplate
        if re.fullmatch(r"\d+", stripped):
            continue
        if stripped.upper() == "BLANK PAGE":
            continue
        if stripped.startswith("© UCLES"):
            continue
        if boilerplate_line_re.match(stripped):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    # Collapse extra blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def extract_questions(text):
    cleaned = _clean_text_for_questions(text)
    # Match question starts at line beginnings:
    # "1 (a)...", "2 The ...", "Question 3 ..."
    pattern = re.compile(r"(?m)^(?:Question\s*)?\d{1,2}\s+(?=(?:\(|[A-Z]))")
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return []
    questions = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = cleaned[start:end].strip()
        # Trim trailing boilerplate that sometimes appears after the last question
        for stop in [
            "Permission to reproduce",
            "Copyright Acknowledgements",
            "This document consists",
            "UNIVERSITY OF CAMBRIDGE INTERNATIONAL EXAMINATIONS",
            "International General Certificate of Secondary Education",
        ]:
            idx = chunk.find(stop)
            if idx != -1:
                chunk = chunk[:idx].strip()
                break
        if chunk:
            questions.append(chunk)
    return questions

def is_multiple_choice(question_text):
    # Detect typical MCQ option layout with A/B/C/D on separate lines.
    mcq_pattern = re.compile(r"(?m)^\s*A\s+.+\n^\s*B\s+.+\n^\s*C\s+.+\n^\s*D\s+.+", re.DOTALL)
    if mcq_pattern.search(question_text):
        return True
    # Fallback: count lines starting with A/B/C/D options
    option_count = len(re.findall(r"(?m)^\s*[ABCD]\s+", question_text))
    return option_count >= 4

def _segment_joined_words(text):
    # Use wordsegment only on long, space-less or jammed word runs to preserve layout.
    def segment_run(match):
        words = segment(match.group(0))
        return " ".join(words)

    fixed_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue
        # If the line has no spaces and contains long letter runs, segment them.
        if " " not in line and re.search(r"[A-Za-z]{12,}", line):
            line = re.sub(r"[A-Za-z]{8,}", segment_run, line)
        else:
            # Also fix any long jammed letter runs inside a line.
            line = re.sub(r"[A-Za-z]{12,}", segment_run, line)
        fixed_lines.append(line)
    return "\n".join(fixed_lines)

def generate_paper_json(mcq_file, text_file, structure , output_file='paper.json'):
    #strucure is a list of tuples like: [("Section A", "multiple", 2), ("Section B", "text", 3)] where the 2nd element is the type of question and the last element is the number of questions in that section 
    #This function will randomly select the specified number of questions for each section using random.choice and save them in a JSON file with the given output filename.
    with open(mcq_file, 'r', encoding='utf-8') as f:
        mcq_questions = json.load(f)
    with open(text_file, 'r', encoding='utf-8') as f:
        text_questions = json.load(f)

    paper = []
    for section_name, question_type, num_questions in structure:
        if question_type == "multiple":
            selected_questions = random.sample(mcq_questions, min(num_questions, len(mcq_questions)))
        elif question_type == "text":
            selected_questions = random.sample(text_questions, min(num_questions, len(text_questions)))
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        paper.append({
            "section": section_name,
            "questions": selected_questions
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(paper, f, ensure_ascii=True, indent=2)

    print(f"Paper generated and saved to '{output_file}'.")


def generate_sample_paper(json_paper):
    with open(json_paper, 'r', encoding='utf-8') as f:
        paper = json.load(f)

    def _clean_sample_text(text):
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped.startswith("©UCLES"):
                continue
            if re.fullmatch(r"\(cid:\d+\)+", stripped):
                continue
            line = re.sub(r"\$(\d)", r"$ \1", line)
            line = re.sub(r"([A-Za-z])\$(\d)", r"\1 $\2", line)
            line = re.sub(r"(\d)([A-Za-z])", r"\1 \2", line)
            line = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", line)
            if "|" in line:
                line = re.sub(r"\s*\|\s*", " | ", line)
                line = re.sub(r"\s{2,}", " ", line)
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    sample_lines = []
    seen = set()
    for section in paper:
        sample_lines.append(f"--- {section['section']} ---\n")
        for question in section['questions']:
            normalized = re.sub(r"\s+", " ", question).strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            sample_lines.append(question + "\n\n")

    sample_text = _clean_sample_text("\n".join(sample_lines))
    with open('sample_paper.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)

    print("Sample paper generated and saved to 'sample_paper.txt'.")
code = input("Enter IGCSE subject code: ")



# Main execution flow
download_links = fetch_past_papers(code)
name = get_igcse_subject_name(code)
print(f"Subject: {name} ({code})")
if download_links:
    all_extracted_text = parse_pdfs(download_links)
    all_extracted_text = all_extracted_text.replace(".", "")
    all_extracted_text = _segment_joined_words(all_extracted_text)

    with open('extracted_igcse_papers.txt', 'w', encoding='utf-8') as f:
        f.write(all_extracted_text)
    questions = extract_questions(all_extracted_text)
    first_1000_questions = questions[:1000]
    with open('extracted_igcse_questions_1000.json', 'w', encoding='utf-8') as f:
        json.dump(first_1000_questions, f, ensure_ascii=True, indent=2)
    mcq_questions = [q for q in first_1000_questions if is_multiple_choice(q)]
    text_questions = [q for q in first_1000_questions if not is_multiple_choice(q)]
    with open('multiple_choice.json', 'w', encoding='utf-8') as f:
        json.dump(mcq_questions, f, ensure_ascii=True, indent=2)
    with open('text_questions.json', 'w', encoding='utf-8') as f:
        json.dump(text_questions, f, ensure_ascii=True, indent=2)
    print("All text extracted and saved to 'extracted_igcse_papers.txt'.")
    print("First 1000 questions saved to 'extracted_igcse_questions_1000.json'.")
    print("Multiple choice questions saved to 'multiple_choice.json'.")
    print("Text questions saved to 'text_questions.json'.")
    generate_paper_json("multiple_choice.json", "text_questions.json", structure[name], 'generated_igcse_paper.json')
    generate_sample_paper('generated_igcse_paper.json')
else:
    print("No download links to process.")
