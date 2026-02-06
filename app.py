#MADE BY KITTEN-STRIKES-BACK.

"""
TISB HACKATHON 2026 - IGCSE Question Paper Generator

This file contains the main logic for fetching, parsing, and processing IGCSE past papers, as well as the Streamlit UI. It also includes the text_to_pdf function from txtpdf.py for converting cleaned text into a PDF format.
The main steps are:
1. Fetch past paper links from the web based on the selected subject and year range.
2. Download and parse the PDFs to extract text, including handling tables.
3. Clean the extracted text to remove boilerplate and irrelevant content.
4. Extract individual questions from the cleaned text.
5. Classify questions as multiple-choice or text-based.
6. Generate a structured JSON representation of a paper based on the subject's typical structure.
7. Create a sample paper text file from the generated structure.


(Developed by Sidharth Banglani - Grade 8, The International School Bangalore)
"""


import os
import json
import re
import random
import signal
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import urljoin
from txtpdf import text_to_pdf
import requests
import pdfplumber
import streamlit as st
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from wordsegment import load as ws_load, segment as ws_segment
    _WORDSEGMENT_AVAILABLE = True
    ws_load()
except Exception:
    _WORDSEGMENT_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if _GENAI_AVAILABLE and GEMINI_API_KEY:
    print("Gemini API key found. Gemini-based features enabled.")
else:
    print("Gemini API key not found. Gemini-based features will be disabled.")
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
    "Afrikaans": [("Reading and Writing", "text", 30)],
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


def fetch_past_papers(subject_code, range=None):
    #range is start_year-end_year, e.g. "2018-2023"
    range = range.split("-") if range else None
    subject_name = get_igcse_subject_name(subject_code)
    if subject_name == "Code not found":
        return []
    base_url = "https://pastpapers.papacambridge.com/"
    slug = subject_name.lower().replace(" ", "-")
    url = urljoin(base_url, f"papers/caie/igcse-{slug}-{subject_code}")
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        return []
    ugly_soup = BeautifulSoup(response.content, "html.parser")
    ugly_soup.prettify()#Not so ugly now :D
    if range:
        papers = ugly_soup.find_all("a", class_="kt-widget4__title kt-nav__link-text cursor colorgrey stylefont fonthover")
        papers = [p for p in papers if p.text.strip() and any(year in p.text for year in range)]
    else:    
        papers = ugly_soup.find_all("a", class_="kt-widget4__title kt-nav__link-text cursor colorgrey stylefont fonthover")
    paper_urls = [urljoin(base_url, paper.get("href", "")) for paper in papers]
    if not paper_urls:
        return []
    download_links = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_fetch_paper_downloads, url, base_url) for url in paper_urls]
        for future in as_completed(futures):
            try:
                download_links.extend(future.result())
            except Exception:
                pass
    if not download_links:
        return []
    if subject_code in {"0500", "0457"}:
        filtered_links = [link for link in download_links if "qp" in link.lower() or "in" in link.lower()]
    else:
        filtered_links = [link for link in download_links if "qp" in link.lower()]
    return filtered_links


def _fetch_paper_downloads(url, base_url):
    uglier_soup = BeautifulSoup(requests.get(url, timeout=15).content, "html.parser")
    uglier_soup.prettify()#NOT ANYMORE UGLY XD. I love beautifulsoup.
    return [
        urljoin(base_url, a.get("href", ""))
        for a in uglier_soup.find_all("a", class_="badge badge-info")
    ]


def _download_to_file(url, filename):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(filename, "wb") as f:
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
                # Extract tables and insert after the page text
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for t_idx, table in enumerate(tables, start=1):
                    if not table:
                        continue
                    lines = []
                    for row in table:
                        row_vals = [cell if cell is not None else "" for cell in row]
                        lines.append(" | ".join(row_vals))
                    if lines:
                        table_text = "\n".join(lines)
                        text_parts.append(
                            f"--- {os.path.basename(filename)} | Page {page_num} | Table {t_idx} ---\n{table_text}\n\n"
                        )
        return "".join(text_parts)
    except Exception:
        return ""
    finally:
        signal.alarm(0)
        try:
            os.remove(filename)
        except OSError:
            pass


def parse_pdfs(download_links, max_workers=None, timeout_seconds=120, max_pdfs=10):
    if not download_links:
        return ""
    download_links = download_links[:max_pdfs]
    all_text_parts = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_parse_single_pdf, link, timeout_seconds) for link in download_links]
        for future in as_completed(futures):
            try:
                all_text_parts.append(future.result())
            except Exception:
                all_text_parts.append("")
    return "".join(all_text_parts)


def _clean_text_for_questions(text):
    # Drop section divider lines like "--- Section A ---"
    cleaned = re.sub(r"(?m)^--- .*? ---\s*$", "", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    boilerplate_line_patterns = [
        r"^Section\s+[ABC]$",
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
    # Any boilerplate line (case-insensitive) should be filtered out
    boilerplate_line_re = re.compile("|".join(boilerplate_line_patterns), re.IGNORECASE)
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
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
    # Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_questions(text):
    cleaned = _clean_text_for_questions(text)
    # Find question starts like "Question 1 ..." or "1 (a) ..." at line start
    pattern = re.compile(r"(?m)^(?:Question\s*)?\d{1,2}\s+(?=(?:\(|[A-Z]))")
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return []
    questions = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = cleaned[start:end].strip()
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
    # Detect a 4-option MCQ block with A/B/C/D on separate lines
    mcq_pattern = re.compile(r"(?m)^\s*A\s+.+\n^\s*B\s+.+\n^\s*C\s+.+\n^\s*D\s+.+", re.DOTALL)
    if mcq_pattern.search(question_text):
        return True
    # Count option lines like "A ..." to decide if this is MCQ
    option_count = len(re.findall(r"(?m)^\s*[ABCD]\s+", question_text))
    return option_count >= 4


def _segment_joined_words(text, enabled=True):
    if not enabled or not _WORDSEGMENT_AVAILABLE:
        return text

    def segment_run(match):
        words = ws_segment(match.group(0))
        return " ".join(words)

    fixed_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue
        # If a line is a single long run of letters, split it into words
        if " " not in line and re.search(r"[A-Za-z]{12,}", line):
            line = re.sub(r"[A-Za-z]{8,}", segment_run, line)
        else:
            line = re.sub(r"[A-Za-z]{12,}", segment_run, line)
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def generate_paper_json(mcq_questions, text_questions, structure_list):
    paper = []
    for section_name, question_type, num_questions in structure_list:
        if question_type == "multiple":
            selected_questions = random.sample(mcq_questions, min(num_questions, len(mcq_questions)))
        elif question_type == "text":
            selected_questions = random.sample(text_questions, min(num_questions, len(text_questions)))
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        paper.append({"section": section_name, "questions": selected_questions})
    return paper


def _clean_sample_text(text):
    # Remove ©UCLES lines and common OCR noise, normalize spacing, and tidy tables.
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.startswith("©UCLES"):
            continue
        # Drop literal escape artifacts
        line = line.replace("\\n", "\n")
        # Drop OCR escape artifacts like "\1" or "\123 \456"
        line = re.sub(r"\\+1\b", "", line)
        line = re.sub(r"\b\\1\b", "", line)
        line = re.sub(r"\\\d+\s+\\\d+", "", line)
        # Drop obvious CID artifact lines
        if re.fullmatch(r"\(cid:\d+\)+", stripped):
            continue
        # Fix common OCR spacing issues around currency and alphanumerics
        line = re.sub(r"\$(\d)", r"$ \1", line)
        line = re.sub(r"([A-Za-z])\$(\d)", r"\1 $\2", line)
        line = re.sub(r"(\d)([A-Za-z])", r"\1 \2", line)
        line = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", line)
        # Normalize table separators
        if "|" in line:
            # Drop large empty table grids
            # Drop large empty table grids
            if re.fullmatch(r"(\s*\|\s*){6,}\s*", line):
                continue
            # Normalize table separators and spacing
            line = re.sub(r"\s*\|\s*", " | ", line)
            line = re.sub(r"\s{2,}", " ", line)
        cleaned_lines.append(line)
    # Collapse excessive blank lines
    text = "\n".join(cleaned_lines)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_sample_paper(paper):
    sample_lines = []
    seen = set()
    for section in paper:
        sample_lines.append(f"--- {section['section']} ---\n")
        for question in section["questions"]:
            # Normalize whitespace to dedupe similar questions
            normalized = re.sub(r"\s+", " ", question).strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            sample_lines.append(question + "\n\n")
    raw = "\n".join(sample_lines)
    text = _clean_sample_text(raw)
    text = gemini_fix_text(text, api_key=GEMINI_API_KEY, model="gemini-2.5-flash")
    return text


def save_outputs(all_extracted_text, questions, output_dir="."):
    first_1000_questions = questions[:1000]
    mcq_questions = [q for q in first_1000_questions if is_multiple_choice(q)]
    text_questions = [q for q in first_1000_questions if not is_multiple_choice(q)]

    with open(os.path.join(output_dir, "extracted_igcse_papers.txt"), "w", encoding="utf-8") as f:
        f.write(all_extracted_text)
    with open(os.path.join(output_dir, "extracted_igcse_questions_1000.json"), "w", encoding="utf-8") as f:
        json.dump(first_1000_questions, f, ensure_ascii=True, indent=2)
    with open(os.path.join(output_dir, "multiple_choice.json"), "w", encoding="utf-8") as f:
        json.dump(mcq_questions, f, ensure_ascii=True, indent=2)
    with open(os.path.join(output_dir, "text_questions.json"), "w", encoding="utf-8") as f:
        json.dump(text_questions, f, ensure_ascii=True, indent=2)

    return first_1000_questions, mcq_questions, text_questions


def _pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")



def spell_correct_text(text, progress_cb=None):
    if not _TEXTBLOB_AVAILABLE:
        return text
    lines = text.splitlines()
    total = max(len(lines), 1)
    corrected = []
    for i, line in enumerate(lines, start=1):
        # Skip spell-correction on lines with no letters
        if re.search(r"[A-Za-z]", line):
            try:
                line = str(TextBlob(line).correct())
            except Exception:
                pass
        corrected.append(line)
        if progress_cb and i % 50 == 0:
            progress_cb(i / total)
    if progress_cb:
        progress_cb(1.0)
    return "\n".join(corrected)


def gemini_feedback(question, answer, api_key, model="gemini-pro"):
    if not _GENAI_AVAILABLE:
        return "google-generativeai is not installed."
    if not api_key:
        return "GEMINI_API_KEY is not set."
    prompt = (
        "You are an IGCSE examiner. Provide concise feedback (strengths + improvements) "
        "and, if appropriate, a short suggested answer outline.\n\n"
        f"Question:\n{question}\n\nStudent answer:\n{answer}"
    )
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model_name=model)
        response = model_obj.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"

def gemini_fix_text(text, api_key, model="gemini-2.5-flash"):
    if not _GENAI_AVAILABLE:
        return text
    if not api_key:
        return text
    prompt = (
        "You are a helpful assistant for cleaning up OCR-extracted text from IGCSE papers. "
        "Fix common OCR errors, tidy tables, remove irrelevant boilerplate, remove question numbers, and(if needed/if images/data required are missing) generate images/data that fit the context. Preserve question integrity. Do not write anything except the cleaned text.\n\n"
        "If there is a missing image/diagram, explain what the image/diagram should contain in the text. Example: for a question on Hooke's law you can say: '(The graph(x:spring length, y:force) shows a line that slants upwards.)'\n\n"
        f"Original text:\n{text}\n\nCleaned text:"
    )
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model_name=model)
        response = model_obj.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"

def gemini_list_models(api_key):
    if not _GENAI_AVAILABLE:
        return []
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
    except Exception:
        return []

# Streamlit part
st.set_page_config(page_title="PaperHarvester", layout="wide")

st.title(" 📝 PaperHarvester - IGCSE Question Paper Generator")
st.markdown("*A tool to generate IGCSE question papers from past papers.*")
col1, col2 = st.columns([2, 1])
with col1:
    subject = st.selectbox("Subject", list(igcse_subject_codes.keys()))
    max_pdfs = st.number_input("Max PDFs to parse", min_value=1, max_value=50, value=10)
    timeout_seconds = st.number_input("PDF parse timeout (seconds)", min_value=30, max_value=600, value=120)
    max_questions = st.number_input("Max questions to keep", min_value=100, max_value=5000, value=1000)
    use_wordsegment = st.checkbox("Fix joined words (wordsegment)", value=True, disabled=not _WORDSEGMENT_AVAILABLE)
    use_textblob = st.checkbox("Spell-correct text (TextBlob)", value=False, disabled=not _TEXTBLOB_AVAILABLE)
    range_input = st.text_input("Year range for papers (e.g. 2018-2023)", value="")
with col2:
    st.markdown("**Outputs**")
    st.markdown("- `extracted_igcse_papers.txt`")
    st.markdown("- `extracted_igcse_questions_1000.json`")
    st.markdown("- `multiple_choice.json`")
    st.markdown("- `text_questions.json`")
    st.markdown("- `generated_igcse_paper.json`")
    st.markdown("- `sample_paper.txt`")

st.markdown("---")

col_run1, col_run2 = st.columns(2)
run_web = col_run1.button("Generate from web")
run_existing = col_run2.button("Use existing extracted_igcse_papers.txt")

if run_web or run_existing:
    code = igcse_subject_codes[subject]
    st.info(f"Subject: {subject} ({code})")

    if run_web:
        with st.spinner("Fetching paper links..."):
            links = fetch_past_papers(code)
        if not links:
            st.error("No papers found or failed to fetch. Check network access.")
            st.stop()
        with st.spinner("Parsing PDFs..."):
            raw_text = parse_pdfs(links, timeout_seconds=timeout_seconds, max_pdfs=max_pdfs)
    else:
        if not os.path.exists("extracted_igcse_papers.txt"):
            st.error("extracted_igcse_papers.txt not found in the working directory.")
            st.stop()
        with open("extracted_igcse_papers.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()

    with st.spinner("Cleaning and extracting questions..."):
        raw_text = raw_text.replace(".", "")
        raw_text = _segment_joined_words(raw_text, enabled=use_wordsegment)
        if use_textblob:
            progress = st.progress(0, text="Spell-correcting (TextBlob)...")
            raw_text = spell_correct_text(raw_text, progress_cb=progress.progress)
        questions = extract_questions(raw_text)

    first_1000, mcq_questions, text_questions = save_outputs(raw_text, questions)

    st.success(f"Questions extracted: {len(questions)}")
    st.write(f"MCQ: {len(mcq_questions)} | Text: {len(text_questions)}")

    paper_structure = structure.get(subject, [])
    if paper_structure:
        paper = generate_paper_json(mcq_questions, text_questions, paper_structure)
        with open("generated_igcse_paper.json", "w", encoding="utf-8") as f:
            json.dump(paper, f, ensure_ascii=True, indent=2)
        with st.spinner("Generating sample paper..."):
            sample = generate_sample_paper(paper)
        with open("sample_paper.txt", "w", encoding="utf-8") as f:
            f.write(sample)

    st.markdown("---")
    st.subheader("Downloads")
    for filename in [
        "extracted_igcse_papers.txt",
        "extracted_igcse_questions_1000.json",
        "multiple_choice.json",
        "text_questions.json",
        "generated_igcse_paper.json",
        "sample_paper.txt",
    ]:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                st.download_button(
                    label=f"Download {filename}",
                    data=f,
                    file_name=filename,
                )

    st.markdown("---")
    st.subheader("Text to PDF")
    if os.path.exists("sample_paper.txt"):
        if st.button("Convert sample_paper.txt to PDF"):
            progress = st.progress(0)
            with open("sample_paper.txt", "r", encoding="utf-8") as f:
                text = f.read()
            with st.spinner("Generating PDF..."):
                pdf_path = text_to_pdf(text, "sample_paper.pdf")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download sample_paper.pdf",
                    data=f,
                    file_name="sample_paper.pdf",
                )

st.markdown("---")
st.subheader("Student Practice Mode")

def _format_question_for_display(question):
    if is_multiple_choice(question):
        # Format options onto separate lines for display
        text = re.sub(r"\s([ABCD])\s", r"\n\1 ", question)
        text = re.sub(r"(\?)\s+(?=[ABCD]\s)", r"\1\n", text)
        return text.strip()
    return question.strip()


def _parse_mcq(question):
    text = _format_question_for_display(question)
    lines = text.splitlines()
    stem_lines = []
    options = []
    for line in lines:
        # Parse option lines like "A option text"
        m = re.match(r"^([ABCD])\s+(.*)$", line.strip())
        if m:
            options.append((m.group(1), m.group(2).strip()))
        else:
            stem_lines.append(line)
    stem = "\n".join(stem_lines).strip()
    return stem, options[:4]


def _render_question_with_images(question_text):
    parts = re.split(r"\[\[IMG:(.+?)\]\]", question_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.text(_format_question_for_display(part))
        else:
            img_path = part.strip()
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True)


practice_col1, practice_col2 = st.columns([2, 1])
with practice_col1:
    practice_source = st.selectbox(
        "Question source",
        ["multiple_choice.json", "text_questions.json", "extracted_igcse_questions_1000.json"],
    )
    practice_count = st.number_input("Number of questions", min_value=1, max_value=100, value=10)
    shuffle_questions = st.checkbox("Shuffle questions", value=True)
with practice_col2:
    st.markdown("**Session**")
    start_practice = st.button("Start new practice session")

if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = []
if "practice_answers" not in st.session_state:
    st.session_state.practice_answers = {}
if "practice_index" not in st.session_state:
    st.session_state.practice_index = 0

if start_practice:
    if not os.path.exists(practice_source):
        st.error(f"{practice_source} not found in the working directory.")
    else:
        with open(practice_source, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        if shuffle_questions:
            random.shuffle(all_questions)
        st.session_state.practice_questions = all_questions[: int(practice_count)]
        st.session_state.practice_answers = {}
        st.session_state.practice_index = 0

questions = st.session_state.practice_questions
if questions:
    idx = st.session_state.practice_index
    total = len(questions)
    st.markdown(f"**Question {idx + 1} of {total}**")
    current_question = gemini_fix_text(questions[idx], api_key=GEMINI_API_KEY, model="gemini-2.5-flash")
    if is_multiple_choice(current_question):
        stem, options = _parse_mcq(current_question)
        _render_question_with_images(stem)
        option_labels = [f"{k} {v}" for k, v in options]
        if option_labels:
            selected_default = st.session_state.practice_answers.get(idx)
            selected = st.radio(
                "Choose an answer",
                options=option_labels,
                index=option_labels.index(selected_default) if selected_default in option_labels else 0,
                key=f"mcq_{idx}",
            )
            st.session_state.practice_answers[idx] = selected
    else:
        _render_question_with_images(current_question)
        answer_key = f"answer_{idx}"
        default_answer = st.session_state.practice_answers.get(idx, "")
        user_answer = st.text_area("Your answer", value=default_answer, height=150, key=answer_key)
        st.session_state.practice_answers[idx] = user_answer

    nav_col1, nav_col2, nav_col3 = st.columns(3)
    if nav_col1.button("Previous", disabled=idx == 0):
        st.session_state.practice_index = max(0, idx - 1)
    if nav_col2.button("Next", disabled=idx >= total - 1):
        st.session_state.practice_index = min(total - 1, idx + 1)
    if nav_col3.button("Save answers to file"):
        out = {
            "source": practice_source,
            "answers": [
                {"question": questions[i], "answer": st.session_state.practice_answers.get(i, "")}
                for i in range(total)
            ],
        }
        with open("practice_answers.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=True, indent=2)
        st.success("Saved to practice_answers.json")

    st.markdown("---")
    st.subheader("AI Feedback (Gemini)")
    if not _GENAI_AVAILABLE:
        st.info("Install google-generativeai to enable feedback.")
    else:
        key_input = st.text_input(
            "Gemini API key",
            type="password",
            value="",
            help="Optional; falls back to GEMINI_API_KEY env var",
        )
        model_name = st.text_input("Model name", value="gemini-1.5-flash")
        api_key = key_input or GEMINI_API_KEY
        if st.button("List available models"):
            models = gemini_list_models(api_key)
            if models:
                st.write(models)
            else:
                st.warning("No models returned (check API key or permissions).")
        if st.button("Get feedback for this answer"):
            q_text = gemini_fix_text(current_question, api_key=api_key)
            a_text = st.session_state.practice_answers.get(idx, "")
            if not a_text:
                st.warning("Please provide an answer first.")
            else:
                with st.spinner("Generating feedback..."):
                    feedback = gemini_feedback(q_text, a_text, api_key=api_key, model=model_name)
                st.write(feedback)
#THE END.
#Just like Gemini, please consider providing feedback on how I can improve this project! 
