import os


def _pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_to_pdf(text, output_path, page_width=595, page_height=842, margin=50, font_size=11, leading=14):
    lines = text.splitlines()
    pages = []
    y = page_height - margin
    content_lines = []

    def flush_page():
        if content_lines:
            pages.append("\n".join(content_lines))

    for line in lines:
        if y < margin + leading:
            flush_page()
            content_lines = []
            y = page_height - margin
        escaped = _pdf_escape(line)
        content_lines.append(f"1 0 0 1 {margin} {y} Tm ({escaped}) Tj")
        y -= leading

    flush_page()

    objects = []
    offsets = []

    def add_object(obj_str):
        offsets.append(sum(len(o) for o in objects))
        objects.append(obj_str)

    add_object("1 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    page_ids = []
    for i, page_content in enumerate(pages, start=1):
        content_stream = f"BT\n/F1 {font_size} Tf\n{page_content}\nET"
        content_id = 1 + len(page_ids) * 2 + 1
        add_object(f"{content_id} 0 obj\n<< /Length {len(content_stream)} >>\nstream\n{content_stream}\nendstream\nendobj\n")

        page_id = content_id + 1
        page_ids.append(page_id)
        add_object(
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent {page_id + 1} 0 R "
            f"/Resources << /Font << /F1 1 0 R >> >> "
            f"/MediaBox [0 0 {page_width} {page_height}] "
            f"/Contents {content_id} 0 R >>\n"
            f"endobj\n"
        )

    pages_id = page_ids[-1] + 1 if page_ids else 2
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    add_object(f"{pages_id} 0 obj\n<< /Type /Pages /Count {len(page_ids)} /Kids [ {kids} ] >>\nendobj\n")

    catalog_id = pages_id + 1
    add_object(f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n")

    header = "%PDF-1.4\n"
    body = "".join(objects)
    xref_offset = len(header) + len(body)
    xref_entries = ["0000000000 65535 f \n"]
    for off in offsets:
        xref_entries.append(f"{off + len(header):010d} 00000 n \n")
    xref = "xref\n0 {0}\n{1}".format(len(xref_entries), "".join(xref_entries))
    trailer = f"trailer\n<< /Size {len(xref_entries)} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"

    with open(output_path, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(body.encode("utf-8"))
        f.write(xref.encode("utf-8"))
        f.write(trailer.encode("utf-8"))

    return output_path


if __name__ == "__main__":
    input_path = "sample_paper.txt"
    output_path = "sample_paper.pdf"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    text_to_pdf(text, output_path)
    print(f"Wrote {output_path}")
