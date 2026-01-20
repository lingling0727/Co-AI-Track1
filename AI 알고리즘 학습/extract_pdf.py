from pypdf import PdfReader

pdf_path = "2025_KMOOC_알고리즘_중간고사.pdf"
out_path = "exam_text.txt"

reader = PdfReader(pdf_path)
with open(out_path, "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages, start=1):
        f.write(f"=== PAGE {i} ===\n")
        text = page.extract_text()
        if text:
            f.write(text)
        f.write("\n\n")

print(f"Wrote text to {out_path}")
