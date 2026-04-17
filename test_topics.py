"""Quick test of fixed _extract_topics_from_fonts on NCERT PDF."""
import fitz, re
from collections import Counter

def _extract_topics_from_fonts(doc):
    all_sizes = []
    for pn in range(doc.page_count):
        page = doc.load_page(pn)
        text = page.get_text().strip()
        if len(text) < 50:
            continue
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        tx = span["text"].strip()
                        if tx and len(tx) > 5:
                            all_sizes.append(round(span["size"], 1))
    if not all_sizes:
        return []
    size_counts = Counter(all_sizes)
    body_size = size_counts.most_common(1)[0][0]
    heading_min_size = body_size * 1.1
    
    heading_lines = []
    for pn in range(doc.page_count):
        page = doc.load_page(pn)
        text = page.get_text().strip()
        if len(text) < 50:
            continue
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    line_y = line["bbox"][1]
                    spans = line["spans"]
                    has_large_bold = False
                    for span in spans:
                        sz = round(span["size"], 1)
                        font = span["font"]
                        is_bold = ("Demi" in font or "Bold" in font or bool(span["flags"] & 16))
                        if sz >= heading_min_size and is_bold:
                            tx = span["text"].strip()
                            if tx and not (tx.isdigit() and len(tx) <= 3):
                                has_large_bold = True
                                break
                    if not has_large_bold:
                        continue
                    parts = []
                    max_sz = 0
                    for span in spans:
                        sz = round(span["size"], 1)
                        tx = span["text"].strip()
                        font = span["font"]
                        if not tx:
                            continue
                        is_bold = ("Demi" in font or "Bold" in font or bool(span["flags"] & 16))
                        if is_bold:
                            parts.append(tx)
                            max_sz = max(max_sz, sz)
                    if parts:
                        full_text = " ".join(parts)
                        heading_lines.append((pn, line_y, max_sz, full_text))
    
    if not heading_lines:
        return []
    
    merged = []
    current = heading_lines[0]
    for i in range(1, len(heading_lines)):
        prev_page, prev_y, prev_sz, prev_text = current
        curr_page, curr_y, curr_sz, curr_text = heading_lines[i]
        if curr_page == prev_page and abs(curr_y - prev_y) < 15:
            current = (prev_page, prev_y, max(prev_sz, curr_sz), prev_text + " " + curr_text)
        else:
            merged.append(current)
            current = heading_lines[i]
    merged.append(current)
    
    topics_raw = []
    text_counts = Counter(m[3].upper().strip() for m in merged)
    for pn, ly, sz, text in merged:
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 3:
            continue
        if text_counts.get(text.upper(), 0) > 2:
            continue
        upper = text.upper()
        if upper in ("BIOLOGY", "IOLOGY", "B IOLOGY", "IOMOLECULES", "B IOMOLECULES"):
            continue
        if upper.startswith("TABLE ") or upper.startswith("FIGURE "):
            continue
        if upper.startswith("ABLE "):
            continue
        if text.replace(".", "").strip().isdigit():
            continue
        topics_raw.append({"title": text, "page": pn + 1, "size": sz})
    
    if not topics_raw:
        return []
    
    heading_sizes = sorted(set(t["size"] for t in topics_raw), reverse=True)
    max_heading_size = heading_sizes[0]
    size_threshold = max_heading_size - 2.0
    filtered = [t for t in topics_raw if t["size"] >= size_threshold]
    if not filtered:
        filtered = topics_raw
    
    results = []
    for i in range(len(filtered)):
        title = filtered[i]["title"]
        start_page = filtered[i]["page"]
        end_page = filtered[i+1]["page"] - 1 if i < len(filtered) - 1 else doc.page_count
        if end_page < start_page:
            end_page = start_page
        results.append({"title": title, "start": start_page, "end": end_page})
    return results

doc = fitz.open("uploads/kebo109.pdf")
results = _extract_topics_from_fonts(doc)
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write(f"PDF: kebo109.pdf ({doc.page_count} pages)\n\n")
    f.write(f"=== FINAL TOPICS ({len(results)}) ===\n")
    for t in results:
        f.write(f"  [{t['start']}-{t['end']}] {t['title']}\n")
doc.close()
print("Done. See test_results.txt")
