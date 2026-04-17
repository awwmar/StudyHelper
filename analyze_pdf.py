import fitz

doc = fitz.open('uploads/kebo109.pdf')
print(f"Pages: {doc.page_count}, TOC entries: {len(doc.get_toc())}")

with open('pdf_analysis.txt', 'w', encoding='utf-8') as f:
    for pn in range(doc.page_count):
        page = doc.load_page(pn)
        text = page.get_text().strip()
        f.write(f"\n=== Page {pn+1} (text length: {len(text)}) ===\n")
        f.write(text[:500] + "\n")
        
        # Font analysis
        blocks = page.get_text('dict')['blocks']
        for b in blocks:
            if 'lines' in b:
                for line in b['lines']:
                    for span in line['spans']:
                        sz = round(span['size'], 1)
                        tx = span['text'].strip()
                        if tx and len(tx) > 1:
                            bold = bool(span['flags'] & 16)
                            flags = span['flags']
                            f.write(f"  sz={sz:5.1f} flags={flags:3d} bold={'Y' if bold else 'N'} font={span['font']:30s} | {tx[:80]}\n")

doc.close()
print("Done - wrote pdf_analysis.txt")
