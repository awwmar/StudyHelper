import fitz

# DSA book
doc = fitz.open("A First Course on Data Structures.pdf")
print(f"DSA Book - Total pages: {doc.page_count}, TOC: {len(doc.get_toc())}")

# Get unique font sizes used across all pages (sample first 20)
font_sizes = {}
for pn in range(min(20, doc.page_count)):
    page = doc.load_page(pn)
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if "lines" in b:
            for line in b["lines"]:
                for span in line["spans"]:
                    sz = round(span["size"], 1)
                    tx = span["text"].strip()
                    if tx:
                        bold = bool(span["flags"] & 16)
                        key = (sz, bold, span["font"])
                        if key not in font_sizes:
                            font_sizes[key] = []
                        if len(font_sizes[key]) < 3:
                            font_sizes[key].append(f"p{pn+1}: {tx[:60]}")

print("\nFont analysis (size, bold, font) -> examples:")
for key in sorted(font_sizes.keys(), key=lambda x: -x[0]):
    sz, bold, font = key
    bstr = "BOLD" if bold else "    "
    print(f"  sz={sz:5.1f} {bstr} font={font:30s}")
    for example in font_sizes[key]:
        print(f"       -> {example}")

# Now check Intro to Computing
print("\n\n--- Intro to Computing ---")
doc2 = fitz.open("Intro to Computing.pdf")
print(f"Total pages: {doc2.page_count}, TOC: {len(doc2.get_toc())}")

font_sizes2 = {}
for pn in range(min(20, doc2.page_count)):
    page = doc2.load_page(pn)
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if "lines" in b:
            for line in b["lines"]:
                for span in line["spans"]:
                    sz = round(span["size"], 1)
                    tx = span["text"].strip()
                    if tx:
                        bold = bool(span["flags"] & 16)
                        key = (sz, bold, span["font"])
                        if key not in font_sizes2:
                            font_sizes2[key] = []
                        if len(font_sizes2[key]) < 3:
                            font_sizes2[key].append(f"p{pn+1}: {tx[:60]}")

print("\nFont analysis (size, bold, font) -> examples:")
for key in sorted(font_sizes2.keys(), key=lambda x: -x[0]):
    sz, bold, font = key
    bstr = "BOLD" if bold else "    "
    print(f"  sz={sz:5.1f} {bstr} font={font:30s}")
    for example in font_sizes2[key]:
        print(f"       -> {example}")

doc.close()
doc2.close()
