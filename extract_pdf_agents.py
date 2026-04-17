import fitz
import re

doc = fitz.open('c:/Users/Amar/Desktop/SOP/research paper.pdf')
text = ""
watermark = "Authorized licensed use limited to: BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE. Downloaded on March 03,2026 at 08:27:37 UTC from IEEE Xplore. Restrictions apply."

for i, page in enumerate(doc):
    # Skip the cover page which usually just contains the citation and a big watermark for IEEE papers
    if i == 0:
        continue 
    page_text = page.get_text()
    if watermark in page_text:
        page_text = page_text.replace(watermark, "")
    text += page_text + "\n"

with open('c:/Users/Amar/Desktop/SOP/paper_utf8.txt', 'w', encoding='utf-8') as f:
    f.write("--- SUMMARY ---\n")
    f.write(text[:2000] + "\n\n")
    f.write("--- AGENT MENTIONS ---\n")
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    agent_sentences = [s for s in sentences if 'agent' in s.lower()]
    for j, s in enumerate(agent_sentences):
        f.write(f"[{j}]: {s.strip()}\n")
