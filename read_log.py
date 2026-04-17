import sys
import os

filename = sys.argv[1]
print(f"--- Reading {filename} ---")
content = ""
try:
    with open(filename, 'r', encoding='utf-16') as f:
        content = f.read()
except Exception as e:
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e2:
        print(f"Error reading file: {e2}")
        sys.exit(1)

start = content.find('Checking Vector Service')
if start != -1:
    print("--- VECTOR SERVICE SECTION ---")
    relevant = content[start:start+2000] # Get next 2000 chars
    print(relevant)
else:
    print("Vector Service section not found.")
