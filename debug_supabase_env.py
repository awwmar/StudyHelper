import os
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

print("--- Supabase Env Debug ---")
if supabase_url:
    print(f"URL found: {supabase_url[:15]}...")
else:
    print("URL not found.")

if supabase_key:
    prefix = supabase_key[:15]
    print(f"KEY found: {prefix}...")
    if "sb_" in prefix:
        print("[Note] This key appears to be a new-style API key (sb_...).")
        print("       The Python Supabase client (v2.3.0) might expect a JWT (starts with 'eyJ...').")
        print("       Please check the 'Legacy anon, service_role API keys' tab in Supabase > Settings > API Keys.")
    elif "eyJ" in prefix:
        print("[Note] This key appears to be a valid JWT.")
else:
    print("KEY not found.")
