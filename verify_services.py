import os
import sys
from dotenv import load_dotenv

# Add current directory to sys.path
sys.path.append(os.getcwd())

load_dotenv()

def verify():
    print("=== Study Assistant System Verification ===")
    
    # 0. Check PyMuPDF
    print("\n[0] Checking PyMuPDF (fitz)...")
    try:
        import fitz
        print(f"[OK] PyMuPDF imported. Version: {fitz.VersionBind}")
    except Exception as e:
        print(f"[FAIL] PyMuPDF (fitz) import failed: {e}")

    
    # 1. Check Env Vars
    print("\n[1] Checking Environment Variables...")
    required = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[FAIL] Missing: {', '.join(missing)}")
        return
    print("[OK] Environment variables present.")

    # 1.5 Check Supabase Storage
    print("\n[1.5] Checking Supabase Storage...")
    try:
        from services.client import supabase
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        print(f"[OK] Buckets found: {bucket_names}")
        if "pdfs" not in bucket_names:
             print("[WARN] 'pdfs' bucket missing! This will cause upload failures.")
        else:
             print("[OK] 'pdfs' bucket exists.")
    except Exception as e:
        print(f"[FAIL] Storage check failed: {e}")


    # 2. Check Supabase DB
    print("\n[2] Checking Supabase DB Connection...")
    try:
        from services.client import supabase
        # Try to select from topics (even if empty)
        supabase.table("topics").select("count", count="exact").execute()
        print("[OK] Supabase DB connected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Supabase DB connection failed: {e}")
        print("Tip: Did you run the 'supabase_schema.sql' in Supabase SQL Editor?")

    # 3. Check Vector Service (FAISS)
    print("\n[3] Checking Vector Service (FAISS)...")
    try:
        from services.vector import vector_service
        print(f"[OK] FAISS Index loaded. Size: {vector_service.index.ntotal}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Vector Service failed: {e}")

    # 4. Check LLM Service
    print("\n[4] Checking LLM Service (Groq)...")
    try:
        from services.llm import llm_service
        # Simple invoke
        resp = llm_service.llm.invoke("Say 'Hello'").content
        print(f"[OK] LLM Connected. Response: {resp}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] LLM Service failed: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify()
