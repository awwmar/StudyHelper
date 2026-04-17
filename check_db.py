from services.client import supabase

def check_doc():
    try:
        print("Checking for biomolecules.pdf in DB...")
        res = supabase.table("documents").select("*").eq("filename", "biomolecules.pdf").execute()
        if res.data:
            print("Document found:", res.data)
        else:
            print("Document NOT found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_doc()
