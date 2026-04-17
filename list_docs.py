from services.client import supabase

def check_docs():
    try:
        print("Checking documents table...")
        res = supabase.table("documents").select("filename, id").execute()
        for doc in res.data:
            print(f"Doc: {doc['filename']} ({doc['id']})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_docs()
