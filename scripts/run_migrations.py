import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    print("Error: DATABASE_URL is not set in .env")
    print("Please add your Supabase Postgres connection string (starts with postgresql://...) to .env")
    print("You can find this in your Supabase Dashboard under Settings -> Database -> Connection string.")
    print("Example: DATABASE_URL=postgresql://postgres.lsjeocoezflsckjjhpbd:[PASSWORD]@aws-0-ap-south-1.pooler.supabase.com:6543/postgres")
    exit(1)

SCHEMA_FILE = "supabase_schema.sql"

if not os.path.exists(SCHEMA_FILE):
    print(f"Error: {SCHEMA_FILE} not found!")
    exit(1)

# Read SQL
with open(SCHEMA_FILE, "r") as f:
    sql_script = f.read()

print("Connecting to database...")
try:
    # Connect
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cursor = conn.cursor()

    print(f"Executing {SCHEMA_FILE}...")
    cursor.execute(sql_script)
    
    print("Migration executed successfully!")

except Exception as e:
    print(f"Migration Failed: {e}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
