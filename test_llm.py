import os
import sys
from dotenv import load_dotenv

# Add current directory to sys.path
sys.path.append(os.getcwd())
load_dotenv()

try:
    from services.llm import llm_service
    print("Testing LLM Service...")
    res = llm_service.llm.invoke("Hello")
    print(f"Result: {res}")
except Exception as e:
    import traceback
    traceback.print_exc()
