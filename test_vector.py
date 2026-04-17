import os
import sys
from dotenv import load_dotenv

# Add current directory to sys.path
sys.path.append(os.getcwd())
load_dotenv()

try:
    from services.vector import vector_service
    print(f"Testing Vector Service... Index Size: {vector_service.index.ntotal}")
except Exception as e:
    import traceback
    traceback.print_exc()
