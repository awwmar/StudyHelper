"""Check Groq independently of Flask, Supabase, and local embedding models."""
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, APIStatusError, APIConnectionError


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("GROQ_API_KEY"):
        print("FAIL: Set GROQ_API_KEY in .env or the server environment.")
        return 1
    models = dict.fromkeys([
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b").strip(),
    ])
    client = Groq(timeout=30, max_retries=0)
    try:
        available = {model.id for model in client.models.list().data}
        failed = False
        for model in models:
            if model not in available:
                print(f"FAIL: {model!r} is not in the available model list.")
                failed = True
                continue
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with only OK."}],
                max_completion_tokens=256,
            )
            if not response.choices or not response.choices[0].message.content:
                print(f"FAIL: {model} returned no visible answer.")
                failed = True
            else:
                print(f"PASS: {model} returned an answer.")
        return int(failed)
    except APIStatusError as exc:
        print(f"FAIL: Groq HTTP {exc.status_code}. Check key, model permissions, and quota.")
        return 1
    except APIConnectionError:
        print("FAIL: Cannot connect to Groq. Check network/proxy settings.")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
