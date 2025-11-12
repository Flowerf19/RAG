import traceback
import time

from llm.ollama_client import OllamaClient


def main():
    try:
        print("Creating Ollama client...")
        client = OllamaClient()
        print("Client created:", client)

        print("Checking availability...")
        avail = client.is_available()
        print("is_available:", avail)

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Ollama' and include current timestamp."}
        ]

        print("Sending generate() request (may take some time)...")
        start = time.time()
        try:
            out = client.generate(prompt, max_tokens=80)
            took = time.time() - start
            print("--- RESPONSE START ---")
            print(out)
            print("--- RESPONSE END ---")
            print(f"Generation took {took:.2f}s, length={len(out)}")
        except Exception as e:
            print("generate() raised:", repr(e))
            traceback.print_exc()

    except Exception as e:
        print("Setup error:", repr(e))
        traceback.print_exc()


if __name__ == '__main__':
    main()
