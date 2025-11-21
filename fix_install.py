import subprocess
import sys


def run(cmd):
    """Run a subprocess command and raise on failure."""
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    print(">>> Installing spaCy-UDPipe and the Vietnamese model (vi)...")
    run([sys.executable, "-m", "pip", "install", "spacy-udpipe"])
    # Some distributions do not expose a __main__ for spacy_udpipe; invoke via import.
    run([sys.executable, "-c", "import spacy_udpipe; spacy_udpipe.download('vi')"])
    print(">>> Done. You can now rerun `streamlit run ui/app.py`.")


if __name__ == "__main__":
    main()
