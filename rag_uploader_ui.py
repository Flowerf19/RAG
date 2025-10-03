"""Streamlit UI for uploading documents to a RAG pipeline."""

from pathlib import Path
from typing import Tuple

import streamlit as st

# Configuration constants
APP_TITLE = "RAG Document Uploader"
UPLOAD_DIR = Path(r"C:\Users\ENGUYEHWC\Prototype\Version_4\RAG\data\pdf")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 50
CHECKSUM_REQUIRED = True  # Backend flag indicating checksum validation will happen post-upload


def ensure_upload_directory(directory: Path) -> None:
    """Ensure the target directory exists before writing files."""
    directory.mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """Return the lower-case file extension."""
    return Path(filename).suffix.lower()


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate extension and size before persisting the file."""
    extension = get_file_extension(uploaded_file.name)
    if extension not in ALLOWED_EXTENSIONS:
        return False, "4xx: Chi ho tro tep PDF hoac DOCX."

    size_in_mb = uploaded_file.size / (1024 * 1024)
    if size_in_mb > MAX_FILE_SIZE_MB:
        return False, "4xx: Tep vuot qua gioi han 50MB."

    return True, ""


def persist_uploaded_file(uploaded_file, destination: Path) -> Path:
    """Write the uploaded file to the destination on disk."""
    destination_path = destination / uploaded_file.name
    with destination_path.open("wb") as buffer:
        buffer.write(uploaded_file.getbuffer())
    return destination_path


def trigger_backend_ingestion(file_path: Path) -> int:
    """Placeholder for calling the RAG backend API after a successful upload."""
    # Example: response = requests.post("http://backend/upload", files={"file": open(file_path, "rb")})
    # return response.status_code
    return 200


def render_header() -> None:
    """Render the page header and introductory text."""
    st.title(APP_TITLE)
    st.write(
        "Tai len tai lieu cua ban de chuan bi cho quy trinh RAG. "
        "He thong chap nhan tep PDF hoac DOCX voi kich thuoc toi da 50MB."
    )
    st.markdown(
        """
        <style>
            .status-box {
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    render_header()

    status_placeholder = st.empty()

    with st.container():
        st.subheader("Tai lieu can tai len")
        uploader_help = "Chon tep PDF hoac DOCX (toi da 50MB)."
        uploaded_file = st.file_uploader(
            "Chon tep",
            type=["pdf", "docx"],
            help=uploader_help,
        )

    if uploaded_file is None:
        status_placeholder.markdown(
            "<div class='status-box'>Chua co tep nao duoc chon.</div>",
            unsafe_allow_html=True,
        )
        return

    is_valid, validation_message = validate_file(uploaded_file)
    if not is_valid:
        status_placeholder.error(validation_message)
        return

    ensure_upload_directory(UPLOAD_DIR)

    with st.spinner("Dang xu ly tep..."):
        saved_path = persist_uploaded_file(uploaded_file, UPLOAD_DIR)
        backend_status = trigger_backend_ingestion(saved_path)

    if backend_status == 200:
        success_message = (
            "200: Upload thanh cong. Tep da duoc luu va dang cho xu ly backend."
        )
        status_placeholder.success(success_message)
        st.caption(f"Duong dan tam thoi: `{saved_path}`")
        if CHECKSUM_REQUIRED:
            st.info(
                "Checksum se duoc doi chieu o backend de dam bao tinh toan ven cua tep."
            )
    else:
        status_placeholder.error(
            f"Backend tra ve ma {backend_status}. Vui long thu lai hoac lien he ho tro."
        )


if __name__ == "__main__":
    main()

# Unit test placeholder:
# def test_validate_file_rejects_large_files():
#     """Vi du minh hoa noi viet unit test cho validate_file."""
#     ...
