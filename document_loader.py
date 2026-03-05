from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredImageLoader,
)
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document
import os

def detect_file_type(file_path:str)->str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg",".jpeg",".png",".bmp",".tiff",".webp"]:
        return "image"
    if ext == ".docx":
        return "docx"
    if ext == ".txt":
        return "txt"
    if ext == ".pdf":
        return check_pdf_type(file_path)
    #fallback
    return "text_pdf"

def check_pdf_type(pdf_path:str)->str:
    try:
        doc = fitz.open(pdf_path)
        total_text = ""
        for page in doc:
            total_text += page.get_text()
        avg_chars = len(total_text.strip())/max(1,len(doc))

        if avg_chars <20:
            return "scanned_pdf"
        return "text_pdf"
    except :
        return "text_pdf"
    
#loaders
def load_text_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_scanned_pdf(file_path):
    docs = []
    pdf = fitz.open(file_path)
    for page_num , page in enumerate(pdf, start=1):
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image)
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page":page_num, "source":file_path},
                )
            )
    return docs

def load_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    if text.strip():
        return [
            Document(
                page_content=text,
                metadata={"page":1, "source":file_path},
            )
        ]
    return []

def load_docx(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()

def load_text(file_path):
    loader = TextLoader(file_path)
    return loader.load()

def smart_load(file_path):
    file_type = detect_file_type(file_path)
    if file_type == "text_pdf":
        return load_text_pdf(file_path)
    elif file_type == "scanned_pdf":
        return load_scanned_pdf(file_path)
    elif file_type == "image":
        return load_image(file_path)
    elif file_type == "docx":
        return load_docx(file_path)
    elif file_type == "txt":
        return load_text(file_path)
    return load_text_pdf(file_path)
    
