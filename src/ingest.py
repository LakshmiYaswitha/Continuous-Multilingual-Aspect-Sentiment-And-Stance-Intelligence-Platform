from io import BytesIO

def extract_upload(uploaded):
    data=uploaded.getvalue(); name=uploaded.name.lower()
    if name.endswith('.txt'):
        return data.decode('utf-8', errors='ignore')
    if name.endswith('.pdf'):
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(BytesIO(data)).pages)
    if name.endswith('.docx'):
        from docx import Document
        d=Document(BytesIO(data)); return '\n'.join(p.text for p in d.paragraphs)
    if name.endswith('.doc'):
        raise ValueError('Legacy .doc files are not reliably parsed. Convert to .docx or PDF.')
    raise ValueError('Unsupported document type')
