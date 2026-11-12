from pathlib import Path
from langchain_core.documents import Document

file_path=Path("cricket.txt")
text=file_path.read_text(encoding="utf-8")

docs=[Document(
    page_content=text,
    metadata={"source":str(file_path)}
)]

print(docs)
 