from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                    metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm
    import re
    
    metadata = metadata or {}
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # Load model and encode sentences
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    
    # Function to calculate cosine similarity
    def cosine_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b) + 1e-9)
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_embedding = None
    
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        if i == 0:
            # First sentence starts a new chunk
            current_chunk.append(sentence)
            current_embedding = embedding
        else:
            # Calculate similarity with previous sentence
            similarity = cosine_sim(current_embedding, embedding)
            
            if similarity < threshold:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    metadata={**metadata, "strategy": "semantic"}
                ))
                # Start new chunk
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                # Update embedding (average of all sentences in chunk)
                current_embedding = (current_embedding + embedding) / 2
    
    # Add the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(Chunk(
            text=chunk_text.strip(),
            metadata={**metadata, "strategy": "semantic"}
        ))
    
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                        child_size: int = HIERARCHICAL_CHILD_SIZE,
                        metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Create parent chunks
    parents = []
    children = []
    
    # Group paragraphs into parent chunks
    current_parent = ""
    parent_id = f"parent_{len(parents)}"
    
    for i, para in enumerate(paragraphs):
        if len(current_parent) + len(para) <= parent_size and current_parent:
            current_parent += "\n\n" + para
        else:
            if current_parent:
                parents.append(Chunk(
                    text=current_parent.strip(),
                    metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id}
                ))
            
            # Start new parent chunk
            current_parent = para
            parent_id = f"parent_{len(parents)}"
    
    # Add the last parent chunk
    if current_parent:
        parents.append(Chunk(
            text=current_parent.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id}
        ))
    
     # Create children chunks from each parent
    for parent in parents:
        parent_text = parent.text
        parent_id = parent.metadata.get("parent_id", "")
        
        # Split parent into children
        if len(parent_text) <= child_size:
            # If parent is already smaller than child_size, just use it as a child
            children.append(Chunk(
                text=parent_text,
                parent_id=parent_id
            ))
        else:
            # Split parent into smaller chunks that are less than child_size
            sentences = [s.strip() for s in parent_text.split(". ") if s.strip()]
            current_child = ""
            
            for sentence in sentences:
                # Check if adding this sentence would exceed child_size
                if len(current_child) + len(sentence) + 1 <= child_size and current_child:
                    current_child += ". " + sentence
                else:
                    # If we have accumulated content, save it as a child
                    if current_child:
                        children.append(Chunk(
                            text=current_child.strip() + ".",
                            parent_id=parent_id
                        ))
                    
                    # Start new child chunk - but make sure it's not larger than child_size
                    if len(sentence) <= child_size:
                        current_child = sentence
                    else:
                        # If sentence itself is larger than child_size, we need to split it
                        # For simplicity, we'll just take the first part that fits
                        current_child = sentence[:child_size-5] + "..."
            
            # Add the last child chunk
            if current_child:
                children.append(Chunk(
                    text=current_child.strip() + ".",
                    parent_id=parent_id
                ))
    
    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    
    # Split text into sections based on markdown headers
    import re
    sections = re.split(r'(^(#{1,3}\s+.+$))', text, flags=re.MULTILINE)
    
    # Combine headers with their content
    chunks = []
    current_header = ""
    current_content = ""
    
    # Process sections in pairs (header, content)
    i = 0
    while i < len(sections):
        if i + 1 < len(sections) and re.match(r'^#{1,3}\s+.+$', sections[i].strip()):
            # This is a header
            if current_header and current_content:
                # Save the previous section
                chunks.append(Chunk(
                    text=f"{current_header}\n\n{current_content.strip()}",
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
                current_content = ""
            
            current_header = sections[i].strip()
        else:
            # This is content
            if current_header:
                current_content += sections[i]
            else:
                # If no header yet, treat as content under a default section
                current_content += sections[i]
        
        i += 1
    
    # Don't forget the last section
    if current_header and current_content:
        chunks.append(Chunk(
            text=f"{current_header}\n\n{current_content.strip()}",
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))
    
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
