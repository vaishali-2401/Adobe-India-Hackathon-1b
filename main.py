import fitz  # PyMuPDF
import json
import nltk
import re
from nltk.tokenize import sent_tokenize
from test import TextChunker
import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sentence_transformers.util import cos_sim
import chromadb
from chromadb.config import Settings

# Extract H1-based chunks from PDF
def extract_chunks_from_pdf(input_json_path):
    all_chunks = []
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    for doc in input_data['documents']:
        filename = f"PDFs/{doc['filename']}"  # PDFs are in PDFs folder
        title = doc['title']
        outline_path = f"output1a/{title}.json"  # Outline files are in output1a folder

        with fitz.open(filename) as pdf:
            full_text = "\n".join([page.get_text() for page in pdf])

        with open(outline_path, "r", encoding="utf-8") as f:
            outline = json.load(f)['outline']

        h1_items = [item for item in outline if item["level"] in {"H1"}]
        h1_positions = []

        for h1 in h1_items:
            heading_text = h1['text'].strip()
            position = full_text.find(heading_text)
            if position == -1:
                
                continue
            h1_positions.append({
                "text": heading_text,
                "position": position,
                "page": h1["page"]
            })

        h1_positions.sort(key=lambda x: x['position'])

        for i in range(len(h1_positions)):
            start = h1_positions[i]['position']
            end = h1_positions[i + 1]['position'] if i + 1 < len(h1_positions) else len(full_text)
            content = full_text[start:end].strip()
            if not content:
                continue

            content = re.sub(r'\s{2,}', ' ', content)
            sentences = sent_tokenize(content)
            clean_text = " ".join(sentences)

            chunk = {
                "section_title": h1_positions[i]["text"],
                "document": filename,
                "content": clean_text,
                "page_range": [h1_positions[i]["page"]]
            }
            all_chunks.append(chunk)
            
    return all_chunks


# Export function
def export_to_structured_json(all_refined_chunks, input_documents, persona, job_to_be_done, output_path="final_output.json"):
    metadata = {
        "input_documents": input_documents,
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": datetime.datetime.now().isoformat()
    }

    grouped = defaultdict(list)
    for chunk in all_refined_chunks:
        key = (chunk['document'], chunk['section_title'])
        grouped[key].append(chunk)

    sorted_sections = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

    extracted_sections = []
    for rank, ((document, section_title), chunks) in enumerate(sorted_sections, start=1):
        extracted_sections.append({
            "document": document,
            "section_title": section_title,
            "importance_rank": rank,
            "page_number": chunks[0]["page_range"]
        })

    subsection_analysis = []
    for chunk in all_refined_chunks:
        subsection_analysis.append({
            "document": chunk["document"],
            "refined_text": chunk["sub_chunk"],
            "page_number": chunk["page_range"]
        })

    final_output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    


# Main processing
def main():
    h1_chunks = extract_chunks_from_pdf("challenge1b_input.json")
    chunker = TextChunker()
    all_refined_chunks = []

    for chunk in h1_chunks:
        
        sentences = sent_tokenize(chunk["content"])
        contextualized = chunker._add_context(sentences, window_size=1)
        embeddings = chunker.model.encode(contextualized)
        distances = chunker._calculate_distances(embeddings)

        if len(distances) == 0:
            
            continue

        breakpoints = chunker._identify_breakpoints(distances, threshold_percentile=9)
        initial_chunks = chunker._create_chunks(sentences, breakpoints)
        initial_chunks = [" "] + initial_chunks
        chunk_embeddings = chunker.model.encode(initial_chunks)
        semantic_chunks = chunker._merge_small_chunks(initial_chunks, chunk_embeddings, min_size=3)

        for sub in semantic_chunks:
            all_refined_chunks.append({
                "section_title": chunk["section_title"],
                "sub_chunk": sub,
                "document": chunk["document"],
                "page_range": chunk["page_range"]
            })

    # ChromaDB setup
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    if "document_chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("document_chunks")
    collection = client.create_collection(name="document_chunks")
    model = SentenceTransformer("models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

    texts, metadatas, ids = [], [], []
    for i, chunk in enumerate(all_refined_chunks):
        text = chunk["section_title"] + ": " + chunk["sub_chunk"]
        if len(text) < 400:
            continue
        texts.append(text)
        metadatas.append({
            "section_title": chunk["section_title"],
            "document": chunk["document"],
            "page_range": chunk["page_range"][0],
        })
        ids.append(f"chunk_{i}")
    embeddings = model.encode(texts).tolist()
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

    # Ranking top 5 refined chunks
    query = "A travel planner: Plan a trip of 4 days for a group of 10 college friends."
    query_embedding = model.encode([query])
    results = collection.get(include=["documents", "metadatas", "embeddings"])
    stored_embeddings = np.array(results["embeddings"])
    stored_documents = results["documents"]
    stored_metadatas = results["metadatas"]
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:15]

    filtered = []
    seen_embeddings = []

    for idx in top_indices:
        embedding = torch.tensor(stored_embeddings[idx])
        if all(cos_sim(embedding, e).item() < 0.95 for e in seen_embeddings):
            seen_embeddings.append(embedding)
            filtered.append((idx, similarities[idx]))
            if len(filtered) == 5:
                break

    # Prepare top 5 refined chunks
    top_refined_chunks = []
    for idx, _ in filtered:
        full_text = stored_documents[idx]
        section_title, sub_chunk = full_text.split(":", 1)
        top_refined_chunks.append({
            "section_title": section_title.strip(),
            "sub_chunk": sub_chunk.strip(),
            "document": stored_metadatas[idx]['document'],
            "page_range": stored_metadatas[idx]['page_range']# default page number if unknown
        })

    # Metadata inputs
    with open("challenge1b_input.json", "r", encoding="utf-8") as f:
      input_data = json.load(f)

    input_documents = [f"PDFs/{doc['filename']}" for doc in input_data["documents"]]
    persona = input_data.get("persona", {}).get("role", "")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")



    # Save only top 5 chunks
    export_to_structured_json(
        all_refined_chunks=top_refined_chunks,
        input_documents=input_documents,
        persona=persona,
        job_to_be_done=job_to_be_done,
        output_path="challenge1b_output.json"
    )


if __name__ == "__main__":
    main()
