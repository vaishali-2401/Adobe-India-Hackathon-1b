"""
Text Chunking Utility

This module provides functionality to intelligently chunk text documents into semantically coherent sections
using sentence embeddings and cosine similarity. It's particularly useful for processing large documents
while maintaining contextual relationships between sentences.

Requirements:
    - nltk
    - sentence-transformers
    - scikit-learn
    - numpy
    - matplotlib
"""
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


class TextChunker:
    def __init__(self, model_name='models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'):
        """Initialize the TextChunker with a specified sentence transformer model."""
        self.model = SentenceTransformer(model_name)
    
    def process_file(self, file_path, context_window=1, percentile_threshold=95, min_chunk_size=3):
        """
        Process a text file and split it into semantically meaningful chunks.
        
        Args:
            file_path: Path to the text file
            context_window: Number of sentences to consider on either side for context
            percentile_threshold: Percentile threshold for identifying breakpoints
            min_chunk_size: Minimum number of sentences in a chunk
            
        Returns:
            list: Semantically coherent text chunks
        """
        # Process the text file
        sentences = self._load_text(file_path)
        contextualized = self._add_context(sentences, context_window)
        embeddings = self.model.encode(contextualized)
        
        # Create and refine chunks
        distances = self._calculate_distances(embeddings)
        breakpoints = self._identify_breakpoints(distances, percentile_threshold)
        initial_chunks = self._create_chunks(sentences, breakpoints)
        
        # Merge small chunks for better coherence
        chunk_embeddings = self.model.encode(initial_chunks)
        final_chunks = self._merge_small_chunks(initial_chunks, chunk_embeddings, min_chunk_size)
        
        return final_chunks
    import fitz  # PyMuPDF

    def _load_text(self, file_path):
        """Load and tokenize text from a file."""
        if file_path.endswith(".pdf"):
            text =text = ""
            with fitz.open(file_path) as doc:
             for page in doc:
                text += page.get_text()
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
             text = file.read()
        return sent_tokenize(text)

    def _add_context(self, sentences, window_size):
        """Combine sentences with their neighbors for better context."""
        contextualized = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context = ' '.join(sentences[start:end])
            contextualized.append(context)
        return contextualized

    def _calculate_distances(self, embeddings):
        """Calculate cosine distances between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _identify_breakpoints(self, distances, threshold_percentile):
        """Find natural breaking points in the text based on semantic distances."""
        threshold = np.percentile(distances, threshold_percentile)
        return [i for i, dist in enumerate(distances) if dist > threshold]

    def _create_chunks(self, sentences, breakpoints):
        """Create initial text chunks based on identified breakpoints."""
        chunks = []
        start_idx = 0
        
        for breakpoint in breakpoints:
            chunk = ' '.join(sentences[start_idx:breakpoint + 1])
            chunks.append(chunk)
            start_idx = breakpoint + 1
            
        # Add the final chunk
        final_chunk = ' '.join(sentences[start_idx:])
        chunks.append(final_chunk)
        
        return chunks

    def _merge_small_chunks(self, chunks, embeddings, min_size):
        """Merge small chunks with their most similar neighbor."""
        final_chunks = [chunks[0]]
        merged_embeddings = [embeddings[0]]
        
        for i in range(1, len(chunks) - 1):
            current_chunk_size = len(chunks[i].split('. '))
            
            if current_chunk_size < min_size:
                # Calculate similarities
                prev_similarity = cosine_similarity([embeddings[i]], [merged_embeddings[-1]])[0][0]
                next_similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                
                if prev_similarity > next_similarity:
                    # Merge with previous chunk
                    final_chunks[-1] = f"{final_chunks[-1]} {chunks[i]}"
                    merged_embeddings[-1] = (merged_embeddings[-1] + embeddings[i]) / 2
                else:
                    # Merge with next chunk
                    chunks[i + 1] = f"{chunks[i]} {chunks[i + 1]}"
                    embeddings[i + 1] = (embeddings[i] + embeddings[i + 1]) / 2
            else:
                final_chunks.append(chunks[i])
                merged_embeddings.append(embeddings[i])
        
        final_chunks.append(chunks[-1])
        return final_chunks


def main():
    """Example usage of the TextChunker class."""
    # Initialize the chunker
    chunker = TextChunker()
    
    # Process a text file
    file_path = r"C:\Users\jayan\OneDrive\Desktop\adobe\South of France - Cities.pdf"
    chunks = chunker.process_file(
        file_path,
        context_window=1,
        percentile_threshold=95,
        min_chunk_size=3
    )
    
    # Print results
    print(f"Successfully split text into {len(chunks)} chunks")
    print("\nFirst chunk preview:")
    for i in chunks:
        print(i)
        print("\n----------\n")


if __name__ == "__main__":
    main()
