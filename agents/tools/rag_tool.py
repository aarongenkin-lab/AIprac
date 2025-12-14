"""
RAG (Retrieval-Augmented Generation) Tool
Allows agents to retrieve relevant information from a knowledge base
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import re


class RAGTool:
    """
    Simple RAG system for retrieving relevant documents.
    Uses TF-IDF or simple keyword matching for retrieval.
    """

    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        """
        Initialize RAG tool

        Args:
            knowledge_base_path: Path to directory containing knowledge documents
        """
        self.kb_path = Path(knowledge_base_path)
        self.documents = []
        self.use_embeddings = False

        # Try to import vector search libraries
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.np = np
            self.use_embeddings = True
        except ImportError:
            self.model = None
            self.use_embeddings = False

        self._load_documents()

    def _load_documents(self):
        """Load all documents from knowledge base"""
        if not self.kb_path.exists():
            self.kb_path.mkdir(parents=True, exist_ok=True)
            return

        self.documents = []

        # Load text files
        for file_path in self.kb_path.glob("**/*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'content': content,
                    'type': 'txt'
                })

        # Load markdown files
        for file_path in self.kb_path.glob("**/*.md"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'content': content,
                    'type': 'md'
                })

        # Load JSON files
        for file_path in self.kb_path.glob("**/*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                self.documents.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'content': json.dumps(content, indent=2),
                    'type': 'json'
                })

        # Compute embeddings if available
        if self.use_embeddings and self.documents:
            self._compute_embeddings()

    def _compute_embeddings(self):
        """Compute embeddings for all documents"""
        if not self.use_embeddings or not self.documents:
            return

        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.model.encode(texts)

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant documents
        """
        if not self.documents:
            return []

        query_words = set(query.lower().split())
        scores = []

        for doc in self.documents:
            content_lower = doc['content'].lower()
            # Count keyword matches
            matches = sum(1 for word in query_words if word in content_lower)

            # Boost score for matches in filename
            filename_matches = sum(1 for word in query_words if word in doc['filename'].lower())
            score = matches + (filename_matches * 2)

            scores.append((score, doc))

        # Sort by score and return top k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scores[:top_k] if score > 0]

    def _embedding_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embedding-based semantic search

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant documents
        """
        if not self.use_embeddings or not self.documents:
            return self._keyword_search(query, top_k)

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Compute cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.np.dot(query_embedding, doc_embedding) / (
                self.np.linalg.norm(query_embedding) * self.np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, self.documents[i]))

        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [doc for sim, doc in similarities[:top_k]]

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant documents based on query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Formatted string with retrieved documents
        """
        if not self.documents:
            return "Knowledge base is empty. No documents found."

        # Use embedding search if available, otherwise keyword search
        if self.use_embeddings:
            results = self._embedding_search(query, top_k)
        else:
            results = self._keyword_search(query, top_k)

        if not results:
            return f"No relevant documents found for query: '{query}'"

        # Format results
        formatted = [f"Retrieved {len(results)} relevant document(s):\n"]

        for i, doc in enumerate(results, 1):
            formatted.append(f"{'='*60}")
            formatted.append(f"Document {i}: {doc['filename']}")
            formatted.append(f"{'='*60}")

            # Truncate long documents
            content = doc['content']
            if len(content) > 2000:
                content = content[:2000] + f"\n... (truncated, {len(doc['content'])} total chars)"

            formatted.append(content)
            formatted.append("")

        return "\n".join(formatted)

    def add_document(self, filename: str, content: str):
        """
        Add a new document to the knowledge base

        Args:
            filename: Name of the document
            content: Content of the document
        """
        file_path = self.kb_path / filename

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Reload documents
        self._load_documents()

    def list_documents(self) -> str:
        """List all documents in knowledge base"""
        if not self.documents:
            return "Knowledge base is empty."

        doc_list = ["Available documents:"]
        for doc in self.documents:
            size = len(doc['content'])
            doc_list.append(f"  - {doc['filename']} ({size} chars, {doc['type']})")

        return "\n".join(doc_list)

    def __call__(self, query: str, top_k: int = 3) -> str:
        """Allow tool to be called directly"""
        return self.retrieve(query, top_k)


# Example usage
if __name__ == "__main__":
    # Create RAG tool
    rag = RAGTool()

    # Add example document
    zebra_strategies = """
# Zebra Puzzle Solving Strategies

## Step 1: Set Up the Grid
Create a grid with:
- Rows: positions (1st, 2nd, 3rd, etc.)
- Columns: attributes (color, nationality, pet, drink, etc.)

## Step 2: Extract Direct Facts
Look for clues that directly state a fact:
- "The Englishman lives in the red house" → Mark this in grid

## Step 3: Process Positional Clues
Handle clues about relative positions:
- "next to" means adjacent (position ± 1)
- "immediately to the right" means position + 1
- "to the left of" means smaller position number

## Step 4: Use Process of Elimination
- If house 1 can't be red, mark it as impossible
- When only one option remains for a cell, fill it in
- Check all constraints after each deduction

## Step 5: Make Logical Inferences
Combine multiple clues:
- If A is next to B, and B is at position 3, then A is at 2 or 4
- If C is to the left of D, and D is at 2, then C must be at 1

## Common Mistakes to Avoid
1. Don't assume "next to" means immediate right
2. Remember "to the right" is NOT the same as "next to"
3. Always check ALL constraints before confirming a deduction
4. Don't forget that each attribute appears exactly once

## Example Clue Processing
Clue: "The Norwegian lives in the first house"
Action: Mark position 1, nationality column = Norwegian

Clue: "The green house is immediately to the right of the ivory house"
Action: Green cannot be position 1. If ivory is at N, green is at N+1.
"""

    rag.add_document("zebra_strategies.md", zebra_strategies)

    print("Documents loaded:")
    print(rag.list_documents())
    print("\n")

    # Test retrieval
    print("Query: 'How do I solve positional clues in zebra puzzles?'")
    print("-" * 60)
    result = rag.retrieve("positional clues zebra puzzle", top_k=1)
    print(result)
