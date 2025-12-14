"""
Search tool for web information retrieval
"""

from typing import List, Dict, Any


class SearchTool:
    """Tool for searching the web using DuckDuckGo"""

    def __init__(self, max_results: int = 5):
        """
        Initialize search tool

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results

    def search(self, query: str) -> str:
        """
        Search the web and return formatted results

        Args:
            query: Search query string

        Returns:
            Formatted search results as string
        """
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

                if not results:
                    return "No search results found."

                # Format results
                formatted = []
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')
                    body = result.get('body', 'No description')
                    url = result.get('href', '')

                    formatted.append(f"{i}. {title}\n   {body}\n   URL: {url}")

                return "\n\n".join(formatted)

        except Exception as e:
            return f"Search failed: {str(e)}"

    def __call__(self, query: str) -> str:
        """Allow tool to be called directly"""
        return self.search(query)
