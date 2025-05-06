#!/usr/bin/env python3
# filepath: /home/spay/dev/poc/hackathon/hande/Deepsearch/GoogleSearcher.py
from googlesearch import search
import re
from typing import List, Set, Optional


class GoogleSearcher:
    """
    A class to perform Google searches based on user queries and retrieve relevant URLs
    from specified domains.
    """
    
    def __init__(self, target_domain: str = "https://www.economie.gouv.fr/entreprises", 
                 stop_words: Optional[Set[str]] = None,
                 default_lang: str = "fr",
                 default_pause: float = 2.0):
        """
        Initialize the GoogleSearcher.
        
        Args:
            target_domain: The domain to restrict search results to
            stop_words: Set of words to exclude from keyword extraction
            default_lang: Default language for search
            default_pause: Time to pause between search requests (to avoid rate limiting)
        """
        self.target_domain = target_domain
        self.default_lang = default_lang
        self.default_pause = default_pause
        
        # Default French stop words if not provided
        if stop_words is None:
            self.stop_words = {'j', 'ai', 'une', 'de', 'la', 'le', 'les', 
                              'quand', 'dois', 'je', 'ma', 'mon', 'et', 'ou',
                              'qui', 'que', 'quoi', 'comment', 'pourquoi'}
        else:
            self.stop_words = stop_words
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from a query by removing stop words.
        
        Args:
            query: The user query string
            
        Returns:
            List of extracted keywords
        """
        words = re.findall(r'\w+', query.lower())
        keywords = [word for word in words if word not in self.stop_words]
        return keywords
    
    def construct_search_query(self, keywords: List[str]) -> str:
        """
        Construct a search query with domain restriction.
        
        Args:
            keywords: List of keywords to include in the query
            
        Returns:
            Formatted search query string
        """
        base_query = " ".join(keywords)
        site_restriction = f"site:{self.target_domain}"
        return f"{base_query} {site_restriction}"
    
    def search(self, query: str, num_results: int = 3) -> List[str]:
        """
        Search for relevant URLs based on a user query.
        
        Args:
            query: The user query string
            num_results: Maximum number of URLs to retrieve
            
        Returns:
            List of relevant URLs
        """
        # Extract keywords and build search query
        keywords = self.extract_keywords(query)
        search_query = self.construct_search_query(keywords)
        
        # Perform Google search
        urls = []
        try:
            # Note: The library parameter may be 'num' instead of 'num_results'
            for url in search(search_query, num=num_results, 
                             lang=self.default_lang, pause=self.default_pause):
                # Ensure URL is from the target domain
                if url.startswith(self.target_domain):
                    urls.append(url)
                if len(urls) >= num_results:
                    break
        except Exception as e:
            print(f"Error during search: {e}")
            # Try alternate parameter if first attempt failed
            if "num_results" in str(e):
                try:
                    for url in search(search_query, tld="fr", lang=self.default_lang, 
                                     pause=self.default_pause, stop=num_results):
                        if url.startswith(self.target_domain):
                            urls.append(url)
                        if len(urls) >= num_results:
                            break
                except Exception as e2:
                    print(f"Second attempt failed: {e2}")
        
        return urls
    
    def batch_search(self, queries: List[str], num_results_per_query: int = 3) -> dict:
        """
        Process multiple queries and return URLs for each.
        
        Args:
            queries: List of query strings
            num_results_per_query: Number of URLs to retrieve per query
            
        Returns:
            Dictionary mapping each query to its search results
        """
        results = {}
        for query in queries:
            urls = self.search(query, num_results_per_query)
            results[query] = urls
        return results


def main():
    # Create an instance of GoogleSearcher
    searcher = GoogleSearcher()
    
    # Example user query
    query = "J'ai fondé une SAS, quand dois-je faire ma déclaration de TVA ?"
    urls = searcher.search(query)
    
    # Output results
    if urls:
        print("Relevant URLs found:")
        for i, url in enumerate(urls, 1):
            print(f"{i}. {url}")
    else:
        print("No relevant URLs found.")
    
    # Example of batch processing
    print("\nBatch processing example:")
    queries = [
        "Comment créer une entreprise en France ?",
        "Quelles sont les aides pour les auto-entrepreneurs ?"
    ]
    batch_results = searcher.batch_search(queries, num_results_per_query=2)
    
    for query, result_urls in batch_results.items():
        print(f"\nQuery: {query}")
        if result_urls:
            for i, url in enumerate(result_urls, 1):
                print(f"{i}. {url}")
        else:
            print("No relevant URLs found.")


if __name__ == "__main__":
    main()
