#!/usr/bin/env python3
# filepath: /home/spay/dev/poc/hackathon/hande/Deepsearch/search_extract_pipeline.py
"""
Complete search and extraction pipeline.
This module combines GoogleSearcher and SeleniumExtractor to create a full pipeline
for processing AI-generated questions, retrieving relevant URLs, and extracting content.
"""

from typing import List, Dict
import os
import json
from GoogleSearcher import GoogleSearcher
from SeleniumExtractor import SeleniumExtractor


class SearchExtractPipeline:
    """
    A pipeline that combines search and content extraction.
    """
    
    def __init__(self, 
                 target_domain: str = "https://en.wikipedia.org", 
                 output_dir: str = "extracted_content",
                 max_results_per_query: int = 3,
                 headless: bool = True,
                 wait_time: int = 3,
                 scroll_page: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            target_domain: Domain to restrict search results to
            output_dir: Directory to save extracted content
            max_results_per_query: Maximum number of URLs to retrieve per query
            headless: Whether to run browser in headless mode
            wait_time: Time to wait after page load for dynamic content to render
            scroll_page: Whether to scroll the page to load lazy-loaded content
        """
        self.searcher = GoogleSearcher(target_domain=target_domain)
        self.extractor = SeleniumExtractor(headless=headless, wait_time=wait_time, scroll_page=scroll_page)
        self.output_dir = output_dir
        self.max_results_per_query = max_results_per_query
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def process_questions(self, questions: List[str]) -> Dict[str, Dict]:
        """
        Process a list of questions through the complete pipeline.
        
        Args:
            questions: List of questions to process
            
        Returns:
            Dictionary with results for each question
        """
        # Step 1: Search for URLs
        print(f"Searching for URLs based on {len(questions)} questions...")
        search_results = self.searcher.batch_search(
            questions, 
            num_results_per_query=self.max_results_per_query
        )
        
        # Step 2: Extract content from URLs
        print("Extracting content from URLs...")
        
        all_results = {}
        
        for question_idx, (question, urls) in enumerate(search_results.items()):
            if not urls:
                print(f"No URLs found for question: {question}")
                all_results[question] = {"urls": [], "markdown_files": []}
                continue
                
            print(f"\nProcessing question {question_idx+1}/{len(questions)}")
            print(f"Found {len(urls)} URLs for: {question}")
            
            # Process the URLs for this question
            markdown_contents = self.extractor.process_urls(urls)
            
            # Save content to files
            question_filename = "".join(c if c.isalnum() else "_" for c in question)[:30]
            markdown_files = []
            
            # Only save for successful extractions
            successful_urls = []
            
            for url_idx, (url, content) in enumerate(zip(urls, markdown_contents)):
                if content:
                    successful_urls.append(url)
                    filename = f"{question_filename}_{url_idx+1}.md"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    markdown_files.append(filepath)
                    print(f"  Saved content to {filename}")
            
            # Store results for this question
            all_results[question] = {
                "urls": successful_urls,
                "markdown_files": markdown_files
            }
            
            # Create summary file with metadata
            summary_file = os.path.join(self.output_dir, f"{question_filename}_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({
                    "question": question,
                    "urls": successful_urls,
                    "markdown_files": markdown_files
                }, f, indent=2)
        
        return all_results
    
    def get_markdown_content(self, questions: List[str]) -> Dict[str, List[str]]:
        """
        Process questions and return the markdown content directly without saving to files.
        
        Args:
            questions: List of questions to process
            
        Returns:
            Dictionary mapping each question to a list of markdown content strings
        """
        # Step 1: Search for URLs
        search_results = self.searcher.batch_search(
            questions, 
            num_results_per_query=self.max_results_per_query
        )
        
        # Step 2: Extract content from URLs
        markdown_results = {}
        
        for question, urls in search_results.items():
            if not urls:
                markdown_results[question] = []
                continue
                
            # Process the URLs for this question
            markdown_contents = self.extractor.process_urls(urls)
            
            # Filter out None results
            markdown_results[question] = [
                content for content in markdown_contents if content
            ]
        
        return markdown_results


def main():
    # Example: List of questions that might come from an AI model
    ai_generated_questions = [
        "What is web scraping?",
        "What is Beautiful Soup library in Python?",
        "What is the Python requests library used for?",
    ]
    
    # Create and run the pipeline with Wikipedia as the target domain
    # (More likely to allow scraping than government sites)
    pipeline = SearchExtractPipeline(target_domain="https://en.wikipedia.org")
    results = pipeline.process_questions(ai_generated_questions)
    
    # Display summary
    print("\n--- Summary ---")
    for question, data in results.items():
        print(f"\nQuestion: {question}")
        print(f"Found {len(data['urls'])} relevant URLs")
        print(f"Created {len(data['markdown_files'])} markdown files")
    
    print(f"\nAll content has been saved to the '{pipeline.output_dir}' directory.")
    

if __name__ == "__main__":
    main()
