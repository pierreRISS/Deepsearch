#!/usr/bin/env python3
# filepath: /home/spay/dev/poc/hackathon/hande/Deepsearch/SeleniumExtractor.py
"""
SeleniumExtractor module for retrieving and processing web content from URLs using Selenium.
This module uses a real browser to bypass anti-scraping measures and extract content from websites.
"""

import time
import re
from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from vector_db import process_documents, create_vector_database, semantic_search, display_results, process_markdown_content


class SeleniumExtractor:
    """
    A class to extract and process content from web pages using Selenium
    and convert it to markdown format.
    """
    
    def __init__(self, 
                 headless: bool = True,
                 timeout: int = 30,
                 max_content_length: int = 20000,  # Increased from 5000 to 20000
                 wait_time: int = 3,
                 scroll_page: bool = True):
        """
        Initialize the SeleniumExtractor.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Timeout for page loads in seconds
            max_content_length: Maximum length of content to extract from each page
            wait_time: Time to wait after page load for dynamic content to render
            scroll_page: Whether to scroll the page to load lazy-loaded content
        """
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.wait_time = wait_time
        self.scroll_page = scroll_page
        self.driver = None
    
    def _setup_driver(self):
        """Set up the Chrome WebDriver."""
        if self.driver is not None:
            return
            
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        
        # Add common options to make the browser more stealthy
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        # Set user agent to look like a real browser
        options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36")
        
        # Add additional preferences to avoid detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        # Initialize the WebDriver with ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        
        # Set page load timeout
        self.driver.set_page_load_timeout(self.timeout)
        
        # Execute script to bypass detection
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    
    def _scroll_page(self):
        """Scroll the page to load lazy content."""
        try:
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            for i in range(1, total_height, 200):
                self.driver.execute_script(f"window.scrollTo(0, {i});")
                time.sleep(0.1)
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
        except Exception as e:
            print(f"Error scrolling page: {e}")
    
    def _close_driver(self):
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                print(f"Error closing driver: {e}")
            finally:
                self.driver = None
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch the HTML content of a web page using Selenium.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML content as string or None if request failed
        """
        try:
            self._setup_driver()
            
            print(f"Loading page: {url}")
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(self.wait_time)
            
            # Scroll the page if enabled
            if self.scroll_page:
                self._scroll_page()
            
            # Get the page source
            html = self.driver.page_source
            return html
            
        except TimeoutException:
            print(f"Timeout loading {url}")
            return None
        except WebDriverException as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching {url}: {str(e)}")
            return None
    
    def extract_content(self, html: str) -> Tuple[str, str, List[str], Optional[BeautifulSoup]]:
        """
        Extract the main content and subheadings from HTML.
        
        Args:
            html: HTML content as string
            
        Returns:
            Tuple containing (title, main content text, list of subheadings, main content element)
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.select('script, style, noscript, svg, footer, nav, aside, iframe, [role="banner"], [role="navigation"], .cookie-banner, .share-buttons, .sidebar, .ads, #header, #footer, [class*="cookie"], [class*="banner"], [class*="popup"], [id*="modal"], [class*="modal"]'):
            if element:
                element.decompose()
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag and title_tag.text:
            title = title_tag.text.strip()
        
        # Extract headings for structure
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            if heading.text.strip():
                headings.append((heading.name, heading.text.strip()))
        
        # More sophisticated content extraction
        main_content_text = ""
        main_content_element = None
        content_candidates = [
            # Try common content containers
            soup.find('main'),
            soup.find('article'),
            soup.find(id='content'),
            soup.find(attrs={"class": "content"}),
            soup.find(attrs={"class": "main-content"}),
            soup.find(attrs={"class": "article-content"}),
            soup.find(attrs={"class": "post-content"}),
            soup.find(attrs={"itemprop": "articleBody"}),
            soup.find(attrs={"role": "main"})
        ]
        
        # Filter out None values
        content_candidates = [c for c in content_candidates if c]
        
        if content_candidates:
            # Use the first found content container
            main_content_element = content_candidates[0]
            main_content_text = main_content_element.get_text(separator='\n\n', strip=True)
        else:
            # Fallback to content with most text
            content_blocks = {}
            for div in soup.find_all(['div', 'section']):
                text = div.get_text(separator='\n\n', strip=True)
                if len(text) > 200:  # Only consider blocks with substantial text
                    content_blocks[div] = len(text)
            
            if content_blocks:
                main_content_element = max(content_blocks, key=content_blocks.get)
                main_content_text = main_content_element.get_text(separator='\n\n', strip=True)
            else:
                # Last resort: just get the body text
                body = soup.find('body')
                if body:
                    main_content_element = body
                    main_content_text = body.get_text(separator='\n\n', strip=True)
        
        # Limit content length if needed
        if len(main_content_text) > self.max_content_length:
            main_content_text = main_content_text[:self.max_content_length] + "..."
        
        # Extract paragraphs and format for better readability
        paragraphs = re.split(r'\n\s*\n', main_content_text)
        formatted_content = '\n\n'.join(p.strip() for p in paragraphs if p.strip())
        
        # Extract extracted headings as a separate list
        subheadings = [heading_text for _, heading_text in headings]
        
        return title, formatted_content, subheadings, main_content_element

    def to_markdown(self, url: str, title: str, content: str, subheadings: List[str], html_element: Optional[BeautifulSoup] = None) -> str:
        """
        Convert the extracted content to markdown format.
        
        Args:
            url: The source URL
            title: The page title
            content: The main content text
            subheadings: List of subheadings found in the content
            html_element: Optional BeautifulSoup element containing the main content HTML
            
        Returns:
            Markdown-formatted string
        """
        markdown = f"# {title}\n\n"
        markdown += f"*Source: {url}*\n\n"
        
        # Add subheadings if available
        if subheadings:
            markdown += "## Table of Contents\n\n"
            for heading in subheadings[:10]:  # Limit to top 10 headings
                markdown += f"- {heading}\n"
            markdown += "\n"
        
        # Add main content with proper formatting
        markdown += "## Full Content\n\n"
        
        # Use markdownify if we have the HTML element
        if html_element:
            # Convert the HTML directly to markdown
            markdown_content = md(str(html_element), heading_style="ATX")
            
            # Clean up the markdown - remove excessive newlines and fix spacing
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
            
            # Remove any empty headings
            markdown_content = re.sub(r'#+\s*\n', '', markdown_content)
            
            # Add the properly converted content
            markdown += markdown_content
        else:
            # Fallback to the extracted text content
            markdown += content
        
        return markdown
    
    def process_url(self, url: str) -> Optional[str]:
        """
        Process a single URL and return markdown-formatted content.
        
        Args:
            url: URL to process
            
        Returns:
            Markdown-formatted content or None if processing failed
        """
        html = self.fetch_page(url)
        if not html:
            return None
        
        try:
            title, content, subheadings, html_element = self.extract_content(html)
            return self.to_markdown(url, title, content, subheadings, html_element)
        except Exception as e:
            print(f"Error processing content from {url}: {str(e)}")
            return None
        
    def process_urls(self, urls: List[str]) -> List[str]:
        """
        Process multiple URLs and return markdown-formatted content for each.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of markdown-formatted strings (None entries for failed URLs are filtered out)
        """
        try:
            results = []
            for url in urls:
                markdown = self.process_url(url)
                if markdown:
                    results.append(markdown)
            
            return results
        finally:
            # Always close the driver when done
            self._close_driver()


def main():
    # Example usage
    extractor = SeleniumExtractor(headless=True, wait_time=5, scroll_page=True)
    
    # Example URLs
    urls = [
        "https://www.economie.gouv.fr/entreprises/professionnels-certification-logiciels-systemes-caisse",
        # Add more URLs to test if needed
        "https://www.python.org/about/",
        "https://en.wikipedia.org/wiki/Web_scraping"
    ]

    markdown_contents = extractor.process_urls(urls)
    query = "Qu'est-ce qu'un système de caisse"
    process_markdown_content(markdown_contents, query)
    if not markdown_contents:
        print("\nNo content could be extracted. All requests failed.")
    else:
        # Create output directory if it doesn't exist
        import os
        output_dir = "extracted_content"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, content in enumerate(markdown_contents, 1):
            print(f"\n--- Document {i} ---\n")
            # Show a preview of the content (first 500 characters)
            print(content[:500] + "...\n")
            
            # Save the full content to file
            filename = f"markdownify_example_{i}.md"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Full content saved to {filepath}")
            print("\n" + "-" * 80)
        
        print(f"\nSuccessfully processed {len(markdown_contents)} out of {len(urls)} URLs.")
        print(f"Output saved to the '{output_dir}' directory.")


if __name__ == "__main__":
    main()
