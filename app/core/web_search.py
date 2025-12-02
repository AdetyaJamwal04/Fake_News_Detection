from ddgs import DDGS
from typing import List, Dict
import time


def web_search(queries: List[str], max_results: int = 6) -> List[Dict]:
    """
    Perform web search using DuckDuckGo for multiple queries.
    
    Args:
        queries: List of search query strings
        max_results: Maximum number of results per query
        
    Returns:
        List of deduplicated search results with 'href', 'title', and 'body' keys
    """
    all_results = []
    seen_urls = set()
    
    with DDGS() as ddgs:
        for query in queries:
            try:
                # Search with DuckDuckGo
                results = ddgs.text(
                    query,
                    max_results=max_results,
                    region='wt-wt',  # worldwide
                    safesearch='moderate'
                )
                
                for result in results:
                    url = result.get('href') or result.get('link')
                    
                    # Deduplicate by URL
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            'href': url,
                            'title': result.get('title', ''),
                            'body': result.get('body', '')
                        })
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                # Continue with other queries if one fails
                print(f"Search failed for query '{query}': {e}")
                continue
    
    return all_results
