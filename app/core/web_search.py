from tavily import TavilyClient
from typing import List, Dict
import os
import time


def web_search(queries: List[str], max_results: int = 6) -> List[Dict]:
    """
    Perform web search using Tavily AI Search for multiple queries.
    
    Args:
        queries: List of search query strings
        max_results: Maximum number of results per query
        
    Returns:
        List of deduplicated search results with 'href', 'title', and 'body' keys
    """
    all_results = []
    seen_urls = set()
    
    # Get API key from environment
    api_key = os.getenv('TAVILY_API_KEY', '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')
    
    try:
        # Initialize Tavily client
        tavily = TavilyClient(api_key=api_key)
        
        for query in queries:
            try:
                # Search with Tavily - optimized for fact-checking
                response = tavily.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced",  # More thorough search
                    include_domains=[],  # All domains
                    exclude_domains=["youtube.com", "facebook.com", "twitter.com"]  # Exclude social media
                )
                
                # Extract results
                results = response.get('results', [])
                
                for result in results:
                    url = result.get('url')
                    
                    # Deduplicate by URL
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            'href': url,
                            'title': result.get('title', ''),
                            'body': result.get('content', '')  # Tavily uses 'content' instead of 'body'
                        })
                
                # Small delay to avoid rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                # Continue with other queries if one fails
                print(f"Search failed for query '{query}': {e}")
                continue
                
    except Exception as e:
        print(f"Tavily client initialization failed: {e}")
        return []
    
    return all_results
