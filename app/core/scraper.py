from trafilatura import fetch_url, extract

def scrape_article(url: str) -> str:
    """
    Extract cleaned article text from URL.
    """
    try:
        html = fetch_url(url)
        if not html:
            return ""
        return extract(html) or ""
    except Exception as e:
        # Optionally log the error for debugging
        # print(f"Error scraping {url}: {e}")
        return ""
