"""
Scraper for Automation Anywhere Blog Posts.

- Scrapes blog posts from automationanywhere.com
"""

import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from nlp import process_blogs, build_indices

''' Configuration '''

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BLOGS_FILE = os.path.join(DATA_DIR, "blogs.json")
REQUEST_DELAY = 1.5  # seconds between requests

# Blog listing pages to scrape links from
BLOG_LISTING_URLS = [
    "https://www.automationanywhere.com/company/blog/browse?category[0]=Automation%20%2B%20AI",
    "https://www.automationanywhere.com/company/blog/browse?category[0]=General%20Technology",
    "https://www.automationanywhere.com/company/blog/browse?category[0]=Product%20Insights",
    "https://www.automationanywhere.com/company/blog/browse?category[0]=Learn%20RPA",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}


''' Discover Blog URLs '''

def discover_blog_urls():
    """Collect blog post URLs from listing pages."""
    blog_urls = set()

    # Hardcoded known blog URLs (as a fallback + seed)
    known_urls = [
        "https://www.automationanywhere.com/company/blog/automation-ai/inside-shift-agentic-intelligence-and-how-enterprises-can-lead-2026",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-in-information-technology-industry",
        "https://www.automationanywhere.com/company/blog/automation-ai/agentic-ai-itsm",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-in-workplace",
        "https://www.automationanywhere.com/company/blog/general-technology/delivering-a-fully-on-prem-enterprise-ai-automation-platform-with-nvidia-nemotron-super",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-in-itsm",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-knowledge-management",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-orchestration",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-reasoning",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-in-fintech",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-in-business-operations",
        "https://www.automationanywhere.com/company/blog/learn-rpa/iot-technology-how-it-works-and-its-benefits",
        "https://www.automationanywhere.com/company/blog/automation-ai/ai-accounts-payable-how-ai-transforms-ap",
        "https://www.automationanywhere.com/company/blog/product-insights/next-frontier-agentic-ai-say-hello-workstreams",
    ]
    blog_urls.update(known_urls)
    
    # Try to discover more from listing pages
    for listing_url in BLOG_LISTING_URLS:
        try:
            print(f"  Scanning: {listing_url[:80]}...")
            resp = requests.get(listing_url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if "/company/blog/" in href and href.count("/") >= 6:
                        if href.startswith("/"):
                            href = "https://www.automationanywhere.com" + href
                        # Filter out listing/browse pages
                        if "/browse" not in href and "/category" not in href:
                            blog_urls.add(href)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f" Error scanning {listing_url}: {e}")

    print(f"  Found {len(blog_urls)} blog URLs")
    return list(blog_urls)


''' Scrape Blog Content '''

def scrape_blog(url):
    """Scrape a single blog post and return structured data."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}: {url}")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Extract title
        title = ""
        title_tag = soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)
        if not title:
            og_title = soup.find("meta", property="og:title")
            title = og_title["content"] if og_title else url.split("/")[-1]

        # Extract meta description
        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"]

        # Extract main content - try common blog content containers
        content = ""
        # Try various content selectors
        content_selectors = [
            soup.find("article"),
            soup.find("div", class_=re.compile(r"blog[-_]?content|article[-_]?body|post[-_]?content|entry[-_]?content", re.I)),
            soup.find("div", class_=re.compile(r"field--name-body|content-area|main-content", re.I)),
            soup.find("main"),
        ]

        for container in content_selectors:
            if container:
                # Remove script, style, nav elements
                for tag in container.find_all(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                content = container.get_text(separator="\n", strip=True)
                if len(content) > 200:  # Meaningful content
                    break

        # Fallback: get all paragraph text
        if len(content) < 200:
            paragraphs = soup.find_all("p")
            content = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

        if not content or len(content) < 100:
            print(f"No content found: {url}")
            return None

        # Clean up content
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        # Extract category from URL
        url_parts = url.split("/company/blog/")
        category = url_parts[1].split("/")[0] if len(url_parts) > 1 else "general"

        return {
            "title": title,
            "url": url,
            "description": description,
            "content": content,
            "category": category,
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def scrape_all_blogs():
    """Scrape all discovered blog posts."""

    urls = discover_blog_urls()

    blogs = []
    for i, url in enumerate(urls):
        print(f"  [{i+1}/{len(urls)}] {url.split('/')[-1][:50]}...")
        blog = scrape_blog(url)
        if blog:
            blogs.append(blog)
        time.sleep(REQUEST_DELAY)

    # Save blogs
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BLOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(blogs, f, indent=2, ensure_ascii=False)

    print(f" Scraped {len(blogs)} blogs → {BLOGS_FILE}")
    return blogs


''' Main '''

def main():
    #  Scrape
    if os.path.exists(BLOGS_FILE):
        print(f"Found existing blogs at {BLOGS_FILE}")
        with open(BLOGS_FILE, "r", encoding="utf-8") as f:
            blogs = json.load(f)
        print(f"  Loaded {len(blogs)} blogs")
        rescrape = input("  Re-scrape? (y/N): ").strip().lower()
        if rescrape == "y":
            blogs = scrape_all_blogs()
    else:
        blogs = scrape_all_blogs()

    if not blogs:
        print("No blogs scraped. Exiting.")
        return

    # Chunk
    chunks = process_blogs(blogs)

    # Index
    build_indices(chunks)

    print("\n" + "=" * 60)
    print(f" Indexed {len(chunks)} chunks from {len(blogs)} blogs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
