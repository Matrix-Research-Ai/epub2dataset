"""Test fixtures: generate small EPUBs for testing."""
import os
from ebooklib import epub

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def make_test_epub(chapters, filename):
    """Create a test EPUB with given chapters.
    
    Args:
        chapters: list of (title, file_id, html_content)
        filename: output filename in fixtures/
    """
    book = epub.EpubBook()
    book.set_identifier(f'test-{filename}')
    book.set_title(filename.replace('.epub', ''))
    book.set_language('en')
    
    items = []
    for title, fid, html in chapters:
        c = epub.EpubHtml(title=title, file_name=f'{fid}.xhtml')
        c.content = html
        book.add_item(c)
        items.append(c)
    
    book.toc = [epub.Link(f'{fid}.xhtml', title, fid) for title, fid, _ in chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav'] + items
    
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    out = os.path.join(FIXTURES_DIR, filename)
    epub.write_epub(out, book)
    return out


def generate_all():
    """Generate all test fixtures."""
    # Simple narrative EPUB
    make_test_epub([
        ("Chapter 1: History", "ch1", """<h1>History of Computing</h1>
        <p>Charles Babbage designed the Analytical Engine in 1837, 
        considered the first general-purpose computer concept. Ada Lovelace 
        wrote the first algorithm intended for machine processing.</p>
        <p>The ENIAC, completed in 1945, was the first electronic 
        general-purpose computer. It weighed 30 tons and performed 
        5,000 additions per second.</p>
        <p>The invention of the transistor in 1947 revolutionized computing. 
        Transistors replaced vacuum tubes, making computers smaller, faster, 
        and more reliable. This led to commercial computers in the 1950s.</p>"""),
        ("Chapter 2: Modern Era", "ch2", """<h1>The Modern Era</h1>
        <p>The Intel 4004, released in 1971, was the first commercially 
        available microprocessor. It contained 2,300 transistors and 
        operated at 740 kHz.</p>
        <p>The personal computer revolution began in the 1970s with machines 
        like the Altair 8800 and Apple II. The IBM PC, introduced in 1981, 
        established the standard for personal computing.</p>"""),
    ], "test_narrative.epub")

    # Technical content EPUB
    make_test_epub([
        ("Data Structures", "ds", """<h1>Data Structures in Python</h1>
        <p>Python provides several built-in data structures including lists, 
        dictionaries, sets, and tuples. Lists are ordered, mutable sequences 
        ideal for storing collections of items.</p>
        <p>The time complexity of common operations varies: list append is 
        O(1), dictionary lookup is O(1) average case, and list search is 
        O(n). Choosing the right data structure impacts performance.</p>"""),
    ], "test_technical.epub")

    # Dialogue EPUB
    make_test_epub([
        ("User Interaction", "dlg", """<h1>User Interaction</h1>
        <p>"Could you help me find information about machine learning?" 
        asked the user. "Certainly," replied the assistant. "Machine 
        learning is a subset of artificial intelligence that enables 
        systems to learn from data."</p>
        <p>"What are the main types?" the user continued. "The three main 
        types are supervised, unsupervised, and reinforcement learning," 
        explained the assistant.</p>"""),
    ], "test_dialogue.epub")

    # EPUB with front matter (should be filtered)
    make_test_epub([
        ("Title Page", "tp", "<h1>Test Book</h1><p>A Comprehensive Guide</p>"),
        ("Copyright", "cr", "<p>Copyright 2024. All rights reserved.</p>"),
        ("Table of Contents", "toc", "<h1>Contents</h1><p>Chapter 1... Chapter 2...</p>"),
        ("Chapter 1: Real Content", "ch1", """<h1>Real Content</h1>
        <p>This is the actual content of the book with meaningful information 
        that should appear in the training dataset. It contains enough text 
        to pass quality filtering based on multiple dimensions including 
        completeness, clarity, and coherence of the written content.</p>"""),
    ], "test_frontmatter.epub")

    print(f"All test fixtures generated in {FIXTURES_DIR}")


if __name__ == "__main__":
    generate_all()
