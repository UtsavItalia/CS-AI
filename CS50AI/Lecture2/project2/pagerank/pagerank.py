import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}
    base_probability = (1 - damping_factor) / len(corpus)
    for filename in corpus:
        if filename in corpus[page]:
            probability_distribution[filename] = base_probability + (
                damping_factor / len(corpus[page])
            )
        else:
            probability_distribution[filename] = base_probability
    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_visits = {filename: 0 for filename in corpus}
    current_page = random.choice(list(corpus.keys()))

    for i in range(n):
        page_visits[current_page] += 1
        probability_distribution = transition_model(
            corpus, current_page, damping_factor
        )
        pages = list(probability_distribution.keys())
        weights = list(probability_distribution.values())
        current_page = random.choices(pages, weights=weights, k=1)[0]

    return {key: value / n for key, value in page_visits.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    for page in corpus:
        if len(corpus[page]) == 0:
            corpus[page] = set(corpus.keys())

    pageranks = {key: 1 / len(corpus) for key in corpus.keys()}
    # print(base_distribution)

    while True:
        new_pageranks = {}
        for page in corpus:
            new_pageranks[page] = (1 - damping_factor) / len(corpus)
            for linking_page in corpus:
                if page in corpus[linking_page]:
                    new_pageranks[page] += damping_factor * (
                        pageranks[linking_page] / len(corpus[linking_page])
                    )
        if all(abs(new_pageranks[page] - pageranks[page]) < 0.001 for page in corpus):
            break
        pageranks = new_pageranks

    return pageranks


if __name__ == "__main__":
    main()
