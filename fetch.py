from datetime import date, timedelta
import arxiv
import json

# Define the categories to search in arXiv
categories = ['cs.LG', 'cs.MM', 'cs.AI', 'cs.CV', 'cs.SD']

# Define today's and yesterday's dates
today = date.today()
interval = 4
hoshiidate = [today - timedelta(days=i) for i in range(interval)]
print(f'Crawling date {hoshiidate}')
dates = set()

# Function to fetch papers from arXiv
def fetch_papers(categories, max_results=200):
    papers = []
    for category in categories:
        query = f'cat:{category}'
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        for paper in arxiv.Client().results(search):
            paper_date = paper.published.date()
            if paper_date in hoshiidate:
                pdf_link = next((link.href for link in paper.links if link.title == 'pdf'), None)
                papers.append({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'date': paper.published.isoformat(),
                    'link': pdf_link  # Assuming you want the PDF link
                })
                dates.add(paper.published.date())
            else:
                break
    return papers

# Fetch recent papers
recent_papers = set(fetch_papers(categories))
print(f'Papers num: {len(recent_papers)}, Dates: {dates}')
with open(f'papers/papers{today}.json', 'w', encoding='utf-8') as file:
    json.dump(recent_papers, file, ensure_ascii=False, indent=4)

# markdown_output = ""
# for paper in recent_papers:
#     markdown_output += f"### {paper['title']}\n"
#     markdown_output += f"**Authors:** {', '.join(paper['authors'])}\n"
#     markdown_output += f"**Published Date:** {paper['date']}\n"
#     markdown_output += f"**Abstract:** {paper['abstract']}\n"
#     markdown_output += f"[PDF Link]({paper['link']})\n\n"

# with open('papers/arxiv_papers.md', 'w', encoding='utf-8') as file:
#     file.write(markdown_output)