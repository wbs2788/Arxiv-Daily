import json
import datetime
from datetime import date, timedelta
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from ttkbootstrap import Style
import requests
import os
import threading
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
from stopword import AI_STOPWORDS

# Define today's and yesterday's dates
today = date.today()
papers_filepath = f'papers/selected_papers_2023-12-13.json'  # Path of data file
selected_papers = []

with open(papers_filepath, 'r', encoding='utf-8') as file:
    papers = json.load(file)

root = tk.Tk()
root.title("Paper Filter")
root.geometry("1000x700")  
style = Style(theme='cyborg')
TOP6 = style.master

paper_index = [0]

remaining_label = tk.Label(root, text="", font=('Arial', 10))
remaining_label.pack()

def update_remaining_label():
    remaining_count = len(papers) - paper_index[0]
    remaining_label.config(text=f"Remaining: {remaining_count}/{len(papers)}")

def generate_wordcloud_from(filepath, key):
    with open(filepath, 'r', encoding='utf-8') as file:
        papers = json.load(file)
    all_abstracts = " ".join(paper[key] for paper in papers)
    STOPWORDS.update(AI_STOPWORDS)
    wordcloud = WordCloud(width=800, height=300, background_color ='white',\
                          stopwords=STOPWORDS).generate(all_abstracts)
    return wordcloud.to_image()



def show_paper():
    if paper_index[0] < len(papers):
        paper = papers[paper_index[0]]
        paper['abstract'] = paper['abstract'].replace('\n', ' ')
        text.delete('1.0', tk.END)

        text.tag_configure('header', font=('Arial', 14, 'bold'), spacing1=10, spacing3=10)
        text.tag_configure('subheader', font=('Arial', 12, 'bold'), spacing1=5, spacing3=5)
        text.tag_configure('content', font=('Arial', 12), spacing1=5, spacing3=5, lmargin1=20, \
                           lmargin2=20, rmargin=10)
        text.tag_configure('abstract', font=('Arial', 12), spacing1=5, spacing2=8, spacing3=5, \
                           lmargin1=20, lmargin2=20, rmargin=10)
        
        text.insert(tk.END, "Title:\n", 'header')
        text.insert(tk.END, f"{paper['title']}\n", 'content')
        text.insert(tk.END, "\nAuthors:\n", 'subheader')
        text.insert(tk.END, f"{', '.join(paper['authors'])}\n", 'content')
        text.insert(tk.END, "\nAbstract:\n", 'subheader')
        text.insert(tk.END, f"{paper['abstract']}\n", 'abstract')  # Use the 'abstract' tag for abstract content
        update_remaining_label()
    else:
        messagebox.showinfo("Mission Success", "All papers have been filtered")
        save_button.config(state=tk.NORMAL)

def save_markdown(paper):
    markdown_content = f"# {paper['title']}\n\n" \
                       f"## Authors\n{', '.join(paper['authors'])}\n\n" \
                       f"## Abstract\n{paper['abstract']}\n\n---\n\n"
    with open('favorites.md', 'a', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

def download_paper(paper, progress_bar):
    pdf_url = paper['link']
    pdf_url:str
    try:
        with requests.get(pdf_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            if total_size == 0:
                raise ValueError("Cannot get the content length of the file")
            downloaded = 0
            progress_bar['maximum'] = total_size
            if not os.path.exists(os.path.join('downloads', str(today))):
                os.mkdir(os.path.join('downloads', str(today)))
            sanitized_title = pdf_url.split('/')[-1]
            pdf_path = os.path.join('downloads', str(today), sanitized_title + '.pdf')
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            pdf_path = os.path.join('downloads', str(today), \
                                paper['title'].replace('/', '_') + '.pdf') 
            with open(pdf_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress_bar['value'] = downloaded
                    root.update_idletasks()
            messagebox.showinfo("Download Complete", "Download finished successfully!")
    except Exception as e:
        messagebox.showerror("Download Error", f"An error occurred: {e}")
    finally:
        progress_bar.pack_forget()

def start_download_thread(paper):
    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
    progress_bar.pack(pady=20)
    threading.Thread(target=download_paper, args=(paper, progress_bar)).start()

def collect_paper(event=None):
    if paper_index[0] < len(papers):
        paper = papers[paper_index[0]]
        save_markdown(paper)
        start_download_thread(paper) 

def keep_paper(event=None):
    if paper_index[0] < len(papers):
        selected_papers.append(papers[paper_index[0]])
        paper_index[0] += 1
        show_paper()

def skip_paper(event=None):
    if paper_index[0] < len(papers):
        paper_index[0] += 1
        show_paper()

def save_results():
    with open(f'{today}.json', 'w', encoding='utf-8') as file:
        json.dump(selected_papers, file, ensure_ascii=False, indent=4)
    messagebox.showinfo("save", "The result has been saved!")

def start_filtering():
    title_wordcloud_label.pack_forget()
    abstract_wordcloud_label.pack_forget()
    start_button.pack_forget()

    text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    button_frame.pack(pady=10, fill=tk.X)
    remaining_label.pack()
    show_paper()

text = scrolledtext.ScrolledText(root, height=30, width=120, wrap=tk.WORD, padx=10, pady=10)
text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=10, fill=tk.X)

keep_button = tk.Button(button_frame, text="Save (↑)", command=keep_paper, \
                        width=15, height=2, bg="lightgreen")
keep_button.pack(side=tk.LEFT, padx=10)

skip_button = tk.Button(button_frame, text="Skip (→)", command=skip_paper, \
                        width=15, height=2, bg="lightcoral")
skip_button.pack(side=tk.LEFT, padx=10)

save_button = tk.Button(button_frame, text="Get The Result", command=save_results, \
                        width=15, height=2, bg="lightblue", state=tk.DISABLED)
save_button.pack(side=tk.LEFT, padx=10)

collect_button = tk.Button(button_frame, text="Collect(↓)", command=collect_paper, \
                        width=15, height=2, bg="lightyellow")
collect_button.pack(side=tk.LEFT, padx=10)

root.bind('<Right>', skip_paper)
root.bind('<Up>', keep_paper)
root.bind('<Down>', collect_paper)

show_paper()

title_wordcloud_image = generate_wordcloud_from(papers_filepath, 'title')
title_image = ImageTk.PhotoImage(title_wordcloud_image)
title_wordcloud_label = tk.Label(root, image=title_image)
title_wordcloud_label.pack(pady=10)
title_wordcloud_label.image = title_image

abstract_wordcloud_image = generate_wordcloud_from(papers_filepath, 'abstract')
abstract_image = ImageTk.PhotoImage(abstract_wordcloud_image)
abstract_wordcloud_label = tk.Label(root, image=abstract_image)
abstract_wordcloud_label.pack(pady=10)
abstract_wordcloud_label.image = abstract_image

start_button = tk.Button(root, text="Start Filting", command=start_filtering)
start_button.pack(pady=20)

text.pack_forget()
button_frame.pack_forget()
remaining_label.pack_forget()

root.mainloop()