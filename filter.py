import json
from datetime import date
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

# Define today's and yesterday's dates
today = date.today()
papers_filepath = f'papers/papers{today}.json'  # 您的论文数据文件路径

# 读取论文数据
with open(papers_filepath, 'r', encoding='utf-8') as file:
    papers = json.load(file)

# 初始化Tkinter窗口
root = tk.Tk()
root.title("论文筛选器")
root.geometry("1000x700")  # 设置窗口尺寸

# 设置论文展示的索引
paper_index = [0]

# 显示剩余论文数量的标签
remaining_label = tk.Label(root, text="", font=('Arial', 10))
remaining_label.pack()

# 更新显示剩余论文数量的函数
def update_remaining_label():
    remaining_count = len(papers) - paper_index[0]
    remaining_label.config(text=f"剩余论文数量: {remaining_count}")
    
# 筛选后的论文列表
selected_papers = []
# 显示论文信息的函数
def show_paper():
    if paper_index[0] < len(papers):
        paper = papers[paper_index[0]]
        paper['abstract'] = paper['abstract'].replace('\n', '')
        text.delete('1.0', tk.END)
        text.tag_configure('header', font=('Arial', 14, 'bold'), spacing1=10, spacing3=10)
        text.tag_configure('subheader', font=('Arial', 12, 'bold'), spacing1=5, spacing3=5)
        text.tag_configure('content', font=('Arial', 12), spacing1=5, spacing3=5, lmargin1=20, lmargin2=20, rmargin=10)
        text.tag_configure('abstract', font=('Arial', 12), spacing1=5, spacing2=8, spacing3=5, lmargin1=20, lmargin2=20, rmargin=10)
        text.insert(tk.END, "Title:\n", 'header')
        text.insert(tk.END, f"{paper['title']}\n", 'content')
        text.insert(tk.END, "\nAuthors:\n", 'subheader')
        text.insert(tk.END, f"{', '.join(paper['authors'])}\n", 'content')
        text.insert(tk.END, "\nAbstract:\n", 'subheader')
        text.insert(tk.END, f"{paper['abstract']}\n", 'abstract')  # Use the 'abstract' tag for abstract content
        update_remaining_label()
    else:
        messagebox.showinfo("完成", "所有论文已筛选完毕！")
        save_button.config(state=tk.NORMAL)

# 保留论文的函数
def keep_paper(event=None):
    if paper_index[0] < len(papers):
        selected_papers.append(papers[paper_index[0]])
        paper_index[0] += 1
        show_paper()

# 跳过论文的函数
def skip_paper(event=None):
    if paper_index[0] < len(papers):
        paper_index[0] += 1
        show_paper()

# 保存筛选结果的函数
def save_results():
    with open('selected_papers.json', 'w', encoding='utf-8') as file:
        json.dump(selected_papers, file, ensure_ascii=False, indent=4)
    messagebox.showinfo("保存", "筛选结果已保存！")

# 设置UI布局
text = scrolledtext.ScrolledText(root, height=30, width=120, wrap=tk.WORD, padx=10, pady=10)
text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=10, fill=tk.X)

keep_button = tk.Button(button_frame, text="保留 (↑)", command=keep_paper, width=15, height=2, bg="lightgreen")
keep_button.pack(side=tk.LEFT, padx=10)

skip_button = tk.Button(button_frame, text="跳过 (→)", command=skip_paper, width=15, height=2, bg="lightcoral")
skip_button.pack(side=tk.LEFT, padx=10)

save_button = tk.Button(button_frame, text="保存结果", command=save_results, width=15, height=2, bg="lightblue", state=tk.DISABLED)
save_button.pack(side=tk.LEFT, padx=10)

# 绑定键盘快捷键
root.bind('<Right>', skip_paper)
root.bind('<Up>', keep_paper)

# 显示第一篇论文
show_paper()

# 运行Tkinter事件循环
root.mainloop()