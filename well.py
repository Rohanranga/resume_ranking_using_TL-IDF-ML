import os
import re
import pdfplumber
import docx
import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

# Define regex patterns
PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3})?\s?(?:\d{6,12})\b')
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# ML Model Training (Interactive training with folder input)
def train_model():
    # Ask for the folder location containing resumes
    folder_path = filedialog.askdirectory(title="Select Folder Containing Resumes")
    if not folder_path:
        messagebox.showwarning("No Folder Selected", "Please select a folder containing resumes.")
        return
    
    # Initialize lists to store training data
    resumes = []
    experiences = []
    
    # Process each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        text = ""
        
        # Extract text based on file type
        if filename.endswith(".pdf"):
            text, _, _ = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text, _, _ = extract_text_from_docx(file_path)
        elif filename.endswith(".txt"):
            text, _, _ = extract_text_from_txt(file_path)
        
        if text:
            # Prompt the user to input experience for the file
            experience = simpledialog.askfloat(
                title="Input Experience",
                prompt=f"Enter years of experience for '{filename}':",
                minvalue=0.0,
                maxvalue=50.0
            )
            
            if experience is not None:  # If the user provides input
                resumes.append(text)
                experiences.append(experience)
    
    # Check if any data was collected
    if not resumes:
        messagebox.showwarning("No Data", "No valid resumes or experience inputs were provided.")
        return
    
    # Train the model
    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(resumes)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, experiences)
        
        # Save the model and vectorizer
        joblib.dump(model, "experience_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        
        # Save the training data to a CSV file
        training_data = pd.DataFrame({"Resume": resumes, "Experience": experiences})
        training_data.to_csv("training_data.csv", index=False)
        
        messagebox.showinfo("Success", "Model trained and data saved to 'training_data.csv'!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while training the model: {e}")

# Extract text from different file types
def extract_text_from_pdf(pdf_path):
    text = ""
    links = []
    
    try:
        import fitz  
        doc = fitz.open(pdf_path)
        
        for page in doc:
            text += page.get_text()
            for link in page.get_links():
                if "uri" in link:
                    links.append(link["uri"])

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    linkedin_links = list(filter(lambda x: "linkedin.com" in x, links or [])) if links else []
    github_links = list(filter(lambda x: "github.com" in x, links or [])) if links else []

    return text, linkedin_links, github_links

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip(), [], []

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read().strip(), [], []

# Predict experience using ML model
def predict_experience(text):
    try:
        model = joblib.load("experience_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        X = vectorizer.transform([text])
        experience = model.predict(X)[0]
        return f"{round(experience, 1)} years"
    except Exception as e:
        print("ML Model Error:", e)
        return "-"

# Process resumes in the selected folder
def process_resumes(folder_path, keyword):
    data = []
    
    for rank, filename in enumerate(os.listdir(folder_path), start=1):
        file_path = os.path.join(folder_path, filename)
        text = ""
        linkedin_links, github_links = [], []

        if filename.endswith(".pdf"):
            text, linkedin_links, github_links = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text, linkedin_links, github_links = extract_text_from_docx(file_path)
        elif filename.endswith(".txt"):
            text, linkedin_links, github_links = extract_text_from_txt(file_path)

        if text:
            phone_numbers = ", ".join(set(PHONE_PATTERN.findall(text))) or "-"
            emails = ", ".join(set(EMAIL_PATTERN.findall(text))) or "-"
            experience = predict_experience(text)  

            keyword_count = text.lower().count(keyword.lower()) if keyword else 0
            score = keyword_count * 10  # Example scoring system (adjust as needed)

            data.append({
                "Rank": rank,
                "File Name": filename,
                "Phone Numbers": phone_numbers,
                "Emails": emails,
                "LinkedIn": linkedin_links[0] if linkedin_links else "-",
                "GitHub": github_links[0] if github_links else "-",
                "Keyword Count": keyword_count,
                "Score": score,
                "Experience": experience
            })

    return data

# Select folder and display results
def select_folder():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return
    
    keyword = keyword_entry.get()
    results = process_resumes(folder_path, keyword)

    for row in table.get_children():
        table.delete(row)

    for entry in results:
        table.insert("", "end", values=(
            entry["Rank"], entry["File Name"], entry["Phone Numbers"],
            entry["Emails"], entry["LinkedIn"], entry["GitHub"],
            entry["Keyword Count"], entry["Score"], entry["Experience"]
        ))

    df = pd.DataFrame(results)
    df.to_csv("resume_data_ml.csv", index=False)
    df.to_excel("resume_data_ml.xlsx", index=False)

    messagebox.showinfo("Success", "Resumes processed successfully!\nData saved as CSV and Excel.")

# Open links
def open_link(event):
    selected_item = table.selection()
    if selected_item:
        col_id = table.identify_column(event.x)
        col_num = int(col_id[1:]) - 1
        item_data = table.item(selected_item)['values']
        
        if col_num in [4, 5]:  
            url = item_data[col_num]
            if url.startswith("http"):
                webbrowser.open(url)

# GUI Setup
root = tk.Tk()
root.title("ML Resume Parser")
root.geometry("1100x550")

frame = tk.Frame(root)
frame.pack(pady=10)

btn_train_model = tk.Button(frame, text="Train Model", command=train_model, font=("Arial", 12), bg="green", fg="white")
btn_train_model.pack(side="left", padx=10)

btn_select_folder = tk.Button(frame, text="Select Resume Folder", command=select_folder, font=("Arial", 12), bg="blue", fg="white")
btn_select_folder.pack(side="left", padx=10)

# Keyword Input
tk.Label(frame, text="Enter Keyword:", font=("Arial", 12)).pack(side="left", padx=10)
keyword_entry = tk.Entry(frame, font=("Arial", 12), width=15)
keyword_entry.pack(side="left")

columns = ("Rank", "File Name", "Phone Numbers", "Emails", "LinkedIn", "GitHub", "Keyword Count", "Score", "Experience")
table = ttk.Treeview(root, columns=columns, show="headings")

for col in columns:
    table.heading(col, text=col)
    table.column(col, width=120)

table.pack(fill="both", expand=True)

table.bind("<Double-1>", open_link)

root.mainloop()