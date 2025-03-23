import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import json
import itertools
import openai

# Define the function to build the prompt.
def build_prompt(unstructured_text: str) -> str:
    prompt = f"""
    You will extract structured data from text.
    The text below is a medical/insurance note. Please return a JSON object with
    the following fields filled from the text provided:
    - patient_name (string)
    - patient_id (string)
    - patient_gender (string)
    - date_of_birth (string)
    - diagnosis (string)
    - symptoms (List of strings)
    - claim_date (string)
    - procedures (list of strings)
    - claim_amount (float)
    - Disabled (boolean)
    - Death (boolean)
    - additional_notes (string)

    Patient Name might be in the format: "Patient name: <NAME>"
    Date of Birth might appear as "Patient Date of Birth: <DOB>"
    Diagnosis might appear as "Diagnosis: <DIAGNOSIS>"

    If any field is missing, use an empty string or empty list.

    Text to parse:
    \"\"\"{unstructured_text}\"\"\"

    Return only valid JSON (and nothing else).
    """
    return prompt

# Define a function to call the OpenAI API and extract structured data.
def extract_data_from_text(unstructured_text: str, api_key: str) -> dict:
    openai.api_key = api_key
    prompt = build_prompt(unstructured_text)

    try:
        response = openai.chat.completions.create(
            model="gpt-4",  # or your desired model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2
        )
        raw_output = response.choices[0].message.content.strip()
        structured_data = json.loads(raw_output)
    except Exception as e:
        messagebox.showerror("Extraction Error", f"Error during extraction: {e}")
        structured_data = {
            "patient_name": "",
            "patient_id": "",
            "patient_gender": "",
            "date_of_birth": "",
            "diagnosis": "",
            "symptoms": [],
            "claim_date": "",
            "procedures": [],
            "claim_amount": 0.0,
            "Disabled": False,
            "Death": False,
            "additional_notes": ""
        }
    return structured_data

# Define the function to process the file and update the UI.
def run_extraction():
    api_key = api_entry.get().strip()
    if not api_key:
        messagebox.showerror("Missing API Key", "Please enter your OpenAI API key.")
        return
    if not file_path.get():
        messagebox.showerror("Missing File", "Please select a file to upload.")
        return

    try:
        with open(file_path.get(), "rb") as f:
            file_bytes = f.read()
            text_data = file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        messagebox.showerror("File Error", f"Error reading the file: {e}")
        return

    # Extract data using OpenAI API
    result = extract_data_from_text(text_data, api_key)
    df_raw = pd.DataFrame([result])

    # Clear and display the raw extracted data in the text widget.
    raw_text.delete(1.0, tk.END)
    raw_text.insert(tk.END, df_raw.to_string(index=False))

    # One-hot encode the "diagnosis" column if it exists.
    df_final = df_raw.copy()
    if "diagnosis" in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=["diagnosis"], prefix="diag")

    # Expand multi-label "procedures" into multiple columns.
    if "procedures" in df_final.columns:
        # Ensure the procedures column is a list for each row.
        # Here we assume that if procedures are provided, they are in list format.
        try:
            all_procs = set(itertools.chain.from_iterable(df_final['procedures'].dropna()))
        except Exception:
            all_procs = set()
        for proc in all_procs:
            df_final[f"procedure_{proc}"] = df_final['procedures'].apply(
                lambda x: 1 if x and proc in x else 0
            )
        df_final.drop(columns=["procedures"], inplace=True)

    # Display the one-hot encoded data.
    encoded_text.delete(1.0, tk.END)
    encoded_text.insert(tk.END, df_final.to_string(index=False))

    # Store final dataframe globally for saving
    global final_df
    final_df = df_final

def select_file():
    selected_file = filedialog.askopenfilename(
        title="Select a text file",
        filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("PDF files", "*.pdf"), ("All Files", "*.*")]
    )
    if selected_file:
        file_path.set(selected_file)
        file_label.config(text=selected_file)

def save_csv():
    if final_df is None or final_df.empty:
        messagebox.showwarning("No Data", "No data available to save. Please run the extraction first.")
        return
    save_file = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save CSV file"
    )
    if save_file:
        try:
            final_df.to_csv(save_file, index=False)
            messagebox.showinfo("Success", f"CSV file saved to {save_file}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving CSV file: {e}")

# Initialize global variable for final DataFrame.
final_df = pd.DataFrame()

# Set up the Tkinter GUI.
root = tk.Tk()
root.title("Structured Data to One-Hot Encoded Data Transformer for Insurance Applications")
root.geometry("800x700")

# API key entry
api_label = tk.Label(root, text="Enter your OpenAI API Key:")
api_label.pack(pady=(10, 0))
api_entry = tk.Entry(root, width=50, show="*")
api_entry.pack(pady=(0, 10))

# File selection
file_path = tk.StringVar()
file_btn = tk.Button(root, text="Select File", command=select_file)
file_btn.pack(pady=(10, 0))
file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=(0, 10))

# Run extraction button
run_btn = tk.Button(root, text="Run Extraction", command=run_extraction)
run_btn.pack(pady=(10, 10))

# Raw extracted data display
raw_label = tk.Label(root, text="Raw Extracted Data:")
raw_label.pack()
raw_text = scrolledtext.ScrolledText(root, height=10, width=100)
raw_text.pack(pady=(0, 10))

# One-hot encoded data display
encoded_label = tk.Label(root, text="One-Hot Encoded Data:")
encoded_label.pack()
encoded_text = scrolledtext.ScrolledText(root, height=10, width=100)
encoded_text.pack(pady=(0, 10))

# Save CSV button
save_btn = tk.Button(root, text="Save as CSV", command=save_csv)
save_btn.pack(pady=(10, 10))

# Optional: Link to LinkedIn (opens in browser)
def open_linkedin():
    import webbrowser
    webbrowser.open("https://www.linkedin.com/in/peter-lee-902920231/")

linkedin_btn = tk.Button(root, text="Visit my LinkedIn!!!", bg="lightblue", command=open_linkedin)
linkedin_btn.pack(pady=(10, 20))

root.mainloop()