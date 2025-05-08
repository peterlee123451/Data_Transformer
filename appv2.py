import os
import io
import json
import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import openai
import streamlit.components.v1 as components

st.title("Structured Data to One-hot Encoded Data Transformer for Insurance Applications")

# initial API key (fallback)
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_HERE")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")
uploaded_file = st.file_uploader("Upload a text file or PDF", type=["txt", "csv", "pdf"])

variables = [
    "Patient Name", "Patient ID", "Patient Gender",
    "Date of Birth", "Diagnosis", "Symptoms",
    "Claim Date", "Procedures", "Claim Amount",
    "Disabled", "Death", "Additional Notes"
]
selected_variables = st.multiselect("Select the Variables to extract", variables)

def build_prompt(text: str) -> str:
    return f"""
You will extract structured data from text.
The text below is a medical/insurance note. Please return a JSON object with
the following fields filled from the text provided:
{selected_variables}

If any field is missing, use an empty string or empty list.

Text to parse:
\"\"\"{text}\"\"\"

Return only valid JSON.
"""

def extract_data_from_text(unstructured_text: str) -> dict:
    prompt = build_prompt(unstructured_text)

    # Using the original chat completion call
    response = openai.chat.completions.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."}, #persona <- helpful assistant
            {"role": "user", "content": prompt} 
        ],
        max_tokens=512,
        temperature=0.2
    )

    raw_output = response.choices[0].message.content.strip()
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # fallback empty structure
        return {v.lower().replace(" ", "_"): "" for v in selected_variables}

if st.button("Run Extraction"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.error("Please upload a file.")
    else:
        openai.api_key = api_key

        # determine file extension
        name, ext = os.path.splitext(uploaded_file.name)
        ext = ext.lower()

        # read & OCR if PDF
        if ext == ".pdf":
            pages = convert_from_bytes(uploaded_file.read(), dpi=300)
            texts = []
            for i, page in enumerate(pages, 1):
                gray = page.convert("L")
                bw   = gray.point(lambda x: 0 if x < 128 else 255, "1")
                texts.append(pytesseract.image_to_string(bw, lang="eng", config="--psm 6"))
            raw_bytes = "\n".join(texts).encode("utf-8")
        else:
            raw_bytes = uploaded_file.read()

        text_data = raw_bytes.decode("utf-8", errors="replace")

        with st.spinner("Extracting data..."):
            result = extract_data_from_text(text_data)

        df = pd.DataFrame([result])

        if df.empty:
            st.warning("No structured data found, or the file was empty.")
        else:
            st.subheader("Raw Extracted Data")
            st.dataframe(df)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="extracted_data.csv",
                mime="text/csv"
            )

def redirect_to(url: str):
    js = f"<script>window.open('{url}');</script>"
    components.html(js)

if st.button("Check out my LinkedIn!"):
    redirect_to("https://www.linkedin.com/in/peter-lee-902920231/")
