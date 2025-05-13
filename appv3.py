import os, io, json, pytesseract, openai
import streamlit as st
import pandas as pd
from pdf2image import convert_from_bytes
import streamlit.components.v1 as components

st.title("Data Transformer")

# initial API key (fallback)
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_HERE")

# allow multiple file uploads
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
uploaded_files = st.file_uploader("Upload text files or PDFs", type=["txt", "csv", "pdf"], accept_multiple_files=True)

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

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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
    elif not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        openai.api_key = api_key
        all_results = []

        for uploaded_file in uploaded_files:
            name, ext = os.path.splitext(uploaded_file.name)
            ext = ext.lower()

            # read & OCR if PDF
            if ext == ".pdf":
                pages = convert_from_bytes(uploaded_file.read(), dpi=300)
                texts = []
                for page in pages:
                    gray = page.convert("L")
                    bw = gray.point(lambda x: 0 if x < 128 else 255, "1")
                    texts.append(pytesseract.image_to_string(bw, lang="eng", config="--psm 6"))
                raw_bytes = "\n".join(texts).encode("utf-8")
            else:
                raw_bytes = uploaded_file.read()

            text_data = raw_bytes.decode("utf-8", errors="replace")
            with st.spinner(f"Extracting from {uploaded_file.name}..."):
                result = extract_data_from_text(text_data)
                result["source_file"] = uploaded_file.name
                all_results.append(result)

        # compile into DataFrame
        df = pd.DataFrame(all_results)

        if df.empty:
            st.warning("No structured data found.")
        else:
            st.subheader("Raw Extracted Data")
            st.dataframe(df)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download all as CSV",
                data=csv_data,
                file_name="extracted_data_batch.csv",
                mime="text/csv"
            )
