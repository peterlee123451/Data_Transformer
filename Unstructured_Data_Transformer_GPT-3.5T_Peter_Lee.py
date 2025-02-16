import os
import openai
openai.api_key = os.getenv("FILL HERE")

import streamlit as st
import openai
import pandas as pd
import json
import itertools


st.title("Structured Data to One-hot encoded Data Transformer for Insurance applications")


api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Upload a text file with unstructured data", type=["txt", "csv", "pdf"])

if st.button("Run Extraction"):

    # Check if we have both the API key and a file
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.error("Please upload a file.")
    else:
        # 2. Set the OpenAI API key
        openai.api_key = api_key

        # 3. Read the uploaded file lines
        file_bytes = uploaded_file.read()
        text_data = file_bytes.decode("utf-8", errors="replace")  # decode file content
        lines = text_data.splitlines()  # each line is considered a separate note

        # 4. Define a helper function to build the prompt
        def build_prompt(unstructured_text):
            prompt = f"""
            You are a helpful assistant that extracts structured data from text.
            The text below is a medical/insurance note. Please return a JSON object with 
            the following fields:
            - patient_name (string)
            - patient_id (string)
            - date_of_birth (string)
            - diagnosis (string)
            - symptoms (List of strings)
            - claim_date(string)
            - procedures (list of strings)
            - claim_amount (float)
            - Disabled (boolean)
            - Death (boolean)
            - additional_notes (string)

            If any field is missing, use an empty string or empty list.

            Text to parse:
            \"\"\"{unstructured_text}\"\"\"

            Return only valid JSON (and nothing else).
            """
            return prompt
        # 5. Function that calls OpenAI to extract structured data
        def extract_data_from_text(unstructured_text):
            prompt = build_prompt(unstructured_text)

            response = openai.Completion.create(
                engine="gpt-3.5-turbo",  # or GPT-4, more expensive.
                prompt=prompt,
                max_tokens=200,
                temperature=0
            )
            raw_output = response["choices"][0]["text"].strip()
            try:
                structured_data = json.loads(raw_output)
            except json.JSONDecodeError:
                # If there's a parsing error, you could set default or handle differently
                structured_data = {
                    "patient_name": "",
                    "patient_id": "",
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

        # 6. Loop through each line of text, extract data
        all_structured = []
        with st.spinner("Extracting data..."):
            for line in lines:
                line = line.strip()
                if line:
                    result = extract_data_from_text(line)
                    all_structured.append(result)

        # 7. Convert to DataFrame
        df = pd.DataFrame(all_structured)

        if df.empty:
            st.warning("No structured data found, or file was empty.")
        else:
            st.subheader("Raw Extracted Data")
            st.dataframe(df)

            # 8. One-hot encode the "diagnosis" column (if it exists)
            if "diagnosis" in df.columns:
                df = pd.get_dummies(df, columns=["diagnosis"], prefix="diag")
            
            # 9. Expand multi-label "procedures" into multiple columns
            if "procedures" in df.columns:
                all_procedures = set(itertools.chain.from_iterable(df['procedures'].dropna()))
                for proc in all_procedures:
                    df[f"procedure_{proc}"] = df['procedures'].apply(lambda x: 1 if x and proc in x else 0)
                # Optionally drop the original procedures column
                df.drop(columns=["procedures"], inplace=True)

            st.subheader("One-Hot Encoded Data")
            st.dataframe(df)

            # 10. Download button for the final DataFrame as CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="extracted_data.csv",
                mime="text/csv"
            )



