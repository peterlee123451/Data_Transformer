import os
import streamlit as st
import pandas as pd
import json
import itertools

# Import the new openai library:
import openai

#API KEY/ LOGIN
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_HERE")

st.title("Structured Data to One-hot Encoded Data Transformer for Insurance Applications")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Upload a text file with unstructured data", type=["txt", "csv", "pdf"])



#Select Variables to extract

st.subheader("Select Variables for extraction")
variables = ["Patient Name", "Patient ID", "Patient Gender",
             "Date of Birth", "Diagnosis", "Symptoms",
             "Claim Date", "Procedures", "Claim Amount",
             "Disabled", "Death", "Additional Notes"]



selected_variables = st.multiselect("Select the Variables to extract", variables)


if st.button("Run Extraction"):

    # 1. Verify we have both the API key and a file
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.error("Please upload a file.")
    else:
        # 2. Set the OpenAI API key at runtime
        openai.api_key = api_key

        # 3. Read the entire uploaded file as one text block
        file_bytes = uploaded_file.read()
        text_data = file_bytes.decode("utf-8", errors="replace")

        # 4. Define a helper function to build the prompt
        def build_prompt(unstructured_text: str) -> str:
            prompt = f"""
            You will extract structured data from text.
            The text below is a medical/insurance note. Please return a JSON object with
            the following fields filled from the text provided:
            {selected_variables}
            Patient Name might be in the format: "Patient name: <NAME>"
            Date of Birth might appear as "Patient Date of Birth: <DOB>"
            Diagnosis might appear as "Diagnosis: <DIAGNOSIS>"

            If any field is missing, use an empty string or empty list.

            Text to parse:
            \"\"\"{unstructured_text}\"\"\"


            Return only valid JSON (and nothing else).
            """
            return prompt

        # 5. Function that calls OpenAI to extract structured data
        def extract_data_from_text(unstructured_text: str) -> dict:
            prompt = build_prompt(unstructured_text)

            # Using the new module-level client interface:
            response = openai.chat.completions.create(
                model="gpt-4",  # Update to your deployed/fine-tuned model if needed
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.2
            )

            # The response is a Completion object; parse out the message content
            raw_output = response.choices[0].message.content.strip()
            try:
                structured_data = json.loads(raw_output)
            except json.JSONDecodeError:
                # If there's a parsing error, return empty defaults
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

        # 6. Parse the entire text_data just once
        with st.spinner("Extracting data..."):
            result = extract_data_from_text(text_data)
            df = pd.DataFrame([result])

        # 7. Display the extracted data
        if df.empty:
            st.warning("No structured data found, or the file was empty.")
        else:
            st.subheader("Raw Extracted Data")
            st.dataframe(df)


            # 8. Download button for the final DataFrame as CSV
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="extracted_data.csv",
                mime="text/csv"
            )

st.markdown(
    """
    <a href="https://www.linkedin.com/in/peter-lee-902920231/" target="_blank">
        <button style="background-color:LightBlue; border:none; padding:10px 20px; cursor:pointer;">
            Visit my LinkedIn!!!
        </button>
    </a>
    """,
    unsafe_allow_html=True
)