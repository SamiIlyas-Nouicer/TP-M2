import streamlit as st
import pandas as pd
import os
import nltk
from ast import literal_eval
import re

# Configure the page
st.set_page_config(page_title="Descriptor and Token Files", layout="centered")

# File configurations
csv_files = ["Split.csv", "SplitLancaster.csv", "SplitPorter.csv",
             "Token.csv", "TokenLancaster.csv", "TokenPorter.csv"]
display_names = [os.path.splitext(file)[0] for file in csv_files]
collection_folder = "Collection"  # Folder containing D1, D2, ..., D6 text files

# Set main title with styling
st.title("📑 Descriptor and Token Files")

# File selection dropdown
selected_display_name = st.selectbox(
    "Select a CSV file to explore:", display_names)
selected_file = selected_display_name + ".csv"

# Function to load CSV data


def load_data(file_path):
    return pd.read_csv(file_path)


# Display selected CSV data and perform operations
if selected_file:
    file_path = os.path.join("test", selected_file)
    df = load_data(file_path)

    # Sidebar for Document ID search
    st.sidebar.subheader("🔍 Search Document ID")
    query = st.sidebar.text_input("Enter Document ID:")

    # Filter DataFrame based on Document ID
    if query:
        try:
            query = int(query)
            df = df[df["Document"] == query].reset_index(drop=True)
        except ValueError:
            st.sidebar.warning("Please enter a valid Document ID (numeric).")

    # Section to view CSV file contents
    st.subheader(f"📂 Contents of {selected_display_name}")
    st.dataframe(df)

    # Document Information Section
    st.subheader("📊 Document Statistics")
    vocabulary_size = df['Token'].nunique()  # Unique tokens count
    document_size = df['Frequency'].sum()  # Total frequency across all tokens
    st.write(f"- **Vocabulary Size (Unique Tokens):** {vocabulary_size}")
    st.write(f"- **Total Frequency (Document Size):** {document_size}")

    # Word search with stemming based on file type
    st.subheader("🔍 Search for a Word")
    search_word = st.text_input("Enter word to search:")

    if search_word:
        stemmers = {
            "SplitPorter": nltk.PorterStemmer(),
            "SplitLancaster": nltk.LancasterStemmer(),
            "TokenPorter": nltk.PorterStemmer(),
            "TokenLancaster": nltk.LancasterStemmer(),
        }
        stemmer = stemmers.get(selected_display_name)
        if stemmer:
            search_word = stemmer.stem(search_word)

        # Filter DataFrame by search term in Token column
        filtered_df = df[df['Token'].str.contains(
            search_word, case=False, na=False)]
        st.write(f"**Results for '{search_word}':**")
        st.dataframe(filtered_df)

        # Text File Viewer for selected Document ID
        if query:
            try:
                doc_num = int(query)
                if 1 <= doc_num <= 6:  # Ensure document number is within range
                    file_name = f"D{doc_num}.txt"
                    file_path = os.path.join(collection_folder, file_name)

                    if os.path.exists(file_path):
                        with open(file_path, "r") as file:
                            file_content = file.read()

                        # Loop through filtered_df to highlight occurrences in the text
                        highlighted_content = file_content  # Start with full content
                        for index, row in filtered_df.iterrows():
                            token = row['Token']
                            occurrences = row['Occurrence']

                            # Ensure occurrences are a valid list
                            positions = []
                            try:
                                positions = literal_eval(occurrences)
                                if not isinstance(positions, list):
                                    raise ValueError(
                                        "Occurrences data is not a list.")
                                positions = [int(pos) for pos in positions]
                            except (ValueError, SyntaxError) as e:
                                # st.warning(f"Occurrences for '{
                                #            token}' are not formatted correctly.")
                                continue  # Skip this token if it's not valid

                            # Highlight positions in the text for this token
                            for pos in positions:
                                # Create a regex to match the token at the correct position
                                # We highlight only whole word matches at exact positions
                                regex_pattern = r"\b" + \
                                    re.escape(token) + r"\b"
                                match = list(re.finditer(
                                    regex_pattern, file_content))
                                for m in match:
                                    if m.start() == pos:
                                        # Highlight the word occurrence
                                        highlighted_content = highlighted_content.replace(
                                            m.group(0),
                                            f"""<span style='color: red; font-weight:bold;'>{
                                                m.group(0)}</span>""",
                                            1)  # Replace only the first occurrence

                        # Display the content with highlighted positions
                        st.write(
                            f"**Contents of Document {doc_num} for Token '{search_word}':**")
                        st.markdown(
                            f"<div style='white-space: pre-wrap; border: none;padding:15px ; border-radius: 8px ;background-color: #262730;'>{highlighted_content}</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"File {file_name} not found.")
                else:
                    st.warning(
                        "Please enter a document number between 1 and 6.")
            except ValueError:
                st.warning("Please enter a valid document number.")
