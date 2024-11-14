import streamlit as st
import pandas as pd
import altair as alt
import os
import nltk
from ast import literal_eval
import re

# Configure the page
st.set_page_config(page_title="Descriptor and Token Files", layout="centered")

# Apply custom CSS for a sophisticated dark theme with padding
st.markdown("""
    <style>
        /* General page style */
        body {
            background-color: purple;
            color: #6f4ac7;
            font-family: Arial, sans-serif;
            padding: 30px;
        }

        /* Title and Header Styling */
        .title-style {
            color: #6f4ac7;
            font-size: 2.6em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            padding: 10px;
        }

        h2, h3 {
            color: #6f4ac7;
            border-bottom: 2px solid #30363D;
            padding-bottom: 8px;
            margin-top: 30px;
            padding-left: 10px;
        }

        /* Dropdown and input styling */
        .stSelectbox, .stTextInput, .stButton > button {
            background-color: #30363D !important;
            color: #6f4ac7 !important;
            border-radius: 6px;
            border: 1px solid #6f4ac7;
            padding: 10px 15px;
        }

        /* Sidebar styling */
        .stSidebar h2 {
            color: #6f4ac7;
            padding: 15px;
        }

        /* Dataframe styling */
        .stDataFrame {
            background-color: #22272E;
            border: 1px solid #30363D;
            border-radius: 8px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
            padding: 15px;
            margin-top: 25px;
        }

        /* Document viewer styling */
        .document-viewer {
            background-color: #22272E;
            padding: 20px;
            color: #6f4ac7;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
            font-size: 1.1em;
            line-height: 1.6;
            padding-left: 20px;
            padding-right: 20px;
        }

        /* Highlighted token styling */
        .highlighted-token {
            background-color: #FF6B6B;
            color: #121212;
            padding: 0 4px;
            border-radius: 4px;
            font-weight: bold;
        }

        /* Input field and button padding */
        .stTextInput, .stButton > button {
            padding: 15px;
        }

        /* Spacing adjustments for overall layout */
        .streamlit-container {
            padding: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title with new style
st.markdown('<div class="title-style">📑 Descriptor and Token Files</div>',
            unsafe_allow_html=True)

# File configurations
csv_files = ["Split.csv", "SplitLancaster.csv", "SplitPorter.csv",
             "Token.csv", "TokenLancaster.csv", "TokenPorter.csv"]
display_names = [os.path.splitext(file)[0] for file in csv_files]
collection_folder = "Collection"  # Folder containing D1, D2, ..., D6 text files

# File selection dropdown
selected_display_name = st.selectbox("Choose a CSV File:", display_names)
selected_file = selected_display_name + ".csv"

# Function to load CSV data


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


# Display selected CSV data and perform operations
if selected_file:
    file_path = os.path.join("test", selected_file)
    df = load_data(file_path)

    # Sidebar for Document ID search
    st.sidebar.subheader("🔍 Search by Document ID")
    query = st.sidebar.text_input("Enter Document ID:")

    # Filter DataFrame based on Document ID
    if query:
        try:
            query = int(query)
            if query < 1 or query > 6:
                st.sidebar.warning("Document ID must be between 1 and 6.")
            else:
                df = df[df["Document"] == query].reset_index(drop=True)
        except ValueError:
            st.sidebar.warning("Please enter a valid numeric Document ID.")

    # Section to view CSV file contents
    st.subheader(f"📂 Contents of {selected_display_name}")
    st.dataframe(df)

    # Document Information Section
    # Document Statistics Calculation
    st.subheader("📊 Document Statistics")

    # Vocabulary and document size calculations
    vocabulary_size = df['Token'].nunique()
    document_size = df['Frequency'].sum()

    # Data for charts
    stats_data = pd.DataFrame({
        'Metric': ['Vocabulary Size', 'Document Size'],
        'Value': [vocabulary_size, document_size]
    })

    # Display bar charts
    st.write("### Bar Chart of Document Statistics")
    chart = alt.Chart(stats_data).mark_bar().encode(
        x='Metric',
        y='Value',
        color='Metric'
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    st.write("Vocabulary Size:", vocabulary_size)
    st.write("Document Size:", document_size)

    # Word search with stemming based on file type
    st.subheader("🔍 Search for Words")
search_words = st.text_input(
    "Enter words to search (separate by spaces or commas):")
tokenizer = nltk.RegexpTokenizer(
    r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*')

# Fix the condition here: checking explicitly
if selected_display_name in ["Split", "SplitPorter", "SplitLancaster"]:
    search_words = search_words.split()
else:
    tokens = tokenizer.tokenize(search_words)
    search_words = [token.lower() for token in tokens]

    # Set up stemmer if needed
    stemmers = {
        "SplitPorter": nltk.PorterStemmer(),
        "SplitLancaster": nltk.LancasterStemmer(),
        "TokenPorter": nltk.PorterStemmer(),
        "TokenLancaster": nltk.LancasterStemmer(),
    }
    stemmer = stemmers.get(selected_display_name)

    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Apply stemming to each search word if stemmer is available
    if stemmer:
        search_words = [stemmer.stem(word) for word in search_words]

    # Filter the DataFrame by each search word and combine results
    filtered_dfs = [df[df['Token'].isin([word])] for word in search_words]

    filtered_df = pd.concat(
        filtered_dfs).drop_duplicates().reset_index(drop=True)

    # Display search results
    st.write(f"**Results for words: {', '.join(search_words)}**")
    st.dataframe(filtered_df)

    weight_sums = filtered_df.groupby(
        'Document')['Poids'].sum().reset_index()
    weight_sums.columns = ['Document', 'Total Weight']

    # Display the aggregated DataFrame
    st.subheader("📊 Total Weight per Document")
    st.dataframe(weight_sums)

    # Optional: Display as a bar chart
    st.write("### Bar Chart of Total Weight per Document")
    chart = alt.Chart(weight_sums).mark_bar().encode(
        x='Document:O',
        y='Total Weight:Q',
        color='Document:O'
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    # Text File Viewer for selected Document ID
    if query:
        try:
            doc_num = int(query)
            file_name = f"D{doc_num}.txt"
            file_path = os.path.join(collection_folder, file_name)

            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    file_content = file.read()

                # Highlight occurrences of tokens in the filtered DataFrame
                highlighted_content = file_content
                for _, row in filtered_df.iterrows():
                    token = row['Token']
                    occurrences = row['Occurrence']

                    # Ensure occurrences are a valid list
                    try:
                        positions = literal_eval(occurrences)
                        if not isinstance(positions, list):
                            raise ValueError(
                                "Occurrences data is not a list.")
                        positions = [int(pos) for pos in positions]
                    except (ValueError, SyntaxError):
                        continue  # Skip this token if it's not valid

                    # Place unique markers around each token at the specified positions
                    for pos in positions:
                        highlighted_content = highlighted_content[:pos] + f"<span class='highlighted-token'>{
                            token}</span>" + highlighted_content[pos + len(token):]

                st.markdown(f"<div class='document-viewer'>{highlighted_content}</div>",
                            unsafe_allow_html=True)
            else:
                st.error("File not found.")
        except ValueError:
            st.sidebar.warning(
                "Invalid Document ID. Please enter a number between 1 and 6.")
