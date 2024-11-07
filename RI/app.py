import streamlit as st
import pandas as pd
import os
import nltk

# List of file names and display names
csv_files = ["Split.csv", "SplitLancaster.csv", "SplitPorter.csv",
             "Token.csv", "TokenLancaster.csv", "TokenPorter.csv"]
display_names = [os.path.splitext(file)[0] for file in csv_files]

# File selection
st.title("Descriptor and Token Files")
selected_display_name = st.selectbox("Select a file", display_names)
selected_file = selected_display_name + ".csv"

# Load CSV data


def load_data(file_path):
    return pd.read_csv(file_path)


if selected_file:
    file_path = os.path.join("test", selected_file)
    df = load_data(file_path)

    # Document ID Search
    query = st.text_input("Search for a Document ID:")
    if query:
        try:
            # Convert query to an integer and filter by Document column
            query = int(query)
            df = df[df["Document"] == query].reset_index(drop=True)
        except ValueError:
            st.write("Please enter a valid Document ID (numeric).")

    # Display the filtered or full DataFrame
    st.write(f"Contents of {selected_display_name}")
    st.dataframe(df)

    # Calculate vocabulary size and document size
    vocabulary_size = df['Token'].nunique()  # Unique tokens count
    document_size = df['Frequency'].sum()  # Total frequency across all tokens

    # Display vocabulary size and document size to the user
    st.write(f"Number of unique tokens (Vocabulary Size): {vocabulary_size}")
    st.write(f"Size of Document (Total Frequency): {document_size}")

    # Search for specific token with stemming based on file type
    search_word = st.text_input("Search for a word:")
    if search_word:
        # Define stemmers based on file type
        stemmers = {
            "SplitPorter": nltk.PorterStemmer(),
            "SplitLancaster": nltk.LancasterStemmer(),
            "TokenPorter": nltk.PorterStemmer(),
            "TokenLancaster": nltk.LancasterStemmer(),
        }
        # Apply stemming if applicable
        stemmer = stemmers.get(selected_display_name)
        if stemmer:
            search_word = stemmer.stem(search_word)

        # Filter DataFrame by search term in Token column
        filtered_df = df[df['Token'].str.contains(
            search_word, case=False, na=False)]
        st.write(f"Results for '{search_word}':")
        st.dataframe(filtered_df)
