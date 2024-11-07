import streamlit as st
import pandas as pd
import os
import nltk
# List of file names
csv_files = ["Split.csv", "SplitLancaster.csv", "SplitPorter.csv",
             "Token.csv", "TokenLancaster.csv", "TokenPorter.csv"]
display_names = [os.path.splitext(file)[0] for file in csv_files]

# File selection
st.title("Descriptor and Token Files")
selected_display_name = st.selectbox(
    "Select a selected_display_name", display_names)
selected_file = selected_display_name + ".csv"

# Load the selected file


def load_data(file_path):
    return pd.read_csv(file_path)


if selected_file:

    file_path = os.path.join("test", selected_file)
    df = load_data(file_path)
    query = st.text_input("Search for a doc:")
    if (query):
        df = df.drop(df[df["Document"] != int(query)].index)
        df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name="N°")

    st.write(f"Contents of {selected_display_name}")

    st.dataframe(df)

    # Display the number of tokens and the sum of weights # Number of unique tokens
    Vocabulare = df['Token'].nunique() - 1
    Size = df['Frequency'].sum()
    st.write(f"Number of unique tokens: {Vocabulare}")
    st.write(f"Size of Doc: {Size}")

    # Search functionality
    search_word = st.text_input("Search for a word:")
    if search_word:

        if selected_display_name == "SplitPorter":
            search_word = nltk.PorterStemmer().stem(search_word)
        elif selected_display_name == "SplitLancaster":
            search_word = nltk.LancasterStemmer().stem(search_word)

        elif selected_display_name == "TokenPorter":
            search_word = nltk.PorterStemmer().stem(search_word)

        elif selected_display_name == "TokenLancaster":
            search_word = nltk.LancasterStemmer().stem(search_word)

        filtered_df = df[df['Token'].str.contains(
            search_word, case=False, na=False)]
        st.write(f"Results for '{search_word}':")
        st.dataframe(filtered_df)
