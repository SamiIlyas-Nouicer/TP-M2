import streamlit as st
import pandas as pd
import os

# List of file names
csv_files = ["Split.csv", "SplitLancaster.csv", "SplitPorter.csv",
             "Token.csv", "TokenLancaster.csv", "TokenPorter.csv"]
display_names = [os.path.splitext(file)[0] for file in csv_files]

# File selection
st.title("Descriptor and Token Files")
selected_display_name = st.selectbox("Select a Mode", display_names)
selected_file = selected_display_name + ".csv"

# Load the selected file


def load_data(file_path):
    return pd.read_csv(file_path)


if selected_file:
    # Display the content of the selected file
    # Adjusted to 'test' folder
    file_path = os.path.join("test", selected_file)
    df = load_data(file_path)
    # Show the display name without .csv
    st.write(f"Contents of {selected_display_name}")
    st.dataframe(df)

    # Display the number of tokens and the sum of weights
    num_tokens = df['Token'].nunique()  # Number of unique tokens
    sum_weights = df['Poids'].sum()     # Sum of 'Poids' column
    st.write(f"Number of unique tokens: {num_tokens}")
    st.write(f"Sum of weights: {sum_weights}")

    # Search functionality
    search_word = st.text_input("Search for a word:")
    if search_word:
        filtered_df = df[df['Token'].str.contains(
            search_word, case=False, na=False)]
        st.write(f"Results for '{search_word}':")
        st.dataframe(filtered_df)
