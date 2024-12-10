import streamlit as st
import pandas as pd
import altair as alt
import os
import nltk
import numpy as np
import re
from ast import literal_eval




# Configure the page
st.set_page_config(page_title="Descriptor and Token Files", layout="centered")
nltk.download('stopwords')
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
    df_copy = df
    # Sidebar for Document ID search
    st.sidebar.subheader("🔍 Search by Document ID")
    query = st.sidebar.text_input("Enter Document ID:",key=0)

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
    search_words = st.text_input("Enter words to search (separate by spaces or commas):",key=1)

def process_search_words(selected_display_name, search_words):
    """
    Processes the search words based on the selected display name, including tokenization and stemming.
    
    Args:
        selected_display_name (str): The selected display name for determining whether to apply stemming.

    Returns:
        list: A list of processed search words with stemming applied to terms only.
    """
    # Define tokenizer and stemmers
    operator = re.compile(r'\b(?:AND|OR|NOT)\b')  # Match operators as uppercase
    tokenizer = nltk.RegexpTokenizer(
        r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*'
    )
    stemmers = {
        "SplitPorter": nltk.PorterStemmer(),
        "SplitLancaster": nltk.LancasterStemmer(),
        "TokenPorter": nltk.PorterStemmer(),
        "TokenLancaster": nltk.LancasterStemmer(),
    }
    stemmer = stemmers.get(selected_display_name)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Tokenize the search words
    if selected_display_name in ["Split", "SplitPorter", "SplitLancaster"]:
        search_words = search_words.split()  # Split by spaces or commas
    else:
        tokens = tokenizer.tokenize(search_words)
        search_words = [token.lower() for token in tokens]

    # Apply stemming to terms only, not operators (keep operators in uppercase)
    processed_words = []
    for word in search_words:
        if operator.match(word.upper()):
            # Keep the operator in uppercase
            processed_words.append(word.upper())
        else:
            # Apply stemming to terms
            if stemmer:
                processed_words.append(stemmer.stem(word))
            else:
                processed_words.append(word)

    return processed_words

search_words = process_search_words(selected_display_name, search_words)

# Filter the DataFrame by each search word and combine results
filtered_dfs = [df[df['Token'].isin([word])] for word in search_words]

filtered_df = pd.concat(
    filtered_dfs).drop_duplicates().reset_index(drop=True)

# Display search results
st.write(f"**Results for words: {', '.join(search_words)}**")
st.dataframe(filtered_df)
# Number of Items per Document
item_counts = filtered_df.groupby('Document').size().reset_index(name='Number of Items')
V = len(filtered_df['Token'].unique())
# Total Weight (Scalar Product) for Filtered Terms
scalar_df = filtered_df.groupby('Document')['Poids'].sum().reset_index()
scalar_df.columns = ['Document', 'Scalar']
# Create Scalar Product DataFrame  
scalar_product_df = scalar_df.copy()

# Squared Weights for All Terms in Each Document
df['Squared_Weights'] = df['Poids'] ** 2
squared_weight_sums = df.groupby('Document')['Squared_Weights'].sum().reset_index()
squared_weight_sums.columns = ['Document', 'Sum of Squared Weights']
squared_weight_sums["square root of squared sums"] = squared_weight_sums['Sum of Squared Weights'] ** 0.5
# Merge Data for Cosine Measure
merged_cosine_data = scalar_df.merge(item_counts, on='Document').merge(squared_weight_sums, on='Document')

# Compute Cosine Measure
merged_cosine_data['Cosine'] = merged_cosine_data['Scalar'] / (
    (np.sqrt(V)) * (squared_weight_sums["square root of squared sums"])
)
RSV_df = merged_cosine_data[['Document', 'Cosine']]

# Total Weight (Query Weight Sum) for Filtered Terms
query_weight_sums = filtered_df.groupby('Document')['Poids'].sum().reset_index()
query_weight_sums.columns = ['Document', 'Query Weight Sum']

# Total Weight (All Terms in Each Document)
all_weight_sums = df.groupby('Document')['Poids'].sum().reset_index()
all_weight_sums.columns = ['Document', 'Total Weight']

# st.dataframe(df)

# st.dataframe(all_weight_sums)
# st.dataframe(merged_cosine_data)


# Merge Data for Jacard Measure
merged_jacard_data = query_weight_sums.merge(all_weight_sums, on='Document').merge(
    squared_weight_sums, on='Document'
).merge(item_counts, on='Document')

# Compute Jacard Measure
merged_jacard_data['Jacard'] = scalar_df['Scalar'] / (
    V + merged_jacard_data['Sum of Squared Weights'] - scalar_df['Scalar']
)
jacard_df = merged_jacard_data[['Document', 'Jacard']]



# Display the aggregated DataFrame
col1, col2, col3 = st.columns(3)

# Display the DataFrames in their respective columns
with col1:
    st.subheader("📊 Scalar Product")
    st.dataframe(scalar_product_df)

with col2:
    st.subheader("📊 Cosine Mesure")
    st.dataframe(RSV_df)

with col3:
    st.subheader("📊 Jaccard Mesure")
    st.dataframe(jacard_df)



# BM25
df_data = df.groupby("Document")["Frequency"].sum().reset_index()
df_data.columns = ["Document","Taille Doc"]
size_all = df_copy["Frequency"].sum()
K = st.number_input("Enter the value of K (e.g., 1.5):", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
B = st.number_input("Enter the value of B (e.g., 0.75):", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
N = st.number_input("Enter the total number of documents (N):", min_value=1, max_value=1000, value=6, step=1)


doc_taille_map = df_data.set_index("Document")["Taille Doc"]

filtered_df["Taille Doc"] = filtered_df["Document"].map(doc_taille_map)

filtered_df["BM-terme"] = (filtered_df["Frequency"] / 
     (K * ((1 - B) + B * (filtered_df["Taille Doc"] / (size_all/6))) + filtered_df["Frequency"])
     ) * np.log10((6 - filtered_df["Occurrence"] + 0.5) / (filtered_df["Occurrence"] + 0.5))



df_BM = filtered_df.groupby("Document")["BM-terme"].sum().reset_index()
df_BM.columns = ["Document","BM-25"]
st.dataframe(df_BM)
# ******************************************************************************************************************************
st.title("Appariement")


import pandas as pd
import re
import nltk

def process_query_for_eval(query, document_tokens):
    """
    Prepares the logical query by replacing logical operators and checking if terms exist in the document's tokens.
    
    Args:
        query (str): The logical query to evaluate (e.g., 'term1 AND term2 OR term3').
        document_tokens (set): A set of tokens present in the document.
    
    Returns:
        bool: The result of the evaluated query.
    """
    # Replace logical operators with Python equivalents
    query = query.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
    
    # Check if terms exist in the document's tokens
    terms = re.findall(r'\b\w+\b', query)  # Extract terms (words)
    
    for term in terms:
        term_lower = term.lower()
        if term_lower not in document_tokens:
            query = query.replace(term, 'False')  # Replace missing term with 'False'
        else:
            query = query.replace(term, 'True')   # Replace present term with 'True'
    
    try:
        # Evaluate the final query using Python's eval function
        return eval(query)
    except Exception as e:
        print(f"Error evaluating query: {query}")
        return False  # In case of invalid query, return False


def check_logical_queries(df, queries):
    """
    Check logical expressions for each document in the DataFrame based on a list of queries.
    
    Args:
        df (DataFrame): DataFrame with tokens and their respective documents.
        queries (list): List of logical queries to evaluate for each document.
    
    Returns:
        DataFrame: A new DataFrame with evaluation results for each query per document.
    """
    # Get a set of unique documents
    documents = df['Document'].unique()
    
    # Prepare a list to store results
    results = []
    
    # Iterate through documents and evaluate queries
    for document in documents:
        # Get all tokens in the document (case insensitive match)
        document_tokens = set(df[df['Document'] == document]['Token'].str.lower())
        
        # Token count for the document
        token_count = len(document_tokens)
        
        # Evaluate each query and store results
        query_results = []
        for query in queries:
            result = process_query_for_eval(query, document_tokens)
            query_results.append('TRUE' if result else 'FALSE')
        
        # Append the document number, token count, and query results
        results.append([document, token_count, query_results])
    
    # Create a DataFrame with the results, grouping queries in a list per document
    result_df = pd.DataFrame(results, columns=['Document', 'Token_Count', 'Query_Results'])
    
    return result_df





def evaluate_queries_for_all_docs(result_df):
    """
    Evaluate logical queries at once and get the overall result for each document.
    
    Args:
        result_df (DataFrame): DataFrame with document, token count, and query results.
    
    Returns:
        DataFrame: A DataFrame with evaluated results, showing 'TRUE' or 'FALSE' based on the OR operation.
    """
    # Combine query results with OR logic for each document
    result_df['Query_Results'] = result_df['Query_Results'].apply(lambda x: 'TRUE' if 'TRUE' in x else 'FALSE')
    return result_df


# Input field for the boolean query in Streamlit
query = st.text_input("Enter a boolean query:", key='2')

# Process the query to apply stemming and tokenization
query = process_search_words(selected_display_name, query)

# Example DataFrame 'df' with token data
result_df = check_logical_queries(df, [query])

# Evaluate the overall query result based on OR operation
result_df = evaluate_queries_for_all_docs(result_df)

# Display the final DataFrame in Streamlit
st.dataframe(result_df)