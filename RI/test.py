from collections import Counter, defaultdict

import nltk


target_words = ["llm", "flan-ul2", 'graph', "query", "retriev", 'int']


def specific_word_positions(text, target_words, mode):
    word_pos = defaultdict(list)

    target_words_set = set(target_words)
    tokenizer = nltk.RegexpTokenizer(
        r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*')
    text = text.lower()
    if mode == "Split":
        tokens = text.split()
    elif mode == "SplitPorter":
        tokens = text.split()
        tokens = [nltk.PorterStemmer().stem(token) for token in tokens]
    elif mode == "SplitLancaster":
        tokens = text.split()
        tokens = [nltk.LancasterStemmer().stem(token) for token in tokens]
    elif mode == "Token":
        tokens = tokenizer.tokenize(text)
    elif mode == "TokenPorter":
        tokens = tokenizer.tokenize(text)
        tokens = [nltk.PorterStemmer().stem(token) for token in tokens]
    elif mode == "TokenLancaster":
        tokens = tokenizer.tokenize(text)
        tokens = [nltk.LancasterStemmer().stem(token) for token in tokens]

    for index, word in enumerate(tokens, start=1):
        if word in target_words_set:  # Only track positions for target words
            word_pos[word].append(index)

    return dict(word_pos)  # Convert back to a regular dictionary


text = """Query reformulation is a well-known problem in Information Retrieval (IR) aimed at enhancing single search successful completion rate by automatically modifying user's input query. Recent methods leverage Large Language Models (LLMs) to improve query reformulation, but often generate limited and redundant expansions, potentially constraining their effectiveness in capturing diverse intents. In this paper, we propose GenCRF: a Generative Clustering and Reformulation Framework to capture diverse intentions adaptively based on multiple differentiated, well-generated queries in the retrieval phase for the first time. GenCRF leverages LLMs to generate variable queries from the initial query using customized prompts, then clusters them into groups to distinctly represent diverse intents. Furthermore, the framework explores to combine diverse intents query with innovative weighted aggregation strategies to optimize retrieval performance and crucially integrates a novel Query Evaluation Rewarding Model (QERM) to refine the process through feedback loops. Empirical experiments on the BEIR benchmark demonstrate that GenCRF achieves state-of-the-art performance, surpassing previous query reformulation SOTAs by up to 12% on nDCG@10. These techniques can be adapted to various LLMs, significantly boosting retriever performance and advancing the field of Information Retrieval."""

text2 = "Query reformulation is a well-known problem in Information Retrieval (IR) aimed at enhancing single search successful completion rate by automatically modifying user's input query."

print(text2.split())
