import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import sqlite3
conn = sqlite3.connect('blog_summarizer.db')
cursor = conn.cursor()

# Create a table to store the collected data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY,
        title TEXT,
        content TEXT,
        url TEXT,
        category TEXT
    );
''')

def textrank(text):
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and lemmatize words
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [[lemmatizer.lemmatize(word.lower()) for word in sentence if word.lower() not in stop_words] for sentence in words]

    # Create graph
    graph = nx.Graph()
    for i, sentence in enumerate(words):
        for word in sentence:
            graph.add_node(word)
        for j in range(len(sentence) - 1):
            graph.add_edge(sentence[j], sentence[j + 1])

    # Calculate PageRank scores
    scores = nx.pagerank(graph)

    # Rank sentences based on scores
    ranked_sentences = []
    for i, sentence in enumerate(sentences):
        score = sum(scores.get(word, 0) for word in words[i])
        ranked_sentences.append((score, sentence))

    # Sort sentences by score
    ranked_sentences.sort(reverse=True)

    return ranked_sentences

# Define a function to collect data from the specified websites
def collect_data(websites):
    # Use web scraping or RSS feed parsing to collect the latest articles
    # Store the collected data in the database
    import feedparser
    for website in websites.split(','):
        feed = feedparser.parse(website)
        st.write(feed)
        for entry in feed.entries:
            title = entry.title
            content = entry.summary
            url = entry.link
            category = entry.category
            cursor.execute('INSERT INTO articles (title, content, url, category) VALUES (?, ?, ?, ?)', (title, content, url, category))
    conn.commit()

# Define a function to extract key points and relevant information
def extract_info(article):
    # Use NLP techniques to extract key points and relevant information
    # Return a dictionary with the extracted information
    text = article['content']
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)
    return {'summary': text_summary}

# Define a function to summarize the extracted information
def summarize_info(info):
    # Use summarization algorithms to condense the extracted information
    # Return a concise summary
    return info['summary']
   
# Streamlit app
st.title('Blog Summarizer')

# User input
st.header('Input')
websites = st.text_input('Enter the websites you want to receive updates from (separated by commas)')
categories = st.multiselect('Select categories or topics of interest', ['Technology', 'Finance', 'Sports', 'etc.'])

# Collect data and extract information
if st.button('Collect Data'):
    collect_data(websites)
    articles = cursor.execute('SELECT * FROM articles').fetchall()
    info_list = []
    for article in articles:
        info = extract_info(article)
        info_list.append(info)

# Summarize and display the results
if st.button('Summarize and Display'):
    summaries = []
    for info in info_list:
        summary = summarize_info(info) #textrank(info)  
        summaries.append(summary)
    st.write(summaries)

# Display the results in a customizable format
st.header('Results')
display_format = st.selectbox('Select display format', ['List', 'Grid'])
if display_format == 'List':
    st.write(summaries)
elif display_format == 'Grid':
    # Display the results in a grid format
    pass

# Close the database connection
conn.close()
