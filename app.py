import streamlit as st
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Preprocess text
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words]

# Perform LDA
def perform_topic_modeling(data, num_topics=5):
    tokenized_reviews = [preprocess(review) for review in data['review']]
    dictionary = corpora.Dictionary(tokenized_reviews)
    corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary

# Streamlit App
st.title("Social Media Topic Modeling")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type="csv")

if uploaded_file:
    # Read data
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", data.head())
    
    # Number of topics
    num_topics = st.slider("Select Number of Topics", min_value=2, max_value=10, value=5)
    
    # Perform Topic Modeling
    st.write("### Performing Topic Modeling...")
    lda_model, corpus, dictionary = perform_topic_modeling(data, num_topics)
    
    # Display Topics
    st.write("### Topics Identified:")
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        st.write(f"Topic {topic[0]}: {topic[1]}")
    
    # Wordcloud for Topics
    st.write("### Topic Word Clouds")
    for topic_id, topic_words in topics:
        topic_dict = {word.split("*")[1].replace('"', '').strip(): float(word.split("*")[0]) for word in topic_words.split(" + ")}
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(topic_dict)
        
        # Display WordCloud
        st.subheader(f"Topic {topic_id}")
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

st.write("Upload a dataset to get started.")
