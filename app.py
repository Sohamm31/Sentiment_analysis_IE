import pickle
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from safetensors.torch import load_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# 🔹 Load Sentiment Analysis Models
lstm_model = pickle.load(open("LSTM_MODEL.pkl", "rb"))
lstm_tokenizer = pickle.load(open("tokenizer_LSTM.pkl", "rb"))

rf_model = pickle.load(open("sentiment_model.pkl", "rb"))
rf_vect = pickle.load(open("vectorizer.pkl", "rb"))

model_dir = "BERT_MODEL_SAVED"
model_weights = f"{model_dir}/model.safetensors"
bert_tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
bert_model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
bert_model.load_state_dict(load_file(model_weights))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# 🔹 Load Hugging Face Model for Summarization
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-alpha", task="text-generation")
model = ChatHuggingFace(llm=llm)

# 🔹 Streamlit UI Enhancements
st.set_page_config(page_title="Sentiment Analysis", layout="centered", page_icon="📊")

st.title(" Sentiment Analysis Dashboard")
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

review_text = st.text_area("✍️ Enter Review", height=120)

# 🔹 Sentiment Prediction Functions
def predict_sentiment_bert(text):
    inputs = bert_tokenizer(text, truncation=True, padding=True, max_length=120, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(outputs, dim=1).item()

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    confidence = probabilities[predicted_class]
    return sentiment, confidence

def predict_sentiment_lstm(review, model, tokenizer, max_length=120):
    sequence = tokenizer.texts_to_sequences([review.lower()])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return sentiment, confidence

def predict_sentiment_rf(review, model, vectorizer):
    sample_features = vectorizer.transform([review])
    prediction = model.predict(sample_features)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

# 🔹 Generate Summary Function
def generate_summary(review):
    messages = [
        SystemMessage(content="You are an AI assistant that summarizes customer reviews concisely."),
        HumanMessage(content=f"Summarize the following review in one or two sentences:\n\n{review}")
    ]
    result = model.invoke(messages)
    return result.content

# 🔹 Predict Sentiment Button
if st.button(" Predict Sentiment", use_container_width=True):
    if review_text.strip():
        sentiment_lstm, confidence_lstm = predict_sentiment_lstm(review_text, lstm_model, lstm_tokenizer)
        sentiment_rf = predict_sentiment_rf(review_text, rf_model, rf_vect)
        sentiment_bert, confidence_bert = predict_sentiment_bert(review_text)

        sentiments = [sentiment_lstm, sentiment_rf, sentiment_bert]
        final_sentiment = max(set(sentiments), key=sentiments.count)

        # 🔹 Store Results in Session State
        st.session_state["sentiment_lstm"] = sentiment_lstm
        st.session_state["confidence_lstm"] = confidence_lstm
        st.session_state["sentiment_rf"] = sentiment_rf
        st.session_state["sentiment_bert"] = sentiment_bert
        st.session_state["confidence_bert"] = confidence_bert
        st.session_state["review_text"] = review_text

        # 🔹 Display Best Sentiment
        st.markdown(f"<h3>Best Sentiment Prediction: <span style='color: {'green' if final_sentiment == 'Positive' else 'red'};'>{final_sentiment}</span></h3>", unsafe_allow_html=True)

# 🔹 Generate Summary Button
if "sentiment_lstm" in st.session_state and st.button(" Generate Summary", use_container_width=True):
    if st.session_state["review_text"].strip():
        summary = generate_summary(st.session_state["review_text"])
        st.session_state["summary"] = summary
        st.subheader(" Summary")
        st.write(summary)

# 🔹 Analyze Results Button
if "sentiment_lstm" in st.session_state and st.button(" Analyze Results", use_container_width=True):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📌 Model Predictions")

    col1, col2, col3 = st.columns(3)
    col1.metric("LSTM", st.session_state["sentiment_lstm"], f"Confidence: {st.session_state['confidence_lstm']:.2f}")
    col2.metric("Random Forest", st.session_state["sentiment_rf"], "N/A")
    col3.metric("BERT", st.session_state["sentiment_bert"], f"Confidence: {st.session_state['confidence_bert']:.2f}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader(" Confidence Scores")

    fig, ax = plt.subplots()
    models = ["LSTM", "BERT"]
    confidence_scores = [st.session_state["confidence_lstm"], st.session_state["confidence_bert"]]
    sns.barplot(x=models, y=confidence_scores, palette=["blue", "red"], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_xlabel("Models")
    ax.set_title("Model Confidence Scores")
    st.pyplot(fig)

    st.subheader(" Sentiment Agreement")
    sentiment_counts = [st.session_state["sentiment_lstm"], st.session_state["sentiment_rf"], st.session_state["sentiment_bert"]]
    positive_count = sentiment_counts.count("Positive")
    negative_count = sentiment_counts.count("Negative")

    fig, ax = plt.subplots()
    ax.pie([positive_count, negative_count], labels=["Positive", "Negative"], autopct="%1.1f%%", colors=["green", "red"])
    st.pyplot(fig)

    st.subheader("🌤️ Word Cloud")
    wordcloud = WordCloud(width=500, height=300, background_color="white").generate(st.session_state["review_text"])
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
