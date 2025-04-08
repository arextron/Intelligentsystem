import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Chatbot Performance Dashboard", layout="wide")

st.title("🤖 Chatbot Metrics Dashboard")

# Load logs
@st.cache_data
def load_chat_logs():
    try:
        return pd.read_csv("chat_logs.csv", parse_dates=["timestamp"])
    except:
        return pd.DataFrame()

@st.cache_data
def load_feedback():
    try:
        return pd.read_csv("feedback.csv", parse_dates=["timestamp"])
    except:
        return pd.DataFrame()

chat_df = load_chat_logs()

# === Metrics Calculation ===
st.subheader("📊 Key Performance Metrics")
if not chat_df.empty:
    total_logs = len(chat_df)
    avg_rating = chat_df["rating"].mean()
    # Convert 'contextual' to numeric (1/0) if it's in string or bool form
    chat_df["contextual"] = chat_df["contextual"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})

    context_rate = chat_df["contextual"].mean() * 100

    high_rating_accuracy = (chat_df["rating"] >= 4).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Avg Rating (Satisfaction)", f"{avg_rating:.2f} / 5")
    col2.metric("🔄 Contextual Coherence", f"{context_rate:.2f}%")
    col3.metric("✅ Approx Accuracy (≥4★)", f"{high_rating_accuracy:.2f}%")
else:
    st.info("No chat data found to compute metrics.")

# === Chat Summary ===
st.subheader("📈 Chat Summary")
if not chat_df.empty:
    st.write(f"Total Chats: {len(chat_df)}")
    st.write(f"Users: {chat_df['user_id'].nunique()}")

    col1, col2 = st.columns(2)
    with col1:
        agent_counts = chat_df['agent'].value_counts()
        st.bar_chart(agent_counts)

    with col2:
        st.metric("Contextual Query Rate", f"{context_rate:.2f}%")

else:
    st.info("No chat logs found.")

# === User Satisfaction Summary ===
st.subheader("📝 User Satisfaction")
if not chat_df.empty:
    rating_counts = chat_df["rating"].value_counts().sort_index()
    st.bar_chart(rating_counts)

    st.metric("Average Rating", f"{avg_rating:.2f} / 5")

    st.dataframe(chat_df.sort_values("timestamp", ascending=False), use_container_width=True)
else:
    st.info("No ratings found.")

# === Search / Filter ===
st.subheader("🔍 Explore Logs")
user_filter = st.text_input("Filter by User ID (partial match allowed):")

if not chat_df.empty:
    display_df = chat_df.copy()
    if user_filter:
        display_df = display_df[display_df["user_id"].str.contains(user_filter, case=False)]

    if "timestamp" in display_df.columns:
        display_df = display_df.sort_values("timestamp", ascending=False)

    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No logs available.")


import requests

# === LLM Evaluation Utilities ===
def evaluate_response_with_llm(question, answer, model="llama3"):
    prompt = f"""Rate the accuracy of the following chatbot response on a scale from 1 to 5.
Question: {question}
Response: {answer}
Criteria: Is the response factually correct, relevant, and clearly addresses the user's question?
Respond with just a number (1 to 5)."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=20
        )
        result = response.json().get("response", "").strip()
        return int(result[0]) if result and result[0].isdigit() else None
    except Exception as e:
        return None

# === Run Auto Evaluation ===
st.subheader("🤖 LLM-Based Auto Evaluation")
if st.button("🔍 Run LLM Accuracy Evaluation (1–5)"):
    with st.spinner("Evaluating responses..."):
        chat_df["llm_accuracy"] = chat_df.apply(
            lambda row: evaluate_response_with_llm(row["input"], row["response"]), axis=1
        )
        chat_df.to_csv("chat_logs_with_llm_accuracy.csv", index=False)
        st.success("Auto-evaluation completed and saved!")

# === Visualize LLM Accuracy (if available)
if "llm_accuracy" in chat_df.columns:
    st.subheader("📊 LLM Auto-Evaluated Accuracy Scores")
    st.metric("Avg LLM Accuracy", f"{chat_df['llm_accuracy'].mean():.2f} / 5")
    st.bar_chart(chat_df["llm_accuracy"].value_counts().sort_index())

# === Accuracy by Agent (LLM-Evaluated)
if "llm_accuracy" in chat_df.columns:
    st.subheader("🧠 LLM Accuracy by Agent")

    agent_avg = chat_df.groupby("agent")["llm_accuracy"].mean().sort_values()
    st.bar_chart(agent_avg)

    st.write("Detailed breakdown by agent:")
    st.dataframe(chat_df.groupby("agent")["llm_accuracy"].agg(["mean", "count"]).round(2))
