import streamlit as st
from agents import agent
from db import init_db, seed_data
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Multi-Agent Concierge", layout="wide")
st.title("🤖 Multi-Agent Concierge — Book doctors, plumbers, flights & house help")


init_db()
seed_data()

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask me (e.g., 'Book a cardiologist tomorrow at 10 AM')")

if st.button("Send") and user_input:
    st.session_state.chat.append(("You", user_input))
    with st.spinner("Agent thinking..."):
        res = agent.run(user_input)
    st.session_state.chat.append(("Agent", res))

for who, text in st.session_state.chat[::-1]:
    if who == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Agent:** {text}")
