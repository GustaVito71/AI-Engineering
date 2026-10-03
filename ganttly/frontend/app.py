import streamlit as st

st.set_page_config(page_title="Ganttly UI", page_icon=":bar_chart:", layout="wide")
st.title("Ganttly UI")

tab_chat, tab_dash = st.tabs(["Chat", "Dashboard"])

with tab_chat:
    st.subheader("Chat")
    prompt = st.text_input("Escribe tu mensaje")
    if st.button("Enviar"):
        st.info(f"Integración de IA pendiente — mensaje recibido: {prompt}")

with tab_dash:
    st.subheader("Dashboard de ejemplo")
    st.bar_chart({"Serie A": [3, 1, 4, 2, 5], "Serie B": [2, 5, 1, 4, 3]})