import streamlit as st
from PIL import Image
import os
import ollama
import numpy as np
from streamlit_image_select import image_select

PROJECT_ROOT = os.getcwd()
IMG_PATH = os.path.join(PROJECT_ROOT, "assets")
VLM_NAME = "llava:7b"
DEMO_CHARTS = {
    "Compenso medio per genere": {
        "path": os.path.join(IMG_PATH, "avg_compensation_by_gender.png"),
        "domain": "Compensazione",
        "description": "Questo grafico mostra il compenso medio per genere."
    },
    "Dipendenti per CR e genere": {
        "path": os.path.join(IMG_PATH, "employees_by_cr_gender.png"),
        "domain": "Compensazione",
        "description": "Questo grafico mostra la distribuzione dei dipendenti per rapporto di compensazione e genere."
    },
    "Numero dipendenti per genere": {
        "path": os.path.join(IMG_PATH, "headcount_by_gender.png"),
        "domain": "Forza lavoro",
        "description": "Questo grafico mostra il numero di dipendenti per genere."
    },
    "Manager per genere": {
        "path": os.path.join(IMG_PATH, "manager_by_gender.png"),
        "domain": "Forza lavoro",
        "description": "Questo grafico mostra la distribuzione dei manager per genere."
    },
    "Anzianità per genere": {
        "path": os.path.join(IMG_PATH, "seniority_by_gender.png"),
        "domain": "Forza lavoro",
        "description": "Questo grafico mostra la distribuzione dei dipendenti per anzianità e genere."
    },
    "Talent 9-Box": {
        "path": os.path.join(IMG_PATH, "talent_box.png"),
        "domain": "Performance",
        "description": "Questo grafico mostra il numero di dipendenti in ciascuna categoria di performance-potenziale nell'ultima valutazione."
    },
    "Compensazione per fascia d'età": {
        "path": os.path.join(IMG_PATH, "total_compensation.png"),
        "domain": "Compensazione",
        "description": "Questo grafico mostra il compenso medio per fascia d'età."
    },
    "Dipendenti per categoria speciale": {
        "path": os.path.join(IMG_PATH, "workers_by_special_category.png"),
        "domain": "Compensazione",
        "description": "Questo grafico mostra la distribuzione dei dipendenti per categoria speciale."
    },
}

# =================================================================
# UI
# =================================================================
st.title("📊 HR Chart Reasoning")

if "last_selected" not in st.session_state:
    st.session_state["last_selected"] = list(DEMO_CHARTS.keys())[0]

selected_chart = st.selectbox(
    "Quale grafico vuoi analizzare?",
    (list(DEMO_CHARTS.keys())),
)

if selected_chart != st.session_state["last_selected"]:
    st.session_state["last_selected"] = selected_chart
    if "messages" in st.session_state:
        st.session_state["messages"] = []

sys_prompt = """Sei un esperto HR. Dato un grafico, devi rispondere a una domanda su di esso.
Ecco le informazioni sul grafico:
- Dominio del grafico: {domain}
- Titolo del grafico: {title}
- Descrizione del grafico: {description}

Sii conciso e diretto, utilizzando le informazioni fornite nel grafico.""".format(
    domain=DEMO_CHARTS[selected_chart]["domain"],
    title=selected_chart,
    description=DEMO_CHARTS[selected_chart]["description"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": sys_prompt}]
    st.session_state.messages.append({"role": "assistant", "content": "Posso aiutarti ad analizzare i grafici HR. Chiedimi qualsiasi cosa!"})

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    st.chat_message(msg["role"]).write(msg["content"])

image = Image.open(DEMO_CHARTS[selected_chart]["path"])

with st.chat_message("user"):
    st.image(image, caption=selected_chart, use_container_width=True)

if question := st.chat_input():
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "images": [DEMO_CHARTS[selected_chart]["path"]]
    })

    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = ollama.chat(
            model=VLM_NAME,
            messages=st.session_state.messages,
            stream=True
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
