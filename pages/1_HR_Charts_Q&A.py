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
    "Average Compensation by Gender": {
        "path": os.path.join(IMG_PATH, "avg_compensation_by_gender.png"),
        "domain": "Compensation",
        "description": "This chart shows the average compensation by gender."
    },
    "Employees by CR and Gender": {
        "path": os.path.join(IMG_PATH, "employees_by_cr_gender.png"),
        "domain": "Compensation",
        "description": "This chart shows the distribution of employees by compensation ratio and gender."
    },
    "Headcount by Gender": {
        "path": os.path.join(IMG_PATH, "headcount_by_gender.png"),
        "domain": "Workforce",
        "description": "This chart shows the headcount by gender."
    },
    "Manager employees by Gender": {
        "path": os.path.join(IMG_PATH, "manager_by_gender.png"),
        "domain": "Workforce",
        "description": "This chart shows the distribution of manager employees by gender."
    },
    "Seniority by Gender": {
        "path": os.path.join(IMG_PATH, "seniority_by_gender.png"),
        "domain": "Workforce",
        "description": "This chart shows the distribution of employees by seniority and gender."
    },
    "Talent 9-Box": {
        "path": os.path.join(IMG_PATH, "talent_box.png"),
        "domain": "Performance",
        "description": "This chart shows the number of employees in each performance-potential category in the latest review."
    },
    "Compensation by age group": {
        "path": os.path.join(IMG_PATH, "total_compensation.png"),
        "domain": "Compensation",
        "description": "This chart shows the average compensation by age group."
    },
    "Employees by Special Category": {
        "path": os.path.join(IMG_PATH, "workers_by_special_category.png"),
        "domain": "Compensation",
        "description": "This chart shows the distribution of employees by special category."
    },
}


st.title("📊 HR Chart Reasoning")

selected_chart = st.selectbox(
    "Which chart do you want to analyze?",
    (list(DEMO_CHARTS.keys())),
)


sys_prompt = """You are an HR expert. Given a chart, you have to answer to a question about it.
Here is the chart information:
- Chart domain: {domain}
- Chart title: {title}
- Chart description: {description}

Be concise and to the point, using the information provided in the chart.""".format(
    domain=DEMO_CHARTS[selected_chart]["domain"],
    title=selected_chart,
    description=DEMO_CHARTS[selected_chart]["description"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": sys_prompt}]
    st.session_state.messages.append({"role": "assistant", "content": "I can help you analyze HR charts. Ask me anything!"})

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
