import streamlit as st
import torch
from models.InspectOutputContext import InspectOutputContext
import os


PROJECT_ROOT = os.getcwd()
MODULE_NAME = ['model.layers.26']

SYS_QUESTION = """Sei un esperto in diversi campi. Ti verrà posta una domanda e il tuo compito sarà quello di fornire la risposta corretta.
Non fornire ulteriori informazioni o spiegazioni.
Rispondi direttamente alla domanda in italiano."""

USR_QUESTION = """Domanda: {question}
Risposta: """

DEMO_QST = "Qual è il pianeta più grande del sistema solare?"
DEMO_ASW = "Giove"


# =================================================================
# Utility functions
# =================================================================
@torch.no_grad()
def ask_and_probe(max_new_tokens=30, question=""):
    usr_prompt = USR_QUESTION.format(question=question)

    messages = st.session_state.messages.copy()
    messages.append({"role": "user","content": USR_QUESTION.format(question=DEMO_QST)})
    messages.append({"role": "assistant","content": DEMO_ASW})

    st.session_state.messages.append({"role": "user","content": usr_prompt})
    messages.append({"role": "user","content": usr_prompt})

    tokens = st.session_state.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    attention_mask = tokens["attention_mask"].to("cuda") if "attention_mask" in tokens else None

    with InspectOutputContext(st.session_state.llm, MODULE_NAME) as inspect:
        output = st.session_state.llm.generate(
            input_ids=tokens["input_ids"].to("cuda"),
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.5,
            top_p=None,
            pad_token_id=st.session_state.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        generated_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
        generated_text = st.session_state.tokenizer.decode(generated_ids, skip_special_tokens=True)

    for module, ac in inspect.catcher.items():
        # ac: [batch_size, sequence_length, hidden_dim]
        ac_last = ac[0, -1].float().to("cuda")
        output_prob = st.session_state.probing_model(ac_last)
        predicted = (output_prob > 0.5).float()
        break

    del tokens
    del messages
    del output
    del output_prob
    del ac
    del ac_last
    del inspect
    del generated_ids
    torch.cuda.empty_cache()

    return generated_text, predicted


def clean_msg(msg):
    msg = msg.replace("Domanda: ", "")
    msg = msg.replace("Risposta: ", "")
    return msg

# =================================================================
# UI
# =================================================================
st.title("🍄 Hallucination Detection")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYS_QUESTION}]
    st.session_state.messages.append({"role": "assistant", "content": "Chiedimi qualsiasi cosa!"})

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue

    st.chat_message(msg["role"]).write(clean_msg(msg["content"]))

if question := st.chat_input():
    st.chat_message("user").write(question)

    full_response, is_hallucination = ask_and_probe(
        question=question,
    )
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response + (" [⚠️ Hallucination detected]" if is_hallucination else " [✅ No hallucination detected]")
    })

    st.chat_message("assistant").write(full_response + (" [⚠️ Hallucination detected]" if is_hallucination else " [✅ No hallucination detected]"))
