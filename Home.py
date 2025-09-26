import streamlit as st
import torch
from typing import Union, Any
from accelerate import PartialState
from pathlib import Path
from models.LogisticRegression import LogisticRegression
from models.function_extraction_llama import LlamaForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

HF_DEFAULT_HOME = os.environ.get("HF_HOME", f"{Path.home()}/.cache/huggingface/")
PROJECT_ROOT = os.getcwd()
INPUT_DIM = 4096
LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROBING_MODEL_PATH = os.path.join(PROJECT_ROOT, "probing_model", "lr_hidden_26.pt")


# =================================================================
# Utility functions
# =================================================================
def load_probing_model(model_path):
    model = LogisticRegression(input_dim=INPUT_DIM, use_bias=True)
    saved_data = torch.load(model_path, weights_only=True)

    if isinstance(saved_data, list):
        if len(saved_data) > 0:
            if hasattr(saved_data[0], 'state_dict'):
                model.load_state_dict(saved_data[0].state_dict())
            elif isinstance(saved_data[0], dict):
                model.load_state_dict(saved_data[0])
            else:
                print(f"Unexpected format in list: {type(saved_data[0])}")
                return
        else:
            print("Empty list found in saved file")
            return
    elif isinstance(saved_data, dict):
        model.load_state_dict(saved_data)
    else:
        print(f"Unexpected save format: {type(saved_data)}")
        return
    
    model.eval()
    
    print("Model loaded successfully. Ready for inference.")

    return model


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config



def get_weight_dir(
    model_ref: str,
    *,
    model_dir: Union[str, os.PathLike[Any]] = HF_DEFAULT_HOME,
    revision: str = "main",
    repo_type="models",
    dataset_extension="json",
    subset=None,
) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist or is not a directory."

    model_path = Path(os.path.join(model_dir, "hub", "--".join([repo_type, *model_ref.split("/")])))
    assert model_path.is_dir(), f"Model path {model_path} does not exist or is not a directory."
    
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir(), f"Weight directory {weight_dir} does not exist or is not a directory."

    if repo_type == "datasets":
        if subset is not None:
            weight_dir = weight_dir / subset
        else:
            # For datasets, we need to return the directory containing the dataset files
            if dataset_extension == "json":
                weight_dir = weight_dir / "data"
    
    return weight_dir


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def load_llm(model_name, bnb_config, local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attention=False):
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "10000MB" for i in range(n_gpus)}
    attention = "flash_attention_2" if use_flash_attention else "eager"
    device_string = PartialState().process_index
    max_memory_config = max_memory if use_device_map else None

    if not local:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            attn_implementation=attention,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",  #{'':device_string},#device_map_config,       #{'':device_string},           #{"": 0},  # forces model to cuda:0 otherwise set as done in the else branch
            max_memory=max_memory_config
        )
    else:
        model_local_path = get_weight_dir(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_local_path,
            local_files_only=True,
            use_cache=False,
            attn_implementation=attention,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map={'':device_string}, #device_map_config,       #{'':device_string}, # dispatch efficiently the model on the available ressources
            max_memory=max_memory_config,
        )

    return model


def load_tokenizer(model_name, local=False):
    if not local:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    else:
        model_local_path = get_weight_dir(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_local_path, local_files_only=True, token=True)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# =================================================================
# UI
# =================================================================
probing_model = load_probing_model(model_path=PROBING_MODEL_PATH).to("cuda")
tokenizer = load_tokenizer(LLM_NAME, local=False)
llm = load_llm(LLM_NAME, create_bnb_config(), local=False)


if "llm" not in st.session_state:
    st.session_state["llm"] = llm

if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = tokenizer

if "probing_model" not in st.session_state:
    st.session_state["probing_model"] = probing_model



st.title("🌃 European Research Night 2025")
st.markdown("---")

# Demo 1: Rilevamento delle Allucinazioni
st.header("🍄 Rilevamento delle Allucinazioni")
st.markdown("🔍 **Scopri come la nostra IA rileva quando sta inventando le cose!**")

col1, col2 = st.columns([2, 1])

with col1:
        st.markdown("""
        ### Cos'è il Rilevamento delle Allucinazioni? 🤔
        
        **L'Allucinazione dell'IA** si verifica quando i modelli linguistici generano informazioni che suonano plausibili ma sono in realtà incorrette o inventate. La nostra demo utilizza **tecniche di probing avanzate** per rilevare queste allucinazioni in tempo reale!
        
        ### Come Funziona 🧠
        
        1. **🔬 Neural Probing**: Addestriamo un classificatore leggero per analizzare le rappresentazioni interne del nostro modello Llama-3.1-8B
        2. **📊 Analisi degli Stati Nascosti**: Il classificatore esamina gli stati nascosti del layer 26 (4096 dimensioni) per identificare pattern di incertezza
        3. **⚡ Rilevamento in Tempo Reale**: Mentre il modello genera risposte, il nostro modello di probing valuta simultaneamente i livelli di confidenza
        4. **🔍 Classificazione Binaria**: Il sistema determina se la risposta è probabile che contenga allucinazioni
        
        ### Perché È Importante 🌟
        - **🛡️ IA Affidabile**: Aiuta gli utenti a identificare quando le risposte dell'IA potrebbero essere inaffidabili
        - **🔬 Impatto della Ricerca**: Contribuisce a un deployment più sicuro dell'IA in applicazioni critiche
        - **📚 Valore Educativo**: Dimostra tecniche all'avanguardia nell'interpretabilità dell'IA
        """)
    
with col2:
    st.markdown("""
    ### Stack Tecnico 🔧
    - **Modello**: Llama-3.1-8B-Instruct
    - **Probing**: Regressione Logistica
    - **Layer Target**: Layer 26
    - **Dimensioni Input**: 4096
    - **Lingua**: Italiano 🇮🇹
    
    ### Caratteristiche Principali ✨
    - Analisi in tempo reale
    - Feedback visivo
    - Chat interattiva
    - Nessuna API esterna necessaria
    """)


st.markdown("---")

# Demo 2: Analisi Grafici HR e Insights
st.header("🚀 Analisi e Insights dei Grafici HR")
st.markdown("📊 **Trasforma i tuoi dati HR in insights azionabili con l'IA!**")

col1, col2 = st.columns([2, 1])

with col1:
        st.markdown("""
        ### Cos'è l'Analisi Grafici HR Q&A? 📈
        
        Il nostro sistema di **Visual Question Answering (VQA)** combina computer vision e natural language processing per analizzare grafici HR e rispondere a domande complesse sui dati della tua forza lavoro.
        
        ### Categorie di Grafici Disponibili 📋
        
        **🏢 Analisi della Forza Lavoro**
        - Distribuzione del personale per genere
        - Analisi della gerarchia manageriale
        - Suddivisione per livello di anzianità
        
        **💰 Intelligence sulla Retribuzione**  
        - Retribuzione media per genere
        - Analisi dell'equità salariale (rapporti CR)
        - Tendenze retributive basate sull'età
        - Classificazioni di categorie speciali
        
        **⭐ Gestione delle Performance**
        - Matrice di valutazione dei talenti 9-Box
        - Correlazioni performance-potenziale
        
        ### Come Funziona 🔄
        
        1. **📸 Upload Grafico**: Seleziona dai grafici di visualizzazione HR pre-caricati
        2. **🤖 Vision AI**: Il modello Llava-7B elabora le informazioni visive
        3. **💬 Query Naturali**: Fai domande in linguaggio naturale sui dati
        4. **🎯 Analisi Esperta**: Ottieni insights focalizzati su HR con comprensione contestuale
        """)

with col2:
    st.markdown("""
        ### Stack Tecnico 🛠️
        - **Modello Vision**: Llava-7B
        - **Framework**: Ollama
        - **Tipi di Grafici**: Visualizzazioni PNG
        - **Domini**: 3 aree HR chiave
        
        ### Casi d'Uso 🎯
        - **📊 Esplorazione Dati**: Insights rapidi da grafici complessi
        - **🔍 Analisi Tendenze**: Identifica pattern e anomalie  
        - **📝 Generazione Report**: Reporting HR assistito dall'IA
        - **⚖️ Valutazione Equità**: Analisi del divario di genere e salariale
        - **📈 Pianificazione Strategica**: Decisioni HR basate sui dati
        
        ### Vantaggi 💡
        - Non serve esperienza nella lettura di grafici
        - Risposte istantanee alle domande
        - Conoscenza HR multi-dominio  
        - Esplorazione interattiva
        """)

st.markdown("---")

# Invito all'Azione
st.markdown("""
### 🎮 Pronto per Esplorare?

Scegli la tua avventura qui sotto e scopri il potere dell'IA in azione!
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px; border: 2px solid #ff6b6b; border-radius: 10px; background-color: #ffe6e6;">
    <h3>🍄 Prova il Rilevamento delle Allucinazioni</h3>
    <p><strong>Fai domande in italiano e guarda la nostra IA rilevare le proprie incertezze!</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; border: 2px solid #4ecdc4; border-radius: 10px; background-color: #e6fffe;">
    <h3>📊 Esplora l'Analisi HR</h3>
    <p><strong>Interroga i grafici della forza lavoro e ottieni insights istantanei sulla tua organizzazione!</strong></p>
    </div>
    """, unsafe_allow_html=True)
