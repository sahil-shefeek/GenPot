import streamlit as st

from server.config_manager import load_config
from server.rag_system import RAGSystem
from server.llm_client import generate_response
from server.core.prompting import HttpPromptStrategy

st.set_page_config(page_title="GenPot RAG Playground", layout="wide")

@st.cache_resource
def get_rag_system():
    return RAGSystem()

@st.cache_resource
def get_config():
    return load_config()

# Load initializations
try:
    rag = get_rag_system()
    config = get_config()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

# Load defaults from config
llm_defaults = config.get("llm_defaults", {})
default_provider = llm_defaults.get("provider", "gemini")
default_model = llm_defaults.get("model", "gemini-1.5-flash")

# Sidebar
st.sidebar.title("Settings")
top_k = st.sidebar.slider("Top K", min_value=1, max_value=15, value=rag.top_k)
alpha = st.sidebar.slider("Alpha", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

providers = ["gemini", "ollama"]
try:
    default_provider_index = providers.index(default_provider.lower())
except ValueError:
    default_provider_index = 0

provider = st.sidebar.selectbox("Provider", options=providers, index=default_provider_index)
model = st.sidebar.text_input("Model", value=default_model)

# Main area
st.title("GenPot RAG Playground")

query = st.text_area("Query", height=100, placeholder="e.g. GET /repos")

col1, col2 = st.columns([1, 1])
with col1:
    retrieve_btn = st.button("Retrieve only", use_container_width=True)
with col2:
    generate_btn = st.button("Retrieve + Generate", use_container_width=True)

if retrieve_btn or generate_btn:
    if not query.strip():
        st.warning("Please enter a query string.")
    else:
        # Retrieval
        with st.spinner("Retrieving context..."):
            inspection_results = rag.inspect_query(query.strip(), top_k=top_k, alpha=alpha)
            chunks = inspection_results.get("chunks", [])
            latency = inspection_results.get("latency_ms", 0.0)
            
        # Metrics row
        st.subheader("Retrieval Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunks Returned", len(chunks))
        m2.metric("Retrieval Latency (ms)", f"{latency:.2f}")
        m3.metric("Top K", top_k)
        m4.metric("Alpha", alpha)
        
        # Chunks Inspector
        st.subheader("Retrieved Chunks")
        if not chunks:
            st.info("No chunks retrieved.")
        else:
            for i, chunk in enumerate(chunks):
                if alpha < 1.0:
                    score_val = chunk.get("hybrid_score", 0.0)
                    score_label = f"hybrid_score: {score_val:.4f}"
                else:
                    score_val = chunk.get("faiss_distance", 0.0)
                    score_label = f"faiss_distance: {score_val:.4f}"
                
                with st.expander(f"Chunk {i+1} ({score_label})"):
                    st.code(chunk.get("text", ""))
                    
        # LLM Generation
        if generate_btn:
            st.subheader("Generation")
            
            with st.spinner("Generating response..."):
                context_text = rag.get_context(query.strip(), alpha=alpha, top_k=top_k)
                
                request_data = {
                    "method": "GET",
                    "path": query.strip(),
                    "body": "",
                    "headers": {},
                    "command": None
                }
                
                strategy = HttpPromptStrategy()
                system_prompt, prompt = strategy.build_prompt(request_data, context_text, "")
                
                with st.expander("Raw prompt"):
                    st.text(prompt)
                
                try:
                    raw_llm_output = generate_response(
                        prompt,
                        system_prompt=system_prompt,
                        provider_type=provider, 
                        model_name=model, 
                        temperature=temperature
                    )
                    
                    parsed_output = strategy.parse_response(raw_llm_output)
                    
                    llm_response = parsed_output.get("response", {})
                    side_effects = parsed_output.get("side_effects", [])
                    
                    similarity_score = rag.compute_similarity(context_text, llm_response)
                    
                    # Score metric with green/red delta
                    delta_diff = similarity_score - 0.5
                    st.metric(
                        "Similarity Score", 
                        f"{similarity_score:.4f}", 
                        delta=f"{delta_diff:+.4f}"
                    )
                    
                    st.markdown("### LLM Response")
                    st.json(llm_response)
                    
                    if side_effects:
                        st.markdown("### State side-effects")
                        st.json(side_effects)
                        
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
