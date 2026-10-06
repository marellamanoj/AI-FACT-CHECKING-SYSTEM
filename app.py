import streamlit as st

from main import load_dataset, load_classifier, fact_check, DATA_PATH
from index_manager import load_or_build_indexes

@st.cache_resource
def load_resources():
    """Load all the resources we need: dataset, indexes, models."""
    df = load_dataset(DATA_PATH)
    statements = df["statement"].tolist()

    # Load or build indexes (with caching for faster subsequent runs)
    bm25, faiss_index, dense_model = load_or_build_indexes(DATA_PATH, statements)

    # Load classifier (fine-tuned BERT)
    classifier, tokenizer = load_classifier()

    return df, bm25, faiss_index, dense_model, classifier, tokenizer

def main():
    st.title("Fact-Checking System with LIAR Dataset")
    st.write("Enter a claim to verify its veracity.")

    # Load resources (caches after first run)
    df, bm25, faiss_index, dense_model, classifier, tokenizer = load_resources()

    # Text input
    user_query = st.text_input("Enter your claim here:")

    # Checkbox for verbose (optional)
    verbose_mode = st.checkbox("Show detailed evidence (verbose mode)?", value=False)

    # Checkbox for LLM-based response generation
    use_llm = st.checkbox("Use LLM for response generation", value=True)

    # Button to run the fact-check
    if st.button("Check Claim"):
        if not user_query.strip():
            st.warning("Please enter a valid claim.")
        else:
            # Call your fact_check function
            response = fact_check(
                user_query,
                bm25,
                faiss_index,
                dense_model,
                df,
                classifier,
                tokenizer,
                verbose=verbose_mode,
                use_llm=use_llm
            )
            st.markdown("### Result")
            st.write(response)

if __name__ == "__main__":
    main()