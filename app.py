import streamlit as st
from rag_pipeline import query_rag_pipeline

st.set_page_config(page_title="Semantic Quote Finder 📚", layout="wide")
st.title("🤖 Semantic Quote Finder")
st.markdown("Use the boxes below to search for quotes by topic, by author, or both!")


query = st.text_input("Search for quotes about a topic (e.g., 'life and love'):")
author_filter = st.text_input("Filter by Author (optional, e.g., 'Oscar Wilde'):")


if st.button("Search"):
  
    if query or author_filter:
        with st.spinner("Finding quotes..."):
            result = query_rag_pipeline(query, author_filter=author_filter)
            answer = result.get('structured_answer', {})
            sources = result.get('source_documents', [])

            st.divider()
            st.subheader("💡 Results")
            
            if "summary" in answer:
                st.markdown(f"> {answer['summary']}")
            
            st.subheader("📊 Structured JSON Response")
            st.json(answer)

            st.subheader("🔍 Found Quotes")
            with st.expander("Click to see the retrieved quotes"):
                if sources:
                    for i, doc in enumerate(sources):
                        score = doc.get('similarity_score', 0)
                        score_display = f"(Similarity: {score:.2f})" if score > 0 else ""
                        st.markdown(f"**Quote {i+1}** {score_display}")
                        st.info(f"_{doc['quote']}_ \n\n**Author:** {doc['author']} | **Tags:** {', '.join(doc.get('tags', []))}")
                else:
                    st.warning("No quotes were found matching your criteria.")