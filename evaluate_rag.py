import os
import time
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from dotenv import load_dotenv
from rag_pipeline import query_rag_pipeline
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def create_evaluation_dataset():
    test_queries = [
        "Who said 'Be yourself; everyone else is already taken.'?",
        "What is the quote about a room without books?",
        "Find the quote about life being what happens when you're making other plans."
    ]
    ground_truths = [
        "The quote 'Be yourself; everyone else is already taken.' is attributed to Oscar Wilde.",
        "The quote is 'A room without books is like a body without a soul.', attributed to Marcus Tullius Cicero.",
        "The quote 'Life is what happens to us while we are making other plans.' is attributed to Allen Saunders."
    ]
    questions, contexts, answers = [], [], []

    print("Generating evaluation data with RAG pipeline...")
    for i, query in enumerate(test_queries):
        print(f"  - Processing query {i+1}/{len(test_queries)}: '{query[:40]}...'")
        result = query_rag_pipeline(query) # Using the corrected pipeline
        retrieved_docs = result.get('source_documents', [])
        summary = result.get('structured_answer', {}).get('summary', 'No summary generated.')
        questions.append(query)
        contexts.append([doc.get('combined', '') for doc in retrieved_docs])
        answers.append(summary)
        time.sleep(1)
        
    return Dataset.from_dict({
        "question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths
    })

def run_evaluation():
    eval_dataset = create_evaluation_dataset()
    if any(ans == "No summary generated." for ans in eval_dataset["answer"]):
        print("\nCould not generate answers. Check your API key and quota. Cannot proceed.")
        return

    print("\nRunning Ragas evaluation with OpenAI model...")
    OPENAI_API_KEY = os.getenv("API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("API_KEY not found in .env file")

    openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    st_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=openai_llm,
        embeddings=st_embeddings,
    )
    print("\n--- Final RAG Evaluation Results ---")
    print(result)

if __name__ == '__main__':
    run_evaluation()