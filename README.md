#  Clinical Trial Matching Agent (Alzheimer’s Edition)

This AI-powered Streamlit app matches unstructured patient EHR notes or genetic data to relevant Alzheimer’s clinical trials using semantic search and GPT-4-based reasoning.

##  What It Does

- Accepts raw patient EHR or genetic profile as input
- Matches top clinical trials based on semantic similarity
- Uses GPT-4 to explain why each trial is a good fit
- Provides a user-friendly Streamlit interface

##  Tech Stack

| Component | Technology |
|----------|------------|
| Language Model | OpenAI GPT-4 |
| Embeddings | `text-embedding-3-small` via OpenAI |
| Vector DB | FAISS |
| Framework | LangChain |
| UI | Streamlit |

##  Sample EHR Input

```text
72-year-old female with mild Alzheimer's disease. MMSE: 22. APOE ε4 carrier. On stable dose of donepezil. No history of seizures or stroke. Interested in disease-modifying therapies.
