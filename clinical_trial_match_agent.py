import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Replace this
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Initialize LLM and Embeddings ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# --- Static Clinical Trials ---
trial_data = [
    {
        "title": "Aducanumab in Early Alzheimer’s Disease",
        "content": "Inclusion: Patients aged 50-85 with early Alzheimer’s disease. MMSE score 20-26. Amyloid PET positive. Stable dose of donepezil or memantine allowed. Exclusion: History of stroke or seizure within the past 2 years."
    },
    {
        "title": "BAN2401 for Mild Cognitive Impairment Due to Alzheimer’s",
        "content": "Inclusion: Ages 55-80. Diagnosis of mild cognitive impairment due to Alzheimer’s or mild Alzheimer’s dementia. Positive CSF biomarkers or amyloid PET scan. APOE ε4 carrier allowed. Exclusion: Uncontrolled hypertension, recent major surgery."
    },
    {
        "title": "Gantenerumab Study for Preclinical Alzheimer’s",
        "content": "Inclusion: Cognitively normal individuals aged 60-75 with APOE ε4 genotype. Must have amyloid deposition confirmed by PET. No history of psychiatric illness or neurological disorder."
    }
]

# --- Process Documents ---
documents = [
    Document(page_content=trial["content"], metadata={"title": trial["title"]})
    for trial in trial_data
]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit UI ---
st.set_page_config(page_title="Clinical Trial Matcher", page_icon="🧠")
st.title("🧠 Alzheimer’s Clinical Trial Matcher")
st.markdown("Paste the patient's EHR notes or genetic data below to find suitable clinical trials.")

# Sample EHR autofill button
sample_note = """72-year-old female with mild Alzheimer's disease. MMSE: 22. APOE ε4 carrier. On stable dose of donepezil. No history of seizures or stroke. Interested in disease-modifying therapies."""
if st.button("📄 Load Sample EHR"):
    st.session_state["query"] = sample_note

query = st.text_area("📝 Patient EHR or Genetic Profile:", value=st.session_state.get("query", ""), height=200)

if query:
    with st.spinner("🔍 Matching to clinical trials..."):
        matches = retriever.get_relevant_documents(query)

        st.subheader("📋 Top Matching Trials")
        for i, doc in enumerate(matches, 1):
            st.markdown(f"**{i}. {doc.metadata['title']}**")
            st.write(doc.page_content)
            st.markdown("---")

        # GPT-4 Explanation
        prompt = f"Given this patient profile:\n{query}\n\nAnd the following trial info:\n{matches[0].page_content}\n\nExplain why this trial is a good match."
        explanation = llm.invoke(prompt)
        st.subheader("🧠 GPT-4 Eligibility Reasoning")
        st.write(explanation)
