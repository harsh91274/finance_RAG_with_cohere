import streamlit as st
import pandas as pd
import pickle
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.docstore.document import Document
import cohere

# Load the loan data with embeddings
loan_data = pd.read_pickle("loan_data_with_embeddings.pkl")

# Load the trained model
with open("loan_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load the FAISS index for regulations
index = faiss.read_index("regulations_faiss_index.bin")

# Load the metadata for regulations
with open("regulations_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

documents = metadata["documents"]
index_to_docstore_id = metadata["index_to_docstore_id"]
docstore = InMemoryDocstore(documents)

# Initialize Cohere embeddings
embedding_function = CohereEmbeddings(cohere_api_key="ufQXgqBRTgMBLnQxg4egMEa14L7NMkMfeXAyDOfL",  model='embed-english-light-v3.0')

# Create FAISS vector store
#faiss_store = FAISS(embedding_function=embedding_function, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Initialize Cohere client
co = cohere.Client('ufQXgqBRTgMBLnQxg4egMEa14L7NMkMfeXAyDOfL')

# Streamlit UI
st.title("FinTrust Loan Application Analysis")
st.write("Analyze loan applications, assess risk, and ensure compliance.")

# Select loan ID
loan_id = st.selectbox("Select Loan ID", loan_data.index)
loan_info = loan_data.loc[loan_id]

# Display loan details
# Display loan details and description on the same horizontal level
col1, col2 = st.columns(2)
with col1:
    st.write("### Loan Information")
    st.write(loan_info[['loan_amnt', 'term', 'int_rate', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'addr_state']])
with col2:
    st.write("### Loan Description")
    st.write(loan_info['desc'])

# Risk assessment using the classification model
st.write("### Loan Payoff Risk")
# Encode categorical variables for the selected loan
for column in ['term', 'grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state']:
    loan_info[column] = label_encoders[column].transform([loan_info[column]])[0]

risk_prediction = model.predict([loan_info[['loan_amnt', 'term', 'int_rate', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'addr_state']]])
risk_score = "Low" if risk_prediction == 1 else "High"
st.write("Risk Level:", risk_score)

# Compliance and risk assessment using embeddings
st.write("### Compliance and Risk Assessment")

# Embed the loan description

# Search against regulations FAISS index
faiss_index = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)

# Search against regulations FAISS index
with st.spinner('Generating Summary'):
    similar_regs=faiss_index.search(loan_info['desc_clean'], k=3, search_type="similarity")
    compliance_results=[reg.page_content for reg in similar_regs]
    #relevant_text

    # Generate the response using Cohere chat model
    response_text = "\n\n".join(compliance_results)
    #chat_prompt = f"Based on the following regulatory documents, provide a compliance risk assessment for the loan description: {loan_info['desc']}\n\n{response_text}"


    chat_prompt = f"""
        Provide a compliance risk assessment for the following loan:

        Loan Amount: {loan_info['loan_amnt']}
        Term: {loan_info['term']}
        Interest Rate: {loan_info['int_rate']}
        Grade: {loan_info['grade']}
        Employment Length: {loan_info['emp_length']}
        Home Ownership: {loan_info['home_ownership']}
        Annual Income: {loan_info['annual_inc']}
        Purpose: {loan_info['purpose']}
        Address State: {loan_info['addr_state']}
        Description: {loan_info['desc']}

        Based on the following regulatory documents:
        {response_text}
        """
    
    cohere_response = cohere_response = co.chat(
        message=chat_prompt,
        max_tokens=200,
        connectors=[{"id": "web-search"}]

    )

# Display qualitative assessment
st.write("Compliance Risk Assessment:")
st.write(cohere_response.text)