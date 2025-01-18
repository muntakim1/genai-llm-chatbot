import os
import time
from operator import itemgetter  # We need this for some wierd reason

import numpy as np
import pandas as pd
import streamlit as st
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings

# from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter

# Layout section: [Main Page]
st.set_page_config(
    "Q2O Chatbot",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon=":bar_chart:",
)
st.title("Q2O Chatbot :robot_face:")

# Layout section: [Sidebar]
st.sidebar.header("Power of LangChain and Ollama")
# st.sidebar.image("langchainbird.jpg", width=100)
st.sidebar.subheader(
    "Chatbot uses NOMIC Text Embeddings model for text embeddings", divider=True
)

# Now, add the widgets that come in the sidebar
with st.sidebar:
    st.header("Choose your LLM")
    llm_input = st.radio(
        "", ["Llama3.1", "GPT 3.5"], captions=["Meta Local LLM [LOCAL]", "OpenAI LLM"]
    )
    # This may be a right palce to get the LLM initailized and ready based on the LLM selection.
    if llm_input == "GPT 3.5":
        pass
    elif llm_input == "Llama3.1":
        # client = LocalLLM( api_base="http://localhost:11434/v1", model= "llama3" )
        client = Ollama(model="llama3.2")

    # Add the area for RAG
    st.header("Retrieval Augmented Generation")
    rag_input = st.text_area("Optionally enter Prompt Augmentation info :dart:")

# Layout section: [Gen AI] Docs and [Gen AI] Data tabs
docs_tab, data_tab, legal_docs_tab = st.tabs(
    ["[Gen AI] General/Docs", "[Gen AI] Data", "[GEN AI] Legal Document"]
)
with docs_tab:
    docs_tab_uploaded_file = st.file_uploader(
        "Choose a document to work on (pdf/text)....:file_folder:", type="pdf"
    )
    docs_tab_input_prompt = st.text_area(
        "How can I help you today? :robot_face:",
        height=68,
        max_chars=200,
        key="docs_tab_input_prompt_key",
    )

    # We will have two buttons in this tab and we want them one next to the other. Create a container to keep the buttons next to each other.
    col1, col2 = docs_tab.columns(2, gap="small")
    with col1:
        docs_tab_ans_qn_button = st.button(
            "Get answer from LLM",
            help="Click the button to call LLM to answer your question",
            key="docs_tab_open_ai_button_key",
        )
    with col2:
        docs_tab_process_docs_button = st.button(
            "Process Document using Local LLM",
            help="Click the button to call the Local LLM to answer your question/process your document",
            key="docs_tab_local_llm_button_key",
        )
with data_tab:
    data_tab_uploaded_file = st.file_uploader(
        "Choose a data file [xlsx] file....:file_folder:", type="csv"
    )
    data_tab_input_prompt = st.text_area(
        "How can I help you today? :robot_face:",
        height=68,
        max_chars=200,
        key="data_tab_input_prompt_key",
    )
    data_tab_tab_local_llm_button = st.button(
        "Call Local LLM",
        help="Click the button to call Local LLM to answer your question relating to the data",
        key="data_tab_tab_local_llm_button_key",
    )
with legal_docs_tab:
    contract_tab_uploaded_file = st.file_uploader(
        "Choose a contract document to work on (pdf/text)....:file_folder:", type="pdf"
    )
    legal_tab_uploaded_file = st.file_uploader(
        "Choose a legal document to work on (pdf/text)....:file_folder:", type="pdf"
    )
    legal_tab_tab_local_llm_button = st.button(
        "Process Legal document",
        help="Click the button to call Local LLM to answer your question relating to the data",
        key="legal_tab_tab_local_llm_button_key",
    )

#
# LOGIC: Processing Area
#
# Logic for "[Gen AI] Docs" tab. It has to do the two following things per the original design:
# There will be two buttons that will implement two different things:
#   1) [Get answer from LLM] button                 :
#           This will process the user input and call the LLM that is selected in the LLM area.
#           GPT-3.5 will not implment RAG while Llama2 does.
#   2) [Process Document using Local LLM] button    :
#           This is primarily for document processing using Local LLM Llama2.
#
if docs_tab_ans_qn_button:
    # Call GPT-3.5 to answer user's question.
    if llm_input == "GPT 3.5":
        if len(docs_tab_input_prompt) == 0:
            st.write("Enter a question to ask Chat GPT :warning:")
        else:
            # If we are here, we have "GPT 3.5" and the user has entered a question. We can call OpenAI.
            with docs_tab:
                with st.spinner("Calling Chat GPT...."):
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a very helpful assistant.",
                            },
                            {"role": "user", "content": docs_tab_input_prompt},
                        ],
                        max_tokens=200,
                    )
                    st.subheader("LLM Response", divider=True)
                    st.success(completion.choices[0].message.content)
    elif llm_input == "Llama2":
        if len(docs_tab_input_prompt) == 0:
            st.write("Enter a question to ask Local LLM :warning:")
        else:
            # If we are here, we have "Llama2" and the user has entered a question. We can call "Llama2".
            # Optionally use the RAG if provided.
            with docs_tab:
                st.write(
                    "Retrieval Augmentation Info used: ", rag_input
                )  # I am writing the RAG for informational purpose
                with st.spinner("Calling Local LLM...."):
                    prompt_template = """
                                        Answer the question below based on the context provided.
                                        Context: {context}
                                        Question: {question}
                                        """
                    parser = StrOutputParser()
                    retriever = ""
                    prompt = ChatPromptTemplate.from_template(prompt_template)

                    chain = (
                        {
                            "context": itemgetter("context"),
                            "question": itemgetter("question"),
                        }
                        | prompt
                        | client
                        | parser
                    )
                    resp = chain.invoke(
                        {"context": rag_input, "question": docs_tab_input_prompt}
                    )
                    docs_tab.subheader("LLM Response")
                    docs_tab.success(resp)
elif docs_tab_process_docs_button:
    # Call the Local LLM to process the document.
    with docs_tab:
        st.write("Button pressed:", docs_tab_process_docs_button)
        st.write("Rag Input: ", rag_input)
        st.write("Prompt Input: ", docs_tab_input_prompt)
        # try:
        # PDF/text summarizing area. LOAD-SPLIT-EMBEDD-{RAG}-PROMPT TEMPLATE-CHAIN-INVOKE
        if docs_tab_uploaded_file:
            uploaded_file_type = docs_tab_uploaded_file.type
            # st.write(uploaded_file_type_contract)
            # st.write(contract_tab_uploaded_file.name)

            if uploaded_file_type == "application/pdf":
                # Save the uploaded file temporarily
                with open(docs_tab_uploaded_file.name, "wb") as f:
                    f.write(docs_tab_uploaded_file.getbuffer())

                # Load the PDF using PyPDFLoader
                loader_contract = PyPDFLoader(docs_tab_uploaded_file.name)
                document = loader_contract.load()

                # Do something with documents_contract

                # Optionally, delete the file after processing
                os.remove(docs_tab_uploaded_file.name)

        # Split the file into chucks and store the chunks in pages
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        pages = text_splitter.split_documents(document)

        # Create embeddings object using "nomic-embed-text" model.
        embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        # Create the embeddings from the embeddings object.
        vector_store = DocArrayInMemorySearch.from_documents(
            pages, embedding=embeddings
        )

        # tab1.write("DocArrayInMemorySearch.from_documents(pages, embedding=embeddings) to create embeddings.")
        # tab1.write("Calling vector_store.similarity_search_with_score(query=rag_input, k=2)) for: " + rag_input)
        # Let's check here the accuracy of the retrieval for the RAG prompt.
        # tab1.write(vector_store.similarity_search_with_score(query=rag_input, k=2))

        # If a RAG prompt is provided, use the retriever to obtain the context from the embeddings.
        # Create retriever object.
        retriever = vector_store.as_retriever()
        # Get the context docs for the RAG input from the retriever object.
        context = retriever.invoke(rag_input)

        # Put togther the Prompt template that will be eventually passed to the llm.
        PROMPT_TEMPLATE = """
                        Answer the question below based on the context below. If you can't answer, reply I don't know.
                        Context: {context}
                        Question: {question}
                        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # We will be doing the prompt and parser only for the Local model so we are good adding the code just here
        prompt = prompt_template.format(context=context, question=docs_tab_input_prompt)

        def format_docs(context):
            return "\n\n".join([d.page_content for d in context])

        # Create the chain: {RAG | Prompt | Model | Parser}
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | client
            | StrOutputParser()
        )

        # Invoke the chain by passing the user prompt and the RAG input.
        if rag_input:
            resp = chain.invoke(
                {"context": rag_input, "question": docs_tab_input_prompt}
            )
            # Print the output from the LLM
            st.success(resp)
        else:
            resp = chain.invoke({"question": docs_tab_input_prompt})
            # Print the output from the LLM
            st.success(resp)
        # except AttributeError as e:
        #     print(str(e))
        #     st.error("Please Upload a file!")

elif data_tab_tab_local_llm_button:
    # Call the Local LLM to process the data.
    with data_tab:
        st.write("Button pressed:", data_tab_tab_local_llm_button)
        st.write("Prompt Input: ", data_tab_input_prompt)
        try:
            if data_tab_uploaded_file:
                # Save the uploaded file temporarily
                with open(data_tab_uploaded_file.name, "wb") as f:
                    f.write(data_tab_uploaded_file.getbuffer())
                global df
                # Load the csv into dataframe
                df = pd.read_csv(data_tab_uploaded_file.name, encoding="latin1")
                # Do something with documents_contract

                # Optionally, delete the file after processing
                os.remove(data_tab_uploaded_file.name)

            # THIS IS WHERE I LEFT OFF. TRYING TO MAKE THIS CALL WITH GPT 3.5 INSTEAD OF LLAMA2.
            # df = SmartDataframe("US_Superstore_data.csv", config={"llm": client})
            # df_llm =
            PROMPT_TEMPLATE = """
                            Answer the question below based on the context below.
                            Do not show any code, or title, or notes, just return output and answers.
                            If you can't answer, reply I don't know.
                            Context: {context}
                            Question: {question}
                            """
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = prompt_template | client | StrOutputParser()
            resp = chain.invoke(
                {
                    "context": df.to_dict(orient="records"),
                    "question": data_tab_input_prompt,
                }
            )

            st.success(resp)
        except ValueError:
            st.error("Please upload a csv file!")

elif legal_tab_tab_local_llm_button:
    with legal_docs_tab:
        st.write("Button pressed:", legal_tab_tab_local_llm_button)
        starttime = time.time()
        # PDF/text summarizing area. LOAD-SPLIT-EMBEDD-{RAG}-PROMPT TEMPLATE-CHAIN-INVOKE
        try:
            if contract_tab_uploaded_file:
                uploaded_file_type_contract = contract_tab_uploaded_file.type
                # st.write(uploaded_file_type_contract)
                # st.write(contract_tab_uploaded_file.name)

                if uploaded_file_type_contract == "application/pdf":
                    # Save the uploaded file temporarily
                    with open(contract_tab_uploaded_file.name, "wb") as f:
                        f.write(contract_tab_uploaded_file.getbuffer())

                    # Load the PDF using PyPDFLoader
                    loader_contract = PyPDFLoader(contract_tab_uploaded_file.name)
                    documents_contract = loader_contract.load()

                    # Do something with documents_contract

                    # Optionally, delete the file after processing
                    os.remove(contract_tab_uploaded_file.name)
            if legal_tab_uploaded_file:
                uploaded_file_type_legal = legal_tab_uploaded_file.type
                # st.write(uploaded_file_type_legal)
                # st.write(legal_tab_uploaded_file.name)

                if uploaded_file_type_legal == "application/pdf":
                    # Save the uploaded file temporarily
                    with open(legal_tab_uploaded_file.name, "wb") as f:
                        f.write(legal_tab_uploaded_file.getbuffer())

                    # Load the PDF using PyPDFLoader
                    loader_legal = PyPDFLoader(legal_tab_uploaded_file.name)
                    documents_legal = loader_legal.load()

                    # Do something with documents_legal

                    # Optionally, delete the file after processing
                    os.remove(legal_tab_uploaded_file.name)

            template = """
                You are an expert legal assistant. Do not made up answers, if you don't know simply answer don't know. 
                Do not show any title or notes, just provide the answer.

                Here is the contract document:
                {pdf_contract}

                Here is the terms and conditions document:
                {pdf_legal}

                Find out the violations of the contract document based on terms and condition document with references as a bullet points.
                """
            pdf_text_contract = " ".join(
                [doc.page_content for doc in documents_contract]
            )
            pdf_text_legal = " ".join([doc.page_content for doc in documents_legal])

            prompt = PromptTemplate(
                input_variables=["pdf_contract", "pdf_legal"],
                template=template,
            )
            formatted_prompt = prompt.format(
                pdf_contract=pdf_text_contract, pdf_legal=pdf_text_legal
            )
            # Replace "your_model_name" with the name of the Ollama model you want to use
            response = client(formatted_prompt)
            print("Took: ", time.time() - starttime)
            # Extract and print the response
            print(response)
            st.success(response)
        except AttributeError as e:
            st.error(str(e))
            st.error("Please upload a file")
