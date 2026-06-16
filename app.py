import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# Load API key from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit page settings
st.set_page_config(
    page_title="SmartStudy AI", 
    page_icon="📚",
    layout="wide"
)

st.title("📚 SmartStudy AI — PDF Study Material Chatbot")
st.write("Upload your study material PDF and ask questions from it.")


# Check API key
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please add it in your .env file.")
    st.stop()


# Gemini model for answer generation
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)


# Local HuggingFace embeddings
# This avoids Google embedding model 404 error
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Function to process uploaded PDF
def process_pdf(uploaded_file):
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Create vector database using local embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, chunks


# Prompt for normal PDF question answering
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful study assistant.

Answer the question using only the given PDF context.
Use very simple English/Hinglish language.
If the answer is not present in the PDF, say:
"I could not find this in the uploaded PDF."

PDF Context:
{context}

Question:
{question}

Answer:
"""
)


# Prompt for 10/20 marks exam answer
exam_prompt = PromptTemplate(
    input_variables=["context", "question", "marks"],
    template="""
You are an exam answer writing assistant.

Use only the given PDF context and create a {marks} marks answer.

Answer format:
1. Definition
2. Explanation
3. Important points
4. Example
5. Diagram/Flowchart if needed
6. Conclusion

Use simple and easy English/Hinglish language.

PDF Context:
{context}

Question:
{question}

Exam Answer:
"""
)


# Prompt for summary
summary_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
Summarize the following study material in simple language.

Include:
1. Short summary
2. Key points
3. Important terms
4. Easy revision notes

Study Material:
{context}

Summary:
"""
)


# Prompt for important questions
important_questions_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
From the given study material, predict important exam questions.

Divide them into:
1. Very Important Questions
2. 10 Marks Questions
3. 20 Marks Questions
4. Short Notes
5. Viva Questions

Use simple language.

Study Material:
{context}

Important Questions:
"""
)


# Sidebar
st.sidebar.header("⚙️ Options")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Ask Question",
        "10/20 Marks Answer",
        "Summary + Key Points",
        "Important Questions"
    ]
)


# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# Process PDF
if uploaded_pdf is not None:
    st.sidebar.write(f"Selected PDF: **{uploaded_pdf.name}**")

    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF... Please wait."):
            try:
                vector_store, chunks = process_pdf(uploaded_pdf)

                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.pdf_name = uploaded_pdf.name

                st.success("PDF processed successfully!")

            except Exception as e:
                st.error("Error while processing PDF.")
                st.write(e)


# Main app logic
if st.session_state.vector_store is not None:

    st.info(f"Current PDF: {st.session_state.pdf_name}")

    if mode == "Ask Question":
        st.subheader("💬 Ask a Question from Your PDF")

        question = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding answer from PDF..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(
                            question,
                            k=4
                        )

                        context = "\n\n".join(
                            [doc.page_content for doc in docs]
                        )

                        final_prompt = qa_prompt.format(
                            context=context,
                            question=question
                        )

                        response = llm.invoke(final_prompt)
                        st.markdown(response.content)

                    except Exception as e:
                        st.error("Error while generating answer.")
                        st.write(e)


    elif mode == "10/20 Marks Answer":
        st.subheader("📝 Generate Exam-Style Answer")

        question = st.text_input("Enter exam question:")
        marks = st.selectbox("Select marks:", ["10", "20"])

        if st.button("Generate Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating exam-style answer..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(
                            question,
                            k=5
                        )

                        context = "\n\n".join(
                            [doc.page_content for doc in docs]
                        )

                        final_prompt = exam_prompt.format(
                            context=context,
                            question=question,
                            marks=marks
                        )

                        response = llm.invoke(final_prompt)
                        st.markdown(response.content)

                    except Exception as e:
                        st.error("Error while generating exam answer.")
                        st.write(e)


    elif mode == "Summary + Key Points":
        st.subheader("📌 Summary + Key Points")

        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                try:
                    # Taking first 15 chunks to avoid very large input
                    all_text = "\n\n".join(
                        [doc.page_content for doc in st.session_state.chunks[:15]]
                    )

                    final_prompt = summary_prompt.format(
                        context=all_text
                    )

                    response = llm.invoke(final_prompt)
                    st.markdown(response.content)

                except Exception as e:
                    st.error("Error while generating summary.")
                    st.write(e)


    elif mode == "Important Questions":
        st.subheader("⭐ Important Exam Questions")

        if st.button("Predict Important Questions"):
            with st.spinner("Predicting important questions..."):
                try:
                    # Taking first 15 chunks to avoid very large input
                    all_text = "\n\n".join(
                        [doc.page_content for doc in st.session_state.chunks[:15]]
                    )

                    final_prompt = important_questions_prompt.format(
                        context=all_text
                    )

                    response = llm.invoke(final_prompt)
                    st.markdown(response.content)

                except Exception as e:
                    st.error("Error while predicting questions.")
                    st.write(e)

else:
    st.info("Please upload and process a PDF from the sidebar first.")





    