import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import CondenseQuestionChatEngine
import tempfile
import shutil

# Title, Description, Styles
st.set_page_config(page_title="BuddyPDF Chatbot", layout="wide")

# Load CSS-file
with open("style.css", "r") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>Hi, I’m <span class="highlight">BuddyPDF</span> 🤖</h1>
  <p>Your friendly PDF assistant – ask me anything about your documents!</p>
</div>
""", unsafe_allow_html=True)

# Session Memory
if "chat_engine" not in st.session_state: st.session_state.chat_engine = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state: st.session_state.pdf_loaded = False
if "pdf_names" not in st.session_state: st.session_state.pdf_names = []

# PDF Upload
with st.sidebar:
   uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
      
if uploaded_file and not st.session_state.pdf_loaded:
	with st.spinner("Load and process PDF..."):
		# Create a temporary folder for the PDF
		temp_dir = tempfile.mkdtemp()
		temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
		with open(temp_pdf_path, "wb") as f: f.write(uploaded_file.read())

		# Read in PDF, create text splitter, embeddings & index
		documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
		splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)
		embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-l6-v2")
		index = VectorStoreIndex.from_documents(documents, transformations=[splitter], embed_model=embedding)

		# Prompt-Template
		prompt_template = PromptTemplate(
			template="""Here is the context: {context_str}
            Answer the following question based on the context above. Keep it short and precise.
				Question: {query_str}
				Answer:	"""
		)

		# Huggingface LLM
		llm = HuggingFaceInferenceAPI(
			model="mistralai/Mistral-7B-Instruct-v0.3",
			token=os.getenv("HF_TOKEN"),
			task="text-generation"
		)

		# Query Engine
		query_engine = index.as_query_engine(
			llm=llm,
			text_qa_template=prompt_template,
			similarity_top_k=2,
			response_mode="tree_summarize"
		)

		condense_prompt = PromptTemplate(template=(
        	"Given the following conversation history:\n"
        	"{chat_history}\n"
        	"And a follow up question:\n"
        	"{question}\n"
        	"Rephrase the follow up question to be a standalone question.")
      	)

		# Chat Engine
		st.session_state.chat_engine = CondenseQuestionChatEngine(
			query_engine=query_engine,
            memory=ChatMemoryBuffer.from_defaults(token_limit=2000),
            llm=llm,
            condense_question_prompt=condense_prompt,
			verbose=False
		)

    	# Note that PDF has been successfully processed
		st.session_state.pdf_loaded = True
		# Clean up temporary directory
		shutil.rmtree(temp_dir)
		st.toast("✅ PDF successfully processed!")
        
# -----------------------------------------------------------------------------------------------------------------------
# Chat UI
if not st.session_state.chat_engine:
   st.markdown("""
	<div class="app-header">
        🧐 No PDF uploaded yet
    	<p>Upload a PDF on the left to ask questions about its content.</p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.chat_engine:
   user_input = st.chat_input("Ask a question about your PDF...")

   if user_input:
      # Leerer Platzhalter für Denk-Animation
      thinking_placeholder = st.empty()
      with st.spinner("🧠 I'm Thinking..."):
         thinking_placeholder.markdown("""<div class='thinking'>🧠 I'm Thinking...</div>""", unsafe_allow_html=True)
         # Antwort vom LLM generieren
         response = st.session_state.chat_engine.chat(user_input)
      st.session_state.chat_history.append(("You", user_input))
      st.session_state.chat_history.append(("Assistant", str(response)))
      thinking_placeholder.empty()
   if st.session_state.chat_history:
      for speaker, message in st.session_state.chat_history:
        role_class = "user-bubble" if speaker == "You" else "bot-bubble"
        alignment_class = "user" if speaker == "You" else "bot"
        st.markdown(f"""
			<div class="chat-wrapper {alignment_class}">
  				<div class="chat-bubble {role_class}">
    				{message}
  				</div>
			</div>
			""", unsafe_allow_html=True)
        
# What is about Liabilities and Shareholders Equity at Apple?
# What is the main topic of the PDF document?