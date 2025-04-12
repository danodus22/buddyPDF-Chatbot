# BuddyPDF Chatbot

BuddyPDF is a local, interactive PDF chatbot that allows users to upload PDF documents and ask questions about their content. It combines PDF parsing, semantic search, and LLM-powered response generation.

## 🚀 Technologies & Techniques

This project uses a range of modern tools and patterns:

- **[Streamlit](https://streamlit.io/):** For building the interactive user interface.
- **[LlamaIndex](https://www.llamaindex.ai/):** To load, index, and query PDF documents using embeddings and a custom query engine.
- **[Sentence Splitting](https://developer.mozilla.org/en-US/docs/Glossary/Sentence):** Used for chunking document text into manageable segments.
- **[HuggingFaceInferenceAPI](https://huggingface.co/docs/huggingface_hub/index):** Connects to Hugging Face-hosted models like `mistralai/Mistral-7B-Instruct-v0.3`.
- **[Prompt Templates](https://docs.llamaindex.ai/en/stable/module_guides/prompts/custom_prompts/):** Define how user input and responses are structured.
- **[ChatMemoryBuffer](https://docs.llamaindex.ai/en/stable/module_guides/memory/chat_memory/):** Stores prior interactions to support contextual dialogue.
- **[CondenseQuestionChatEngine](https://docs.llamaindex.ai/en/stable/module_guides/chat_engines/condense_question_chat_engine/):** Reformulates follow-up questions into standalone questions.

## 📦 Libraries & Models

Some noteworthy technologies used in this project:

- [`sentence-transformers/all-MiniLM-l6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): Lightweight embedding model for semantic similarity.
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3): Compact, performant open-weight language model.
- **[`tempfile`](https://docs.python.org/3/library/tempfile.html):** Secure handling of user-uploaded files without permanent storage.

## 🗂️ Project Structure

```txt
.
├── .env
├── buddychat.py
├── style.css
```

### Directory Breakdown

- [`buddychat.py`](./app2.py): Main application containing frontend logic, document processing, and chatbot integration.
- [`style.css`](./style.css): Custom CSS for styling the Streamlit interface.

## 🔗 Fonts and Styling Notes

The app uses a local CSS file. If specific fonts are referenced, they will be defined inside [`style.css`](./style.css).
