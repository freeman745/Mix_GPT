import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
import pinecone
from langchain.vectorstores import Pinecone, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from utils import RealWeatherTool, StockPriceTool, StockPercentageChangeTool, StockGetBestPerformingTool


st.set_page_config(page_title="GPT: Online learning")
st.title("GPT: Online learning")
openai_api_key = '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'
pinecone_api_key = 'f2f8a0c9-36c3-4021-9cf0-5fc8633d7146'

pinecone.init(api_key=pinecone_api_key, environment='gcp-starter')

serpapi_api_key = 'b97301ece7e8c52122c591ff3494e48720cdf89f13ae9930fe27a89b2d59dc82'

#search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
search = DuckDuckGoSearchRun()

tools = []

tools.append(StockPriceTool())
tools.append(StockPercentageChangeTool())
tools.append(StockGetBestPerformingTool())


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif file.name.endswith(".txt") or file.name.endswith(".csv") or file.name.endswith(".json"):
            loader = UnstructuredFileLoader(temp_filepath)
        elif file.name.endswith(".doc") or file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_filepath)
        elif file.name.endswith(".html"):
            loader = UnstructuredHTMLLoader(temp_filepath)
            print("finish")
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

uploaded_files = st.sidebar.file_uploader(
    label="Upload files", accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, k=4)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.9, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

tools.append(
    Tool.from_function(
        name="state-of-union-qa",
        func=qa_chain.run,
        description="State of the Union QA - useful for when you need to ask questions about openweb3."
    )
)

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = agent.run(user_query, callbacks=[retrieval_handler, stream_handler])