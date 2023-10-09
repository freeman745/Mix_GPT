# coding:utf-8
from flask import Flask, request, jsonify, stream_with_context, Response
from configparser import ConfigParser
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.agents import load_tools, initialize_agent, AgentType
from utils import RealWeatherTool, StockPriceTool, StockPercentageChangeTool, StockGetBestPerformingTool
from flask_cors import CORS
import urllib
import json
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



app = Flask(__name__)
CORS(app, methods=['GET','POST'], allow_headers='Content-Type')#defult orgin=*

# init
config_path = 'config.ini'
config = ConfigParser()
config.read(config_path)
# 储存上传的文件的地方
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    # 创建目录
    os.makedirs(app.config['UPLOAD_FOLDER'])
# 允许的文件类型的集合
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'doc', 'docx', 'json', 'csv'])
app.config['JSON_AS_ASCII'] = False
openai_api_key = config.get('Key', 'openai_api_key')
pinecone_api_key = config.get('Key', 'pinecone_api_key')
pinecone.init(api_key=pinecone_api_key, environment='asia-southeast1-gcp-free')

docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=4)
llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.9, streaming=True
            )

search = DuckDuckGoSearchRun()

tools = []

tools.append(
    Tool.from_function(
        name="Search",
        func=search.run,
        description="only useful for when you need to answer questions about news search 搜索 新闻 微博 and chatgpt can not answer. prompt必须包含搜索。"
    )
)

tools.append(RealWeatherTool())
tools.append(StockPriceTool())
tools.append(StockPercentageChangeTool())
tools.append(StockGetBestPerformingTool())

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
        # 记得结束后这里置true
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")
        self.tokens.append(token)

    def on_llm_end(self, response: str, **kwargs: any) -> None:
        self.finish = 1

    def on_llm_error(self, error: Exception, **kwargs: any) -> None:
        print(str(error))
        self.tokens.append(str(error))

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass


# health check
@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200


def allowed_file(filename):
    """
    判断传入的文件后缀是否合规。不合规返回False
    :param filename:
    :return: True or False
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# set route
@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    if allowed_file(filename):
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = {
            'code': 200,
            'result': 'success',
            'message': ''
        }
    else:
        result = {
            'code': 201,
            'result': 'failure',
            'message': ",".join(filename)
        }
        
    return result


@app.route("/savehtml", methods=["POST"])
def savehtml():
    try:
        url = request.form['url']
        html = urllib.request.urlopen(url).read()
    except Exception as e:
        result = {
            'code': 204,
            'result': 'failure',
            'message': 'url not reachable'
        }
    try:
        t_name = url.replace('/','_')+'.html'
        t_name = t_name.replace(':','_')
        filename = os.path.join(app.config['UPLOAD_FOLDER'],t_name)
        with open(filename, 'wb') as f:
            f.write(html)
        result = {
            'code': 200,
            'result': 'success',
            'message': ''
        }
    except Exception as e:
        print(e)
        result = {
            'code': 205,
            'result': 'failure',
            'message': 'html save error'
        }
    
    return result


@app.route("/delete", methods=["POST"])
def delete():
    delete_file = request.form['filename']
    delete_file = os.path.join(app.config['UPLOAD_FOLDER'], delete_file)
    if os.path.exists(delete_file):
        os.remove(delete_file)
        result = {
            'code': 200,
            'result': 'success',
            'message': ''
        }
    else:
        result = {
            'code': 203,
            'result': 'failure',
            'message': request.form['filename']+' not exist'
        }
        
    return result
    

@app.route("/learn", methods=["POST"])
def learn():
    try:
        for folder_name, subfolders, filenames in os.walk('uploads/'):
            for filename in filenames:
                temp_filepath = os.path.join('uploads/', filename)
                if temp_filepath.endswith(".pdf"):
                    loader = PyPDFLoader(temp_filepath)
                elif temp_filepath.endswith(".txt") or temp_filepath.endswith(".csv") or temp_filepath.endswith(".json"):
                    loader = UnstructuredFileLoader(temp_filepath)
                elif temp_filepath.endswith(".doc") or temp_filepath.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(temp_filepath)
                elif temp_filepath.endswith(".html"):
                    loader = UnstructuredHTMLLoader(temp_filepath)
            docs.extend(loader.load())

        # Split documents
        global splits
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        index_name = 'test'
        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=index_name,
                metric='cosine',
                dimension=1536  
                )
        global embeddings, vectordb, retriever, qa_chain, agent
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = Pinecone.from_documents(splits, embeddings, index_name=index_name)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True
            )
        
        tools.append(
            Tool.from_function(
                name="state-of-union-qa",
                func=qa_chain.run,
                description="State of the Union QA - useful for when you need to ask questions about the Openweb3."
            )
        )

        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory=memory, verbose=True)
        
        result = {
            'code': 200,
            'result': 'success',
            'message': ''
        }
    except Exception as e:
        result = {
            'code': 202,
            'result': 'failure',
            'message': e
        }


    return result


@app.route('/predict', methods=['POST'])
def predict():
    handler = ChainStreamHandler()
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        try:
            json_data = request.json
            query = json_data['message']
            stream = json_data['stream']
        except Exception as e:
            print("1")
            print(e)
            result = {"message":"request format error"}
            return jsonify(result)
    else:
        try:
            query = request.form['message']
            stream = request.form['stream']
        except Exception as e:
            print("2")
            print(e)
            result = {"message":"content-type not support"}
            return jsonify(result)

    try:
        if stream == 0:
            result = agent.run({"input": query})
            result = {"message":result}
            return jsonify(result)
        else:
            result = agent.run({"input": query}, callbacks=[handler])
    except Exception as e:
        print("3")
        print(e)
        result = {"message":"Sorry I can't answer this question."}
        return jsonify(result)
    
    return Response(handler.generate_tokens(), mimetype='text/plain', headers={'X-Accel-Buffering': 'no'})


@app.route('/dbsearch', methods=['POST'])
def dbsearch():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        try:
            json_data = request.json
            query = json_data['message']
        except Exception as e:
            print("1")
            print(e)
            result = {"message":"request format error"}
            return jsonify(result)
    else:
        try:
            query = request.form['message']
        except Exception as e:
            print("2")
            print(e)
            result = {"message":"content-type not support"}
            return jsonify(result)
    result = vectordb.similarity_search_with_score(query)
    result = result[0]
    if result[1] < 0.8:
        result = {"message":"Sorry I can't answer this question."}
    else:
        result = {"message":result[0].page_content}
    return jsonify(result)


if __name__ == '__main__':
    ip = config.get('Server', 'ip')
    port = config.get('Server', 'port')
    app.config['JSON_AS_ASCII'] = False
    app.run(host=ip, port=port)