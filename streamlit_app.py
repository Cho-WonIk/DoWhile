import streamlit as st
import tiktoken
from loguru import logger
import os
import json

import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore
from langchain.schema import Document

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    # Streamlit 페이지 설정
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    # 페이지 제목 설정
    st.title("_Private Data :red[QA Chat]_ :books:")

    # 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 사이드바 위젯 설정
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx', 'json'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
        
    if process:
        # OpenAI API 키 확인
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # Firebase 인증서 설정 및 초기화
        cred = credentials.Certificate("auth.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        else:
            print("Firebase app is already initialized.")

        # Firestore 데이터베이스 클라이언트 가져오기
        db = firestore.client()

        # Firestore에 데이터 작성
        doc_ref = db.collection('test').document('testDocument')
        doc_ref.set({
            'api': openai_api_key,
            'grade': 1,
            'major': "AI 빅데이터 학과"
        })

        # Firestore에서 데이터 읽기
        doc_ref1 = db.collection('user').document('20211447')
        doc1 = doc_ref1.get()
        if doc1.exists:
            print("Document data:", doc1.to_dict())
        else:
            print("No such document!")

        # 업로드된 파일 처리 및 문서 리스트 생성
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        # 대화 체인 설정
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 

        st.session_state.processComplete = True
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    

        # 어시스턴트 답변을 채팅 메시지에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    # 텍스트의 토큰 수를 계산하는 함수
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)
def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        elif file_name.endswith('.json'):
            with open(file_name, "r", encoding="utf-8") as file:
                json_text = get_text_from_json(file)
            # JSON 데이터를 텍스트 덩어리로 변환하여 추가합니다.
            doc_list.append({"page_content": json_text, "metadata": {"source": file_name}})
        else:
            logger.warning(f"Unsupported file type: {file_name}")

        doc_list.extend(documents)
    return doc_list


def get_text_from_json(file):
    file_content = file.read().decode("utf-8")
    # JSON 문자열을 파싱
    data = json.loads(file_content)
    # JSON 데이터를 문자열로 변환 (필요에 따라 데이터를 평탄화하거나 적절히 조정할 수 있습니다)
    text = json.dumps(data, ensure_ascii=False, indent=4)
    return text

def get_text_chunks(text):
    # 텍스트를 청크로 나누는 함수
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    # 문서 청크를 벡터스토어로 변환하는 함수
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    # 대화 체인을 설정하는 함수
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()
