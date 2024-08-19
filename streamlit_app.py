import streamlit as st
import tiktoken
from loguru import logger
import os
import json

import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.schema import Document
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
        uploaded_files = ["2024학년도 2학기 컴공강의.pdf"]
        openai_api_key = st.secrets["openai_api_key"]
        process = st.button("Process")
        
    process = True
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


    # 채팅 메시지 상태 초기화
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 메시지 기록 초기화
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    # 사용자 입력 받아 query 변수에 저장
    if query := st.chat_input("질문을 입력해주세요."):

        # 사용자가 입력한 질문을 st.session_state.messages 리스트에 추가
        st.session_state.messages.append({"role": "user", "content": query})

        # 사용자 메시지 화면에 표시
        with st.chat_message("user"):
            st.markdown(query) # 사용자 질문 넣기

        # 답변 메시지 화면에 표시
        with st.chat_message("assistant"):
            # 이전에 설정한 대화 체인을 가져옴
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                # 대화 체인에 사용자의 질문을 전달
                result = chain({"question": query})

                # OpenAI API 호출에 대한 콜백을 설정
                with get_openai_callback() as cb:
                    # 대화 체인에서 반환된 대화 기록을 세션 상태에 저장
                    st.session_state.chat_history = result['chat_history']

                # 답변 가져옴
                response = result['answer']
                # 답변에 사용된 문서들을 가져옴
                source_documents = result['source_documents']

                st.markdown(response) # 답변 표시
                with st.expander("참고 문서 확인"):
                    # 참고 문서 3개까지 가져옴
                    # 각 문서의 메타데이터에서 'source'를 추출하여 문서의 출처를 표시하고, 문서의 내용을 help 매개변수로 추가
                    # st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    # st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    # st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    for doc in source_documents:
                        st.markdown(doc.metadata.get('source', 'No source available'), help=doc.page_content)
                    

        # 어시스턴트 답변을 채팅 메시지에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    # 텍스트의 토큰 수를 계산하는 함수
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(uploaded_files):
    doc_list = []
    
    for uploaded_file in uploaded_files:
        # 파일 이름을 가져오기
        file_name = uploaded_file.name
        
        # 파일을 저장하기
        with open(file_name, "wb") as file:
            file.write(uploaded_file.getvalue())
            logger.info(f"Uploaded {file_name}")
        
        # 파일 확장자에 따라 적절한 로더 사용
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        elif file_name.endswith('.json'):
            try:
                with open(file_name, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    # JSON 데이터가 리스트인지 확인
                    if isinstance(json_data, list):
                        # JSON 데이터가 리스트인 경우
                        documents = [Document(page_content=json.dumps(item)) for item in json_data]
                    elif isinstance(json_data, dict):
                        # JSON 데이터가 딕셔너리인 경우
                        documents = [Document(page_content=json.dumps(json_data))]
                    else:
                        # JSON 데이터가 리스트나 딕셔너리가 아닌 경우
                        logger.error(f"Unsupported JSON format in file: {file_name}")
                        continue  # 다음 파일로 이동

                    logger.info(f"Loaded JSON data from {file_name}")

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON file {file_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading JSON file {file_name}: {e}")

        print("문서 출력", documents)
        doc_list.extend(documents)
    
    return doc_list


def get_text_chunks(text):
    # 텍스트를 청크로 나누는 함수
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    # 문서 청크를 벡터스토어로 변환하는 함수
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    # 대화 체인을 설정하는 함수
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    print(conversation_chain)
    return conversation_chain

if __name__ == '__main__':
    main()