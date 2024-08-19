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



# 학생 정보 초기화
if 'student_info' not in st.session_state:
    st.session_state.student_info = {
        'grade': 1,
        'major': "컴퓨터공학과",
        'student_id': "20001234",
        'student_career': "미정",
        'general_credits': 0,
        'major_credits': 0
    }

# 페이지 1: 정보 입력
def page_one():
    st.subheader("학생 정보")
    with st.container():
        student_info_col, grade_col = st.columns(2)

        with student_info_col:
            st.session_state.student_info['student_id'] = st.text_input("학번", value=st.session_state.student_info['student_id'])
            st.session_state.student_info['major'] = st.text_input("학과", value=st.session_state.student_info['major'])
            st.session_state.student_info['grade'] = st.number_input("학년", value=st.session_state.student_info['grade'], format="%d")
            st.session_state.student_info['student_career'] = st.selectbox("희망 직종 선택", ["미정", "프론트엔드", "백엔드", "임베디드", "보안", "인공지능"], index=["미정", "프론트엔드", "백엔드", "임베디드", "보안", "인공지능"].index(st.session_state.student_info['student_career']))

        with grade_col:
            grad_info = st.text_area("이수 정보 입력(입력방법: 과목명, 구분, 학점)", height=375)  # 이수한 과목 및 학점 입력

            # 입력된 데이터를 줄별로 분리
            lines = grad_info.split('\n')
            general_credits, major_credits = 0, 0

            # 각 줄을 처리
            for line in lines:
                if line.strip():  # 공백이 아닌 줄만 처리
                    parts = line.split(',')  # 쉼표로 데이터 분리
                    if len(parts) == 3:  # 정확히 세 부분으로 나뉘는지 확인
                        category = parts[1].strip()  # 구분 (교양 혹은 전공)
                        credits = int(parts[2].strip())  # 학점

                        # 교양과 전공 학점 합산
                        if category == "교양":
                            general_credits += credits
                        elif category == "전공":
                            major_credits += credits

            st.session_state.student_info['general_credits'] = general_credits
            st.session_state.student_info['major_credits'] = major_credits

        if st.button("저장"):
            
            # Firestore 데이터베이스 클라이언트 가져오기
            db = firestore.client()
            uploaded_files = [
            "2024학년도 2학기 컴공강의.pdf", 
            "게임프로그래밍.pdf",
            "논리회로및실습.pdf",
            "데이터베이스론.pdf",
            "리눅스시스템.pdf",
            "분산.객체시스템설계.pdf",
            "운영체제론.pdf",
            "웹프로그래밍.pdf",
            "이산구조론.pdf",
            "자바프로그래밍.pdf",
            "정보컴퓨터교과교육론.pdf",
            "정보컴퓨터교과교재연구및지도법.pdf",
            "지능정보시스템설계.pdf",
            "캡스톤디자인(소프트웨어공학).pdf",
            "컴퓨터알고리즘.pdf",
            "컴퓨터프로그래밍기초.pdf",
            "프론트엔드웹디자인.pdf"]
            openai_api_key = st.secrets["openai_api_key"]
            # OpenAI API 키 확인
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()

            # 업로드된 파일 처리 및 문서 리스트 생성
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vetorestore = get_vectorstore(text_chunks)

            # 대화 체인 설정
            st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 

            st.session_state.processComplete = True

            chain = st.session_state.conversation
 
            # 컬렉션의 모든 문서를 가져옴
            collection_ref = db.collection('langchain')
            docs = collection_ref.stream()

            # 숫자로 된 필드 이름을 수집하기 위한 리스트
            documents = []
            field_numbers = 0

            for doc in docs:
                doc_data = doc.to_dict()
                documents.append(doc_data)
                field_numbers = max(doc['id'] for doc in documents)


            # 가장 높은 숫자를 찾아 다음 필드 이름 생성
            if field_numbers != 0:
                id = field_numbers + 1
            else:
                id = 1  # 숫자 필드가 없는 경우 1로 시작


            query = st.session_state.student_info['major'] + st.session_state.student_info['grade'] + "학년 " + st.session_state.student_info['student_career'] + "꿈인 학생이 들을만한 강의목록만 출력해줘"
            result = chain({"question": query})
            response = result['answer']
            # Firestore에 데이터 작성
            doc_ref = collection_ref.document(str(id))
            doc_ref.set({
                'id': id,
                'answer': query,
                'response': response
            })
            st.session_state.conversation2 = response

            query2 = st.session_state.student_info['major'] + st.session_state.student_info['grade'] + "학년 " + st.session_state.student_info['student_career'] + "꿈인 학생이 들을만한 강의추천이유 출력해줘"
            result2 = chain({"question": query2})
            response2 = result2['answer']
            st.session_state.conversation3 = response2


            st.session_state.page = "AI 컨설팅"


# 페이지 2: AI 컨설팅
def page_two():

    st.subheader("AI 컨설팅")

    response = st.session_state.conversation2
    response2 = st.session_state.conversation3

    student_id = int(st.session_state.student_info['student_id'])
    if student_id >= 20160000:
        required_general_credits = "/46 미만"
        required_major_credits = "/42 이상"
    else:
        required_general_credits = "/40 미만"
        required_major_credits = "/46 이상"

    
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.write("희망 직종 : ", st.session_state.student_info['student_career'])
        
        with col2:
            st.write("교양 이수 최소 학점 : 12학점 이상")
            st.write("교양 학점 : ", st.session_state.student_info['general_credits'], required_general_credits)
            st.write("전공 학점 : ", st.session_state.student_info['major_credits'], required_major_credits)

    st.write("---")

    with st.container():
        col2_1, col2_2 = st.columns(2)

        with col2_1:
            st.text_area("추천 강좌", response, height=200)
        
        with col2_2:
            st.text_area("AI의견", response2, height=200)


def main():
    if 'page' not in st.session_state:
        st.session_state.page = "정보 입력"

   

    # 사이드바 위젯 설정
    with st.sidebar:
        
        
        #process = st.button("Process")

        page = st.sidebar.radio("MENU", ["정보 입력", "AI 컨설팅"], index=["정보 입력", "AI 컨설팅"].index(st.session_state.page))

    if page == "정보 입력":
        page_one()
    elif page == "AI 컨설팅":
        page_two()


    # Firebase 인증서 설정 및 초기화
    cred = credentials.Certificate("auth.json")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    else:
        print("Firebase app is already initialized.")
        
    

def tiktoken_len(text):
    # 텍스트의 토큰 수를 계산하는 함수
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc  # doc 객체의 이름을 파일 이름으로 사용

        if '.pdf' in doc:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

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
    pre_context = (
        ""
    )
        
    system_prompt = (
        "당신은 컴퓨터 공학 분야에 대한 깊은 지식을 갖춘 유능한 어시스턴트입니다. "
        "사용자의 질문에 대해 유용하고 도움이되는 정확하며 유용한 답변을 제공해야 합니다. "
        "친절하고 전문적인 톤을 유지하면서 응답하세요."
    )
    # ).format(pre_context=pre_context)
        
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, 
        model_name='gpt-4o', 
        temperature=0
        # prompt=system_prompt
        )
    
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