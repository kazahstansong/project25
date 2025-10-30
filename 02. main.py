# ==============================================
# ⚙️ AI 기반 장애 대응 상황창 어시스턴트 (RAG + Azure Search)
# ==============================================
import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import json
import warnings
import re
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# 환경 설정
# ---------------------------------------------------------------------
load_dotenv()

# 🔹 OpenAI 설정
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# 🔹 Azure Search 설정
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# 🔹 Embedding 설정
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")

# 템플릿 파일 로드
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"
template_path = data_dir / "incident_template.txt"

if not template_path.exists():
    st.error(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
    st.stop()

with template_path.open("r", encoding="utf-8") as f:
    incident_template = f.read()

# 🔧 디버깅용 헬퍼 함수
def debug_log(title, value):
    print(f"\n===== 🔍 {title} =====")
    try:
        if isinstance(value, dict):
            print(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            print(str(value)[:1000])  # 너무 길면 1000자까지만 표시
    except Exception as e:
        print(f"(⚠️ 디버깅 출력 중 오류: {e})")
    print("========================\n")

st.set_page_config(page_title="AI 기반 장애 대응 상황창 어시스턴트", layout="wide")

# 🔧 스타일 커스터마이징 (ChatGPT 느낌)
st.markdown("""
<style>
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        border-top: 1px solid #ddd;
        padding: 0.5rem 1rem;
    }
    .stChatMessage {
        max-width: 80%;
        padding: 0.75rem 1rem;
        border-radius: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #DCF8C6;
        margin-left: auto;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #F1F0F0;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ AI 기반 장애 대응 상황창 어시스턴트")
st.markdown("시스템 장애 발생 시, AI가 매뉴얼을 참고해 중간보고를 제안합니다.")
st.markdown("---")

# LangChain 구성 (RAG)
try:
    # 1️⃣ Embedding 생성
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        model=AZURE_EMBEDDING_MODEL
    )
    print("🔧 AZURE_EMBEDDING_MODEL:", os.getenv("AZURE_EMBEDDING_MODEL"))
    print("🔧 AZURE_EMBEDDING_DEPLOYMENT:", os.getenv("AZURE_EMBEDDING_DEPLOYMENT"))

    # 2️⃣ Azure AI Search 연결
    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embeddings,
        vector_field="content_vector",
        text_field="content"
    )

    print("🔗 현재 연결된 Azure 인덱스:", AZURE_SEARCH_INDEX)
    print("🔗 현재 연결된 Azure 엔드포인트:", AZURE_SEARCH_ENDPOINT)

    print("🔍 AzureSearch 테스트 중...")
    

    # 📄 Retriever 생성
    retriever = vector_store.as_retriever(
        search_type="hybrid",
        k=5
    )

    # 3️⃣ LLM 초기화
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.2
    )

    # 4️⃣ 프롬프트 정의 (템플릿 + 문서 기반)
    prompt = ChatPromptTemplate.from_template("""
    너는 시스템 장애 대응을 지원하는 어시스턴트야.
    아래 매뉴얼과 템플릿을 참고해 상황창 대화를 중간보고 형식으로 정리해줘.

    [장애 대응 템플릿]
    {template}

    [관련 문서 내용]
    {context}

    [상황창 메시지]
    {question}
    """)

    # 5️⃣ 체인 생성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
except Exception as e:
    st.error(f"LangChain 초기화 중 오류 발생: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 세션 관리
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# ✅ rerun 방지용: Streamlit이 rerun될 때 messages 유지
messages = st.session_state.get("messages", [])

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# ✅ 채팅 영역 (스크롤 가능)
chat_container = st.container()

# ✅ 입력창 하단 고정
user_input = st.chat_input("🗨 상황창 메시지를 입력하세요 (예: ERP 로그인 오류, DB 접속 불가, 복구 진행 중...)")

# ✅ 이전 대화 출력
with chat_container:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

# ✅ 새 메시지 입력 처리
if user_input:
    # 사용자 메시지
    st.chat_message("user").markdown(user_input.replace("\n", "<br>"), unsafe_allow_html=True)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # ✅ 이전 대화 포함
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])
    query_with_context = f"{chat_history}\nuser: {user_input}"

    with st.spinner("AI가 중간보고를 작성 중입니다..."):
        try:
            rag_input = {"question": query_with_context, "template": incident_template}

            retriever_output = retriever.invoke(query_with_context)
            context_texts = "\n".join([d.page_content for d in retriever_output]) if retriever_output else ""

            prompt_input = {
                "context": context_texts,
                "question": query_with_context,
                "template": incident_template
            }

            # 🔹 LLM 호출
            raw_llm_response = llm.invoke(prompt.format(**prompt_input))
            ai_output = str(raw_llm_response)

            # 🔹 content 부분만 추출
            match = re.search(r"content='(.*?)' additional_kwargs=", ai_output, re.DOTALL)
            if match:
                content_text = match.group(1).strip()
            else:
                content_text = ai_output.strip()

            # 🔹 \n → 줄바꿈 복원
            content_text = content_text.replace("\\n", "\n")

            # 🔹 대화 추가 및 출력
            st.session_state.messages.append({"role": "assistant", "content": content_text})
            st.chat_message("assistant").markdown(content_text)

        except Exception as e:
            st.error(f"RAG 처리 중 오류 발생: {e}")





