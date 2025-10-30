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

# 로그 디렉토리 자동 생성
os.makedirs("logs", exist_ok=True)
from datetime import datetime
log_path = Path("logs") / "incident_history.json"

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
st.title("⚙️ AI 기반 장애 대응 상황창 어시스턴트")


# ==============================================
# 🧭 탭 상태 관리
# ==============================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab1"  # 초기: tab1 활성화
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================
# 🎨 탭 버튼
# ==============================================
col1, col2 = st.columns(2)
with col1:
    if st.button("💬 상황창", use_container_width=True):
        st.session_state.active_tab = "tab1"
with col2:
    if st.button("📝 보고서", use_container_width=True):
        st.session_state.active_tab = "tab2"

# ==============================================
# 🎨 CSS 스타일
# ==============================================
st.markdown("""
<style>
.tab-div {
    padding: 30px;
    border-radius: 12px;
    height: 75vh;
    overflow-y: auto;
}
.tab1-div {
    /* background-color: #f5f0e1;   베이지 */
    height: 10px;
}
.tab2-div {
    /*  background-color: #d9d9d9;  그레이 */
    height: 10px;
}
.chat-bubble {
    max-width: 75%;
    padding: 0.8rem 1rem;
    border-radius: 1rem;
    margin: 0.4rem 0;
    word-wrap: break-word;
}
.user-msg {
    background-color: #E1E9F6;
    margin-left: auto;
    text-align: left;
}
.assistant-msg {
    background-color: #F1F0F0;
    margin-right: auto;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

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

    [현재 보고유형]
    {report_type}
                                              
    [장애 대응 템플릿]
    {template}

    [관련 문서 내용]
    {context}

    [상황창 메시지]
    {question}
    """)

    # 5️⃣ 체인 생성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "report_type": RunnablePassthrough()}
        | prompt
        | llm
    )
except Exception as e:
    st.error(f"LangChain 초기화 중 오류 발생: {e}")
    st.stop()


# ==============================================
# 💬 TAB 1 - 상황창 (Chat 기능)
# ==============================================
if st.session_state.active_tab == "tab1":
    st.markdown("<div class='tab-div tab1-div'>", unsafe_allow_html=True)
    st.markdown("### 🟤 상황창 (활성화됨)")

    chat_container = st.container()



    user_input = st.chat_input("🗨 상황창 메시지를 입력하세요 (예: ERP 로그인 오류, DB 접속 불가, 복구 진행 중...)")

    # ✅ 대화 출력
    with chat_container:
        for msg in st.session_state.messages:
            #st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)
            role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
            st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    if user_input:
        # 사용자 메시지
        st.markdown(
            f"<div class='chat-bubble user-msg'>{user_input.replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True
        )
        #st.chat_message("user").markdown(user_input.replace("\n", "<br>"), unsafe_allow_html=True)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        if "복구 완료" in user_input or "정상화" in user_input:
            report_type = "최종보고"
        else:
            report_type = "중간보고"


        # ✅ 이전 대화 포함
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])
        query_with_context = f"{chat_history}\nuser: {user_input}"

        with st.spinner("AI가 "+report_type+"를 작성중입니다..."):
            try:
                rag_input = {"question": query_with_context, "template": incident_template, "report_type": report_type}

                retriever_output = retriever.invoke(query_with_context)
                context_texts = "\n".join([d.page_content for d in retriever_output]) if retriever_output else ""

                prompt_input = {
                    "context": context_texts,
                    "question": query_with_context,
                    "template": incident_template,
                    "report_type": report_type
                }

                # 🔹 LLM 호출
                raw_llm_response = llm.invoke(prompt.format(**prompt_input))
                ai_output = str(raw_llm_response)

                print("\n🔍 input: "+ prompt.format(**prompt_input))
                print("\n🔍 output: "+ ai_output)
        

                # 🔹 content 부분만 추출
                match = re.search(r"content='(.*?)' additional_kwargs=", ai_output, re.DOTALL)
                if match:
                    content_text = match.group(1).strip()
                else:
                    content_text = ai_output.strip()

                # 🔹 \n → 줄바꿈 복원
                content_text = content_text.replace("\\n", "\n")

            
                st.session_state.messages.append({"role": "assistant", "content": content_text})
                st.chat_message("assistant").markdown(content_text)

            

                # 🔹 로그 저장
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "input": user_input,
                    "output": content_text
                }

                with open(log_path, "a", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False)
                    f.write("\n")

            except Exception as e:
                st.error(f"RAG 처리 중 오류 발생: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
# 📝 TAB 2 - 보고서 (단순 표시용)
# ==============================================
elif st.session_state.active_tab == "tab2":
    st.markdown("<div class='tab-div tab2-div'>", unsafe_allow_html=True)
    st.markdown("### ⚪ 보고서 (활성화됨)")
    st.write("이 영역은 보고서 표시용 공간입니다.")
    st.markdown("</div>", unsafe_allow_html=True)
