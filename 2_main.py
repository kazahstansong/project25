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
import random

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


# 🔹 좌 / 메인 / 우 구조 정의
left_col, main_col, right_col = st.columns([0.05, 0.7, 0.25], gap="large")




st.set_page_config(page_title="AI 기반 장애 대응 상황창 어시스턴트", layout="wide")
 
# ==============================================
# 🧭 탭 상태 관리
# ==============================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab1"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "reports" not in st.session_state:
    st.session_state.reports = []

# 🔹 좌측 패널 구성
with st.sidebar:
    st.header("🧭 메뉴")
    st.write("아래 탭을 선택하세요.")
    if st.button("💬 상황창 분석", use_container_width=True):
        st.session_state.active_tab = "tab1"
    if st.button("📝 보고서", use_container_width=True):
        st.session_state.active_tab = "tab2"

    st.markdown("---")
    st.markdown("### ⚙️ 빠른 제어")

    if st.button("🧹 대화 초기화", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("📝 보고서 초기화", use_container_width=True):
        st.session_state["reports"] = []
        st.rerun()

    if st.button("🔄 전체 새로고침", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 설정")
    st.text_input("Azure Search 인덱스", AZURE_SEARCH_INDEX, disabled=True)
    st.text_input("모델 배포 이름", AZURE_OPENAI_DEPLOYMENT, disabled=True)
    st.markdown("---")



    st.caption("© 2025 AI 기반 장애 대응 어시스턴트")

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
[data-testid="stSpinnerOverlay"] {
    background-color: rgba(255, 255, 255, 1) !important;
    opacity: 1 !important;
}
html, body, [class*="css"]  {
    font-size: 11px;   /* 기본값 약 16px → 14px로 축소 */
}
.right-panel {
    position: fixed;
    top: 80px;
    right: 0;
    width: 22%;
    height: 90%;
    padding: 20px;
    background-color: #f8f9fa;
    border-left: 2px solid #ddd;
    overflow-y: auto;
    box-shadow: -2px 0 8px rgba(0,0,0,0.1);
}
.main-content {
    margin-right: 25%;
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
    아래 매뉴얼과 템플릿을 참고해 상황창 대화를 중간보고/최종보고 형식으로 정리해줘.
                                              
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


with main_col:
    # 🔹 메인 화면 제목
    st.title("⚙️ AI 기반 장애 대응 상황창 어시스턴트")

    # ==============================================
    # 💬 TAB 1 - 상황창 (Chat 기능)
    # ==============================================
    if st.session_state.active_tab == "tab1":
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False
        st.markdown("<div class='tab-div tab1-div'>", unsafe_allow_html=True)
        st.markdown("### 🟤 상황창 분석 (활성화됨)")

        chat_container = st.container()

        # ✅ 대화 출력
        if not st.session_state.is_generating:
            with chat_container:
                for msg in st.session_state.messages:
                    # spinner 실행 중에는 assistant 메시지 숨기기
                    if st.session_state.get("is_generating") and msg["role"] == "assistant":
                        continue  # 🔹 이전 어시스턴트 메시지 출력 안 함

                    role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
                    st.markdown(
                        "<div class='chat-bubble {}'>{}</div>".format(role_class, msg["content"].replace("\n", "<br>")),
                        unsafe_allow_html=True
                    )

        st.markdown("""
            <style>
            /* 입력창 전체 영역 하단 고정 */
            [data-testid="stChatInput"] {
                position: fixed;
                bottom: 0;
                left: 24%;            /* ✅ 좌측 여백 — 조정 가능 */
                width: 55%;           /* ✅ 입력창 전체 폭 — 중앙 main_col 크기에 맞게 */
                z-index: 999;
                background-color: white;
                border-top: 1px solid #ddd;
                padding-top: 8px;
            }
            </style>
            """, unsafe_allow_html=True)
        # 입력창 하단 고정
        user_input = st.chat_input("🗨 메시지를 입력하세요...")

        if user_input:
            # 사용자 메시지
            st.markdown(
                "<div class='chat-bubble user-msg'>{}</div>".format(user_input.replace("\n", "<br>")),
                unsafe_allow_html=True
            )
                        
            #st.chat_message("user").markdown(user_input.replace("\n", "<br>"), unsafe_allow_html=True)
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # 보고서 유형 판단
            report_type = "최종보고" if ("복구 완료" in user_input or "전체 완료" in user_input or "이상 없음 보고" in user_input) else "중간보고"

            # 대화 이력 포함
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])
            query_with_context = f"{chat_history}\nuser: {user_input}"

            st.session_state.is_generating = True
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

                    # \n → 줄바꿈 복원
                    content_text = content_text.replace("\\n", "\n")

                    #st.chat_message("assistant").markdown(content_text)
                    # st.session_state.messages.append({"role": "assistant", "content": content_text})
                    # role_class = "assistant-msg"
                    # st.markdown(f"<div class='chat-bubble {role_class}'>{content_text}</div>", unsafe_allow_html=True)
                    # ✅ AI 응답 즉시 표시 (rerun 중복 방지)
                    st.markdown(
                        "<div class='chat-bubble assistant-msg'>{}</div>".format(content_text.replace("\n", "<br>")),
                        unsafe_allow_html=True
                    )

                    # ✅ rerun 이후에도 유지되도록 세션에 추가
                    st.session_state.messages.append({"role": "assistant", "content": content_text})


                    st.session_state.reports.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": report_type,
                        "content": ai_output
                    })
                    
                    # 로그 저장
                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "assistant",
                        "report_type": report_type,
                        "input": user_input,
                        "output": ai_output
                    }

                    with open(log_path, "a", encoding="utf-8") as f:
                        json.dump(log_data, f, ensure_ascii=False)
                        f.write("\n")

                except Exception as e:
                    st.error(f"RAG 처리 중 오류 발생: {e}")

            st.session_state.is_generating = False

            # 자동 스크롤
            st.markdown("""
                <script>
                var chatDiv = window.parent.document.querySelector('.stMarkdown');
                if (chatDiv) { chatDiv.scrollTop = chatDiv.scrollHeight; }
                </script>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================
    # 📝 TAB 2 - 보고서
    # ==============================================
    elif st.session_state.active_tab == "tab2":
        st.markdown("<div class='tab-div tab2-div'>", unsafe_allow_html=True)
        st.markdown("### ⚪ 보고서 내역")

        if st.session_state.reports:
            latest_report = st.session_state.reports[-1]
            st.markdown(f"#### 📄 최신 보고서 ({latest_report['timestamp']})")
            st.markdown(f"**유형:** {latest_report['type']}")
            st.markdown("---")
            raw_text = latest_report["content"]
            match = re.search(r"content='(.*?)' additional_kwargs=", raw_text, re.DOTALL)
            cleaned_text = match.group(1).strip() if match else raw_text.strip()
            cleaned_text = cleaned_text.replace("\\n", "\n")
            st.markdown(cleaned_text)

            # 전체 보고서 히스토리
            with st.expander("📚 전체 보고서 기록 보기"):
                for r in reversed(st.session_state.reports):
                    st.markdown(f"- 🕓 `{r['timestamp']}` | **{r['type']}**")
                    raw_text = r["content"]
                    match = re.search(r"content='(.*?)' additional_kwargs=", raw_text, re.DOTALL)
                    cleaned_text = match.group(1).strip() if match else raw_text.strip()
                    cleaned_text = cleaned_text.replace("\\n", "\n")
                    st.markdown(f"> {cleaned_text}")

        else:
            st.info("아직 생성된 보고서가 없습니다.")

        st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown("### 💡 장애 발생 상황창")

    with open("data/sample_incident_chat.txt", "r", encoding="utf-8") as f:
        content = f.read()
        quotes = [q.strip() for q in content.split("|||") if q.strip()]

    if "random_quote" not in st.session_state:
        st.session_state["random_quote"] = random.choice(quotes)

    quote_html = st.session_state["random_quote"].replace("\\n", "<br>").replace("\n", "<br>")

    st.markdown(f"""
    <div style="
        background-color:#f8f9fa;
        border-radius:8px;
        padding:12px;
        border:1px solid #ddd;
        line-height:1.5;
    ">
    {quote_html}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 상황창 새로고침", use_container_width=True):
        st.session_state["random_quote"] = random.choice(quotes)
        st.rerun()