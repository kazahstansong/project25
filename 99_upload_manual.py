## add_index_content.json 로 인덱스 생성 후 실행.

import json
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from openai import AzureOpenAI

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

# ✅ 1. JSON 파일 로드 (스크립트 위치 기준 상대 경로 사용)
BASE_DIR = Path(__file__).resolve().parent
# 후보 경로들: (1) 현재 스크립트 폴더의 data, (2) 상위 폴더의 data
candidate_paths = [
    BASE_DIR / "data" / "incident_manual.json",
    BASE_DIR.parent / "data" / "incident_manual.json",
]

DATA_PATH = None
for p in candidate_paths:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    checked = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "데이터 파일을 찾을 수 없습니다. 다음 경로들을 확인했습니다:\n" + checked
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

raw_stripped = raw.strip()
# remove surrounding triple-backticks if present (e.g., markdown paste)
if raw_stripped.startswith("```"):
    lines = raw_stripped.splitlines()
    # drop first and last lines
    if len(lines) >= 3:
        raw = "\n".join(lines[1:-1])

docs = []
try:
    # 첫 시도: 전체가 JSON 배열인 경우
    docs = json.loads(raw)
except json.JSONDecodeError:
    # NDJSON (줄 단위 JSON 객체) 처리
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            docs.append(json.loads(s))
        except json.JSONDecodeError:
            # 무시하거나 로그로 남길 수 있음
            raise

print(f"Loaded {len(docs)} docs from {DATA_PATH}")
    

# ✅ 2. Azure OpenAI 클라이언트 초기화
client = AzureOpenAI(
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-06-01"
)

def embed_text(text):
    """Azure OpenAI 임베딩 생성 (환경변수 버전)"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip() or " "  # 빈문자 방지
    
    response = client.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT,  # ✅ 반드시 배포 이름 사용
        input=[text]
    )
    return response.data[0].embedding

# ✅ 3. Azure Search 업로드 데이터 생성
upload_data = []
for doc in docs:
    try:
        vector = embed_text(doc["content"])
        upload_data.append({
            "@search.action": "upload",
            "id": doc["id"],
            "content": f"[{doc['category']}] {doc['content']}",
            "content_vector": vector
        })
        print(f"✅ {doc['id']} 업로드 준비 완료")
    except Exception as e:
        print(f"❌ {doc['id']} 임베딩 생성 실패:", e)

# ✅ 4. Azure Search 업로드 요청
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_KEY
}
payload = {"value": upload_data}

response = requests.post(
    f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/index?api-version=2024-07-01",
    headers=headers,
    json=payload
)

print("\n🔍 상태 코드:", response.status_code)
print("📦 응답 내용:", response.text)