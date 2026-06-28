import os
import time
import streamlit as st
from dotenv import load_dotenv

# Import LangChain modules
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Get absolute paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# Page Configuration
st.set_page_config(
    page_title="CCSP Exam Prep Chatbot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injected via CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main background & styling */
.reportview-container, .main {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0b0f19 !important;
}

/* Custom Header Card */
.header-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    gap: 20px;
}

.header-icon {
    font-size: 48px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-title {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
}

.header-subtitle {
    margin: 4px 0 0 0;
    font-size: 14px;
    color: #94a3b8;
}

/* Glassmorphism sidebar & widget styling */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.sidebar-status {
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 20px;
}

.status-ready {
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}

.status-warning {
    background-color: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

/* Chat bubble styling overrides */
.stChatMessage {
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

.stChatMessage[data-testid="stChatMessageUser"] {
    background-color: rgba(59, 130, 246, 0.1) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] {
    background-color: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Scrollbar customization */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Helper function to check if vector DB exists on disk
def is_vector_db_indexed():
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))

# Helper function to get embedding model with automatic fallback
def get_embeddings(api_key):
    # 1. 嘗試最新的 models/text-embedding-004
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        embeddings.embed_query("test")
        return embeddings
    except Exception:
        pass

    # 2. 嘗試 Gemini 專用的 models/gemini-embedding-001 (目前主要推薦的模型)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        embeddings.embed_query("test")
        st.sidebar.info("💡 已自動載入 Gemini 推薦的 models/gemini-embedding-001 向量模型。")
        return embeddings
    except Exception:
        pass

    # 3. 嘗試經典的 models/embedding-001
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        embeddings.embed_query("test")
        st.sidebar.info("💡 已自動載入經典款 models/embedding-001 向量模型。")
        return embeddings
    except Exception as e:
        st.sidebar.error(f"❌ 無法載入任何 Embedding 模型。請點擊側邊欄「🔍 檢測 API Key 與可用模型」按鈕進行連線測試。最後錯誤: {e}")
        raise e

# Sidebar Configuration
st.sidebar.title("⚙️ CCSP Chatbot 設定")

# 1. API Key Setup (Check environment first, else request input)
env_api_key = os.getenv("GOOGLE_API_KEY", "")
api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=env_api_key,
    type="password",
    help="請輸入您的 Google Gemini API Key。若已在 .env 設定則會自動帶入。"
)

# API Key Diagnostics Button
if st.sidebar.button("🔍 檢測 API Key 與可用模型"):
    if not api_key:
        st.sidebar.error("❌ 請先輸入 API Key")
    else:
        with st.sidebar.status("正在與 Google 伺服器連線...", expanded=True) as status:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                models = list(client.models.list())
                embed_models = [m.name for m in models if "embed" in m.name.lower()]
                all_models = [m.name for m in models]
                status.update(label="✅ 連線成功！", state="complete")
                st.sidebar.success("API Key 驗證成功！")
                st.sidebar.write("**可用的 Embedding 模型：**")
                if embed_models:
                    for m in embed_models:
                        st.sidebar.code(m)
                else:
                    st.sidebar.warning("⚠️ 找不到任何支援 Embedding 的模型。請檢查此 API Key 是否有 Generative Language API 權限。")
                st.sidebar.write("**前 10 個可用模型：**")
                st.sidebar.json(all_models[:10])
            except Exception as e:
                status.update(label="❌ 連線失敗", state="error")
                st.sidebar.error(f"檢測時發生錯誤: {e}")
                st.sidebar.info("💡 提示：請確認您的 API Key 是否正確。如果是從 Google Cloud Console 申請，請確認是否已啟用 'Generative Language API' 服務。")

# 2. LLM Model Selection
model_option = st.sidebar.selectbox(
    "Reasoning Model",
    options=["gemini-3.5-flash", "gemini-3.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
    help="Gemini 3.5 Flash 是目前推薦的標準模型，速度快且效能極佳；Gemini 3.5 Pro 具備頂級推理能力，適合複雜情境題分析。"
)

# 3. Model Parameters
temperature = st.sidebar.slider(
    "Temperature (隨機度)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
    help="較低的值可獲得更精準與事實一致的回答，適合資安考試準備。"
)

# 4. RAG Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("📄 文件向量化參數")
chunk_size = st.sidebar.slider("Chunk Size (字元數)", 500, 3000, 2000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap (重疊字元)", 50, 600, 400, 50)

# Load Vector DB
@st.cache_resource(show_spinner=False)
def get_vector_db(api_key):
    if not is_vector_db_indexed():
        return None
    try:
        embeddings = get_embeddings(api_key)
        # Load local FAISS database (safe to deserialize local files)
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.sidebar.error(f"載入向量庫時出錯: {e}")
        return None

# Build Vector DB
def trigger_indexing(api_key, chunk_size, chunk_overlap):
    if not api_key:
        st.error("❌ 請先在側邊欄輸入有效的 Google Gemini API Key 才能進行向量化。")
        return
        
    status_card = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Step 1: Scan
        status_card.info(f"🔍 Step 1/4: 正在掃描 `{DATA_DIR}` 資料夾中的 PDF 文件...")
        progress_bar.progress(10)
        
        if not os.path.exists(DATA_DIR):
            st.error(f"❌ 找不到 `{DATA_DIR}` 資料夾，請先建立此資料夾並上傳 CCSP 相關教材。")
            progress_bar.empty()
            status_card.empty()
            return
            
        loader = PyPDFDirectoryLoader(DATA_DIR)
        documents = loader.load()
        
        if not documents:
            st.error("❌ `./data` 資料夾中沒有找到任何 PDF 檔案。")
            progress_bar.empty()
            status_card.empty()
            return
            
        # Step 2: Split
        status_card.info(f"✂️ Step 2/4: 成功載入 {len(documents)} 頁 PDF。正在將內容切割成 text chunks (Size: {chunk_size}, Overlap: {chunk_overlap})...")
        progress_bar.progress(40)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # ⚠️ 針對免費版額度與速率限制進行警示
        if len(chunks) > 500:
            st.warning(f"⚠️ 警告：目前切割出 {len(chunks)} 個區塊。由於 Gemini API 免費版限制每日 1,500 次呼叫，且每分鐘限額 100 個區塊，您極有可能在處理過程中耗盡額度。建議在 `./data` 中只放置 1~2 本核心教材檔案（例如 outline 或精簡版講義）以利成功向量化。")
        
        # Step 3: Embed & Index (Batching with Rate Limit sleep)
        embeddings = get_embeddings(api_key)
        
        status_card.info(f"🧠 Step 3/4: 已切分為 {len(chunks)} 個區塊。正在分批產生向量並建立資料庫...")
        progress_bar.progress(50)
        
        # Batch size settings (Gemini Free Tier has limits on requests per minute: 100 items/min)
        # 設為 15 個區塊，配合 10 秒延遲，每分鐘處理 90 個區塊，能完美控制在 100 RPM 限制內
        batch_size = 15
        total_chunks = len(chunks)
        
        # Initialize the FAISS database with the first batch
        first_batch = chunks[:batch_size]
        db = FAISS.from_documents(first_batch, embeddings)
        
        # Process the remaining batches
        for i in range(batch_size, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            db.add_documents(batch)
            
            # 💾 每一批次處理完立即存檔，確保即使因為每日額度限制中斷，已完成的進度仍可使用！
            db.save_local(INDEX_DIR)
            
            # Update progress (scale from 50% to 90% progress bar)
            progress_percent = 50 + int((i / total_chunks) * 40)
            progress_bar.progress(progress_percent)
            status_card.info(f"🧠 Step 3/4: 正在分批產生向量 ({min(i + batch_size, total_chunks)}/{total_chunks} 區塊)...")
            
            # Delay to prevent API Rate Limits (429 Resource Exhausted) on Gemini Free Tier (Sleep 10s)
            time.sleep(10.0)
        
        # Step 4: Save
        status_card.info(f"💾 Step 4/4: 正在將向量資料庫存檔至本地目錄 `{INDEX_DIR}`...")
        progress_bar.progress(90)
        db.save_local(INDEX_DIR)
        
        progress_bar.progress(100)
        status_card.success(f"🎉 向量化完成！共處理 {len(chunks)} 個文字區塊。")
        
        # Clear st.cache_resource to force reloading of newly indexed database
        st.cache_resource.clear()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ 向量化過程中發生錯誤: {e}")
        progress_bar.empty()
        status_card.empty()

# Show Status in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📡 向量資料庫狀態")
if is_vector_db_indexed():
    st.sidebar.markdown('<div class="sidebar-status status-ready">🟢 向量資料庫已就緒 (本地有存檔)</div>', unsafe_allow_html=True)
    if st.sidebar.button("🔄 重新掃描並向量化文件"):
        trigger_indexing(api_key, chunk_size, chunk_overlap)
else:
    st.sidebar.markdown('<div class="sidebar-status status-warning">🟡 向量資料庫未建立 (請先向量化)</div>', unsafe_allow_html=True)
    if st.sidebar.button("⚡ 開始掃描並向量化文件"):
        trigger_indexing(api_key, chunk_size, chunk_overlap)

# Main Application Layout
st.markdown("""
<div class="header-card">
    <div class="header-icon">🛡️</div>
    <div>
        <h1 class="header-title">CCSP 雲端資安專家 — 考試準備 Chatbot</h1>
        <p class="header-subtitle">搭配 RAG (檢索增強生成) 系統，為您即時查詢本地 CCSP 教材與模擬試題庫</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Try loading DB if API Key is present
db = None
if api_key:
    db = get_vector_db(api_key)

# App logical flow
if not api_key:
    st.info("💡 **歡迎使用 CCSP 考試準備 Chatbot**\n\n請在左側邊欄輸入您的 **Google Gemini API Key** 以啟動服務。")
elif not is_vector_db_indexed():
    st.warning("⚠️ **向量資料庫尚未建立**\n\n偵測到本地還沒有建立向量資料庫。請在左側側邊欄點選 **「⚡ 開始掃描並向量化文件」** 按鈕，系統會自動處理 `./data` 資料夾中的所有 PDF 檔案。")
else:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "您好！我是您的 CCSP 雲端資安專家導師。我已經研讀了本地資料夾中的教材，隨時可以為您解答關於 CCSP 考試、雲端資安觀念（如 CSA CCM, OWASP Top 10, SDLC, GDPR 等）或解析模擬試題。您想先詢問哪一部分呢？", "sources": []}
        ]
        
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render sources if available
            if msg.get("sources"):
                with st.expander("📖 查看本題參考來源資料"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**來源 {idx+1}:** `{src['source']}` (Page {src['page']})")
                        st.caption(f"*\"... {src['snippet']} ...\"*")
                        st.markdown("---")

    # Chat Input
    if prompt := st.chat_input("請輸入您想發問的 CCSP 觀念或題目..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Assistant generation
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            spinner_placeholder = st.empty()
            
            with spinner_placeholder:
                st.spinner("正在尋找相關資料並思考回答...")
                
            try:
                # 1. Set up Retriever
                # Retrieve top 4 most relevant chunks
                retriever = db.as_retriever(search_kwargs={"k": 4})
                
                # 2. Build QA Chain
                llm = ChatGoogleGenerativeAI(
                    model=model_option, 
                    google_api_key=api_key, 
                    temperature=temperature
                )
                
                # System prompt guiding LLM behavior as an exam coach
                system_prompt = (
                    "你是一位專業的 CCSP (Certified Cloud Security Professional) 雲端資安專家導師。\n"
                    "請根據下方提供的 CCSP 參考教材內容，為使用者提供專業、精準、結構化的解答。\n"
                    "若教材內容中無法找到答案，請基於一般的雲端資安知識進行推導與解答，並明確告知哪些是基於一般觀念補充而非教材原文。\n"
                    "在解答時請盡量條列式說明，若使用者問及考題，請協助分析各選項對錯的原因。\n\n"
                    "參考教材內容：\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # 3. Invoke RAG Chain
                response = rag_chain.invoke({"input": prompt})
                
                # Extract response text and context source documents
                answer = response["answer"]
                source_docs = response.get("context", [])
                
                # Process source documents
                sources = []
                for doc in source_docs:
                    meta = doc.metadata
                    # Extract file name and page number if available
                    source_path = meta.get("source", "Unknown Document")
                    source_file = os.path.basename(source_path)
                    page_num = meta.get("page", 0) + 1  # PyPDF is 0-indexed, human pages are usually 1-indexed
                    
                    snippet = doc.page_content[:200].replace('\n', ' ')
                    sources.append({
                        "source": source_file,
                        "page": page_num,
                        "snippet": snippet
                    })
                
                # Remove duplicate source page references to keep UI tidy
                unique_sources = []
                seen_refs = set()
                for src in sources:
                    ref_key = f"{src['source']}_page_{src['page']}"
                    if ref_key not in seen_refs:
                        seen_refs.add(ref_key)
                        unique_sources.append(src)
                
                # Clean spinner
                spinner_placeholder.empty()
                
                # Render answer in Streamlit chat UI
                response_placeholder.markdown(answer)
                
                # Render sources
                if unique_sources:
                    with st.expander("📖 查看本題參考來源資料"):
                        for idx, src in enumerate(unique_sources):
                            st.markdown(f"**來源 {idx+1}:** `{src['source']}` (Page {src['page']})")
                            st.caption(f"*\"... {src['snippet']} ...\"*")
                            st.markdown("---")
                            
                # Save message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": unique_sources
                })
                
            except Exception as e:
                spinner_placeholder.empty()
                st.error(f"❌ 產生回應時發生錯誤: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"非常抱歉，在處理您的請求時發生錯誤。請確認您的 API Key 是否正確且具備額度。\n\n詳細錯誤訊息：`{e}`",
                    "sources": []
                })
