import os
import textwrap
import streamlit as st
from embeddings import Embeddings
from vector_db import VectorDatabase
from reranker import Reranker
import time

# ---------------- Page setup ----------------
st.set_page_config(page_title="NeoRAG Chat", page_icon="🧠", layout="wide")
st.title("🧠 NeoRAG Chat — ProPTIT")
st.caption("RAG được lập trình tại cuộc thi NeoRAG-Cup-2025 của TeamAIProPTIT")

# Subtle CSS to make it look neat
# st.markdown(
#     """
#     <style>
#       .stChatFloatingInputContainer { bottom: 1.2rem; }
#       .block-container { padding-top: 1.5rem; }
#       .small-note { color: #6b7280; font-size: 0.88rem; }
#       .ctx { background: #0b13241a; padding: .6rem .8rem; border-radius: .5rem; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# ---- Chat CSS ----
st.markdown("""
<style>
.chat-row {
    display: flex;
    align-items: flex-start;
    margin: 0.4rem 0;
}
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
}
.chat-bubble {
    padding: 0.6rem 1rem;
    margin: 0 0.5rem;
    border-radius: 1rem;
    max-width: 75%;
    font-size: 0.95rem;
    word-wrap: break-word;
    line-height: 1.4;
}
.assistant-row {
    flex-direction: row;
}
.user-row {
    flex-direction: row-reverse;   /* đảo ngược avatar + bubble */
}
.user-bubble {
    background-color: #DCF8C6; /* Xanh nhạt WhatsApp */
    margin-left: auto;
    margin-right: 0;
    border-bottom-right-radius: 0.3rem;
}
.assistant-bubble {
    background-color: #F1F0F0;
    margin-right: auto;
    margin-left: 0;
    border-bottom-left-radius: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Cache heavy resources ----------------
@st.cache_resource(show_spinner=False)
def get_vector_db(db_type: str):
    return VectorDatabase(db_type=db_type)

@st.cache_resource(show_spinner=False)
def get_embedder(provider: str, model_name: str):
    return Embeddings(model_name=model_name, type=provider)

@st.cache_resource(show_spinner=False)
def get_reranker(model_name: str, use_fp16: bool = True, normalize: bool = True):
    return Reranker(model_name=model_name, use_fp16=use_fp16, normalize=normalize)

def streaming_answer(text):
    for chunk in text.split():
        yield chunk + " "
        time.sleep(0.05)
# ---------------- Core helpers ----------------
def retrieve_and_rerank(query: str, collection: str, top_k: int, top_n: int,
                        vdb: VectorDatabase, embedder: Embeddings, reranker: Reranker):
    """Retrieve top_k docs then rerank and return top_n.
    Returns (pre_results, reranked_results).
    """
    try:
        q_emb = embedder.encode(query)
    except Exception:
        st.error("Không tạo được embedding cho truy vấn. Kiểm tra API key/model.")
        return [], []

    init_limit = max(top_k, top_n) * 2
    pre_results = vdb.query(collection, q_emb, init_limit)
    if not pre_results:
        return [], []

    passages = [d.get("information", "") for d in pre_results]
    try:
        scores, ranked_passages = reranker(query, passages)
        score_map = {p: s for p, s in zip(ranked_passages, scores)}
        reranked = []
        for p in ranked_passages:
            m = next((r for r in pre_results if r.get("information", "") == p), None)
            if m:
                item = dict(m)
                item["rerank_score"] = float(score_map.get(p, 0.0))
                reranked.append(item)
        return pre_results[:top_k], reranked[:top_n]
    except Exception:
        return pre_results[:top_k], pre_results[:top_n]


def generate_response(query: str, contexts: list, llm_choice: str, model_name: str) -> str:
    context_text = "\n\n".join([f"- {d.get('information','')}" for d in contexts])
    system = (
        """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
    )
    prompt = f"Context:\n{context_text}\n\nCâu hỏi: {query}"
    try:
        if llm_choice == "gemini":
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            content = client.models.generate_content(
                model=model_name,
                contents=[
                    {"role": "user", "parts": [system]},
                    {"role": "user", "parts": [prompt]},
                ],
            )
            return getattr(content, "text", "").strip()
        elif llm_choice == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=0.2,
            )
            full_answer = ""
        message_holder = st.empty()  # placeholder cho UI

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                full_answer += delta.content
                message_holder.markdown(full_answer + "▌")

        message_holder.markdown(full_answer)  # in kết quả cuối
        return full_answer  # trả về string sạch để lưu vào session
    except Exception:
        return "Không thể gọi LLM. Vui lòng kiểm tra cấu hình và API key."

# ---------------- Sidebar config ----------------
with st.sidebar:
    st.header("Cấu hình")
    db_type = st.selectbox("Vector DB", options=["mongodb"], index=0)
    collection = st.text_input("Collection", value="information")

    st.divider()
    st.subheader("Embedding")
    embed_provider = st.selectbox("Provider", options=["gemini"], index=0)
    default_embed = {
        "gemini": "gemini-embedding-001"
    }[embed_provider]
    emb_model = st.text_input("Model", value=default_embed)

    st.divider()
    st.subheader("Reranker")
    rerank_model_name = st.text_input("Model", value="namdp-ptit/ViRanker")
    top_k = st.slider("Top-K retrieval", 1, 20, 5)
    top_n = st.slider("Top-N sau rerank", 1, 20, 5)

    st.divider()
    st.subheader("LLM")
    llm_choice = st.selectbox("Provider", options=["gemini", "openai"], index=0)
    default_llm = {"gemini": "gemini-1.5-flash", "openai": "gpt-4o-mini"}[llm_choice]
    llm_model = st.text_input("Model", value=default_llm)

    st.divider()
    show_sources = st.toggle("Hiển thị nguồn trong câu trả lời", value=False)

    if st.button("Xóa lịch sử chat"):
        st.session_state.clear()
        st.success("Đã xóa lịch sử chat.")

# ---------------- Build resources ----------------
vdb = get_vector_db(db_type)
embedder = get_embedder(embed_provider, emb_model)
reranker = get_reranker(rerank_model_name)

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []

# ---------------- Render chat history (top -> bottom) ----------------
# We render full history first; new messages will be added then rerun will show them at the end
for m in st.session_state.messages:
    if m["role"] == "user":
        avatar = "https://cdn-icons-png.flaticon.com/512/1946/1946429.png"
        row_class = "user-row"
        role_class = "user-bubble"
    else:
        avatar = "https://tse1.mm.bing.net/th/id/OIP.RS529nxwk4usbXFkQvrwKQHaHa?rs=1&pid=ImgDetMain&o=7&rm=3"
        row_class = "assistant-row"
        role_class = "assistant-bubble"

    st.markdown(f"""
    <div class="chat-row {row_class}">
        <img class="chat-avatar" src="{avatar}">
        <div class="chat-bubble {role_class}">{m['content']}</div>
    </div>
    """, unsafe_allow_html=True)



# ---------------- Chat input ----------------
prompt = st.chat_input("Nhập câu hỏi về CLB Lập Trình PTIT...")
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
<div class="chat-row user-row">
    <img class="chat-avatar" src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png">
    <div class="chat-bubble user-bubble">{prompt}</div>
</div>
""", unsafe_allow_html=True)
    # Show user's message immediately (like ChatGPT)
    # with st.chat_message("user"):
    #     st.write_stream(streaming_answer(prompt))

    # Retrieve + rerank
    with st.spinner("🔎 Đang truy xuất và rerank..."):
        pre, post = retrieve_and_rerank(
            query=prompt,
            collection=collection,
            top_k=top_k,
            top_n=top_n,
            vdb=vdb,
            embedder=embedder,
            reranker=reranker,
        )
        st.session_state.pre_results = pre
        st.session_state.reranked_results = post
        st.session_state.last_contexts = post or pre

    # Generate answer
    with st.spinner("✍️ Đang tạo câu trả lời…"):
        ans = generate_response(prompt, st.session_state.last_contexts, llm_choice, llm_model)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.markdown(f"""
<div class="chat-row">
    <img class="chat-avatar" src="https://tse1.mm.bing.net/th/id/OIP.RS529nxwk4usbXFkQvrwKQHaHa?rs=1&pid=ImgDetMain&o=7&rm=3">
    <div class="chat-bubble assistant-bubble">{ans}</div>
</div>
""", unsafe_allow_html=True)

    # Force rerun so that full history renders top->bottom cleanly once
    st.rerun()

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0b1324;
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        opacity: 0.85;
    }
    </style>
    <div class="footer">
        🚀 Powered by <b>HieuGM</b>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Optional sources note ----------------
if show_sources and st.session_state.get("last_contexts"):
    st.markdown("---")
    st.subheader("Nguồn tham khảo (tóm tắt)")
    for i, d in enumerate(st.session_state.last_contexts, 1):
        title = d.get("title", "(no title)")
        info = d.get("information", "")
        snippet = textwrap.shorten(info, width=280, placeholder=" …")
        st.markdown(f"**{i}. {title}**\n\n<div class='ctx'>{snippet}</div>", unsafe_allow_html=True)
