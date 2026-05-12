import streamlit as st
import json
import os
import re
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# 知识库相关
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
import hashlib

# ==================== 配置 ====================
load_dotenv(Path(__file__).parent / ".env")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4-flash")
EMBED_MODEL = "embedding-2"  # 智谱 embedding 模型

# ==================== 员工花名册 ====================
MY_TEAM = {
    "media": {
        "name": "📢 自媒体运营Agent",
        "desc": "文案撰写、爆款选题、评论区初稿",
    },
    "dianshang": {
        "name": "🛒 电商Agent",
        "desc": "订单整理、库存预警、详情页草稿",
    },
    "xiaofang": {
        "name": "📁 资料处理Agent",
        "desc": "消防文档关键信息提取、模板化报告",
    },
    "guanli": {
        "name": "📅 个人管理Agent",
        "desc": "项目进度、待办清单、周报复盘",
    },
    "duanju": {
        "name": "🎬 AI视频生成剪辑自动化流Agent",
        "desc": "写作短剧小说、生成角色、推理短剧分片、剪辑成片",
    },
}

# ==================== 加载已部署的Agent ====================
def load_deployed_agents():
    deployed = {}
    base = Path(__file__).parent
    for d in base.glob("agent_*_project"):
        agent_id = d.name.replace("agent_", "").replace("_project", "")
        prompt_file = d / "prompts" / "media_expert.txt"
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            deployed[agent_id] = {
                "system_prompt": system_prompt,
                "dir": d,
            }
    return deployed

deployed_agents = load_deployed_agents()
deployed_ids = [aid for aid in MY_TEAM if aid in deployed_agents]

# ==================== 初始化 session ====================
if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = []
if "collab_mode" not in st.session_state:
    st.session_state.collab_mode = False
if "collab_agents" not in st.session_state:
    st.session_state.collab_agents = []
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {aid: "空闲" for aid in MY_TEAM}
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
for aid in MY_TEAM:
    if f"messages_{aid}" not in st.session_state:
        st.session_state[f"messages_{aid}"] = []

# ==================== 知识库 ChromaDB 初始化 ====================
PERSIST_DIR = str(Path(__file__).parent / ".chroma_db")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name="my_library")

# ==================== 界面 ====================
st.set_page_config(page_title="一人公司 · 我的AI团队", page_icon="🏢", layout="wide")

# 模式切换
with st.sidebar:
    st.header("🧭 功能导航")
    mode = st.radio(
        "选择模式",
        ["💬 团队对话", "📚 知识库"],
        index=0
    )

if mode == "💬 团队对话":
    # ========== 原有对话界面 ==========
    st.markdown("# 🏢 我的AI虚拟团队")
    st.caption("一人公司 · 5位核心AI员工 · 7×24全天候待命 · 群聊协作已就绪")

    # 任务状态看板
    status_cols = st.columns(len(deployed_ids) if deployed_ids else 1)
    for i, aid in enumerate(deployed_ids):
        if aid in MY_TEAM:
            info = MY_TEAM[aid]
            with status_cols[i]:
                status = st.session_state.agent_status.get(aid, "空闲")
                color = "🟢" if status == "空闲" else "🟡" if status == "工作中" else "🔵"
                st.metric(label=f"{info['name']}", value=f"{color} {status}")
    st.divider()

    with st.sidebar:
        st.header("👥 我的AI员工")
        st.caption("勾选后点击“组队协作”即可多人讨论")
        selected = []
        for agent_id, agent_info in MY_TEAM.items():
            is_active = agent_id in deployed_agents
            if is_active:
                if st.checkbox(f"🟢 {agent_info['name']}", value=(agent_id in st.session_state.selected_agents), key=f"chk_{agent_id}"):
                    selected.append(agent_id)
            else:
                st.markdown(f"⚪ {agent_info['name']}")
            st.caption(f"   {agent_info['desc']}")
        st.session_state.selected_agents = selected

        st.divider()
        if st.button("🤝 组队协作 (先勾选多人，再点这里)"):
            if len(selected) >= 2:
                st.session_state.collab_mode = True
                st.session_state.collab_agents = selected.copy()
                st.success(f"已组建协作小组：{', '.join([MY_TEAM[a]['name'] for a in selected])}")
            else:
                st.warning("请至少勾选两个员工再组队")

        if st.button("🔙 退出协作 (单人模式)"):
            st.session_state.collab_mode = False
            st.session_state.collab_agents = []
            if selected:
                st.session_state.selected_agents = [selected[0]]
        st.divider()
        st.caption("🟢 = 已激活 | ⚪ = 未部署")
        st.caption("协作时，消息会自动按顺序传递")

    # 主对话区
    collab_mode = st.session_state.collab_mode
    if collab_mode:
        collab_agents = st.session_state.collab_agents
        st.markdown("## 🤝 协作小组模式")
        st.caption(f"当前小组成员：{' ➔ '.join([MY_TEAM[a]['name'] for a in collab_agents])} （任务将按此顺序传递）")
        for agent_id in collab_agents:
            msgs = st.session_state[f"messages_{agent_id}"]
            if msgs:
                st.markdown(f"### {MY_TEAM[agent_id]['name']}")
                for msg in msgs:
                    with st.chat_message(msg["role"]):
                        st.markdown(f"**{msg.get('sender_name', '')}**: {msg['content']}")
        prompt = st.chat_input("对协作小组下达总任务...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(f"📢 总任务：{prompt}")
            for agent_id in collab_agents:
                st.session_state[f"messages_{agent_id}"].append({"role": "user", "content": f"📢 协作总任务：{prompt}", "sender_name": "指挥官"})
            previous_output = prompt
            for idx, agent_id in enumerate(collab_agents):
                st.session_state.agent_status[agent_id] = "工作中"
                system_prompt = deployed_agents[agent_id]["system_prompt"]
                task_input = previous_output if idx == 0 else f"前一个部门（{MY_TEAM[collab_agents[idx-1]]['name']}）的输出成果如下：\n{previous_output}\n\n请基于此，完成你的专业部分，并给出完整结果。"
                with st.chat_message("assistant"):
                    with st.spinner(f"{MY_TEAM[agent_id]['name']} 正在思考..."):
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": task_input}
                                ],
                                temperature=0.8,
                            )
                            ai_text = response.choices[0].message.content
                            st.markdown(f"**{MY_TEAM[agent_id]['name']}**：{ai_text}")
                            st.session_state[f"messages_{agent_id}"].append({"role": "assistant", "content": ai_text, "sender_name": MY_TEAM[agent_id]['name']})
                            previous_output = ai_text
                            st.session_state.agent_status[agent_id] = "完成"
                            st.session_state.total_messages += 1
                        except Exception as e:
                            st.error(f"{MY_TEAM[agent_id]['name']} 出错：{str(e)}")
                            st.session_state.agent_status[agent_id] = "空闲"
                            break
            else:
                st.success("✅ 协作任务全部完成！")
    else:
        if not st.session_state.selected_agents and deployed_ids:
            st.session_state.selected_agents = [deployed_ids[0]]
        current_agent = st.session_state.selected_agents[0] if st.session_state.selected_agents else None
        if current_agent and current_agent in deployed_agents:
            agent_info = MY_TEAM[current_agent]
            st.markdown(f"## {agent_info['name']}")
            st.caption(agent_info['desc'])
            msgs = st.session_state[f"messages_{current_agent}"]
            for msg in msgs:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            prompt = st.chat_input(f"对 {agent_info['name']} 说点什么...")
            if prompt:
                msgs.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                system_prompt = deployed_agents[current_agent]["system_prompt"]
                with st.chat_message("assistant"):
                    with st.spinner(f"{agent_info['name']} 正在思考..."):
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.8,
                            )
                            ai_text = response.choices[0].message.content
                            st.markdown(ai_text)
                            msgs.append({"role": "assistant", "content": ai_text})
                            st.session_state.agent_status[current_agent] = "空闲"
                            st.session_state.total_messages += 1
                        except Exception as e:
                            st.error(f"出错了：{str(e)}")
        else:
            st.info("👈 请在左侧选择一个激活的员工开始对话")

elif mode == "📚 知识库":
    # ========== 知识库界面 ==========
    st.markdown("# 📚 知识库 · 文档问答")
    st.caption("上传 PDF 或 TXT 文件，让 AI 基于你的资料回答问题。使用智谱 Embedding 进行语义检索。")

    # 辅助函数
    def split_text(text, chunk_size=1500, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def get_embedding(text):
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    # 侧边栏文件上传
    with st.sidebar:
        st.header("📤 上传文档")
        uploaded_file = st.file_uploader("选择 PDF 或 TXT 文件", type=["pdf", "txt"])
        if uploaded_file is not None:
            with st.spinner("正在解析文档..."):
                # 读取文件内容
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                else:
                    text = uploaded_file.getvalue().decode("utf-8")
                # 分块
                chunks = split_text(text)
                # 计算每个块的 embedding 并存入 ChromaDB
                file_id = hashlib.md5(uploaded_file.getvalue()).hexdigest()
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        try:
                            emb = get_embedding(chunk)
                            collection.add(
                                embeddings=[emb],
                                documents=[chunk],
                                ids=[f"{file_id}_{i}"],
                                metadatas=[{"source": uploaded_file.name, "file_id": file_id}]
                            )
                        except Exception as e:
                            st.error(f"嵌入失败：{e}")
                st.success(f"已添加《{uploaded_file.name}》到知识库（{len(chunks)} 个片段）")

        if st.button("🗑 清空知识库"):
            # 删除 collection 中所有数据
            collection.delete(where={})
            st.success("知识库已清空")

        # 显示已入库的文件
        st.divider()
        st.caption("📋 已入库文件")
        # 获取所有记录，提取唯一 source
        all_data = collection.get()
        sources = set()
        if all_data['metadatas']:
            for meta in all_data['metadatas']:
                sources.add(meta.get('source', ''))
            for src in sources:
                st.text(f"📄 {src}")
        else:
            st.text("暂无文件")

    # 主区域：问答
    st.subheader("🔍 向知识库提问")
    question = st.text_input("输入你的问题", placeholder="例如：这本书主要讲了什么？")
    if question:
        with st.spinner("正在搜索知识库..."):
            try:
                q_emb = get_embedding(question)
                results = collection.query(
                    query_embeddings=[q_emb],
                    n_results=3
                )
                docs = results['documents'][0] if results['documents'] else []
                if not docs:
                    st.warning("知识库中没有找到相关内容。")
                else:
                    context = "\n\n".join(docs)
                    prompt = f"""你是一个知识渊博的助手，根据以下资料回答问题。如果资料中没有相关信息，请如实回答“资料中未提及”。
资料：
{context}
问题：{question}
回答："""
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 📖 回答")
                    st.write(answer)
                    with st.expander("查看相关原文片段"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**片段 {i+1}**")
                            st.text(doc)
            except Exception as e:
                st.error(f"检索出错：{str(e)}")
