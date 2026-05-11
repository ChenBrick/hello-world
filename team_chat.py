import streamlit as st
import json
import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4-flash")

AGENTS = {}
AGENT_DIRS = {}
base_path = Path(r"D:\AI_Agents")
for d in base_path.glob("/home/ubuntu/AI_Agents"):
    prompt_file = d / "prompts" / "media_expert.txt"
    if prompt_file.exists():
        agent_name = d.name.replace("agent_", "").replace("_project", "")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
        AGENTS[agent_name] = prompt
        AGENT_DIRS[agent_name] = d

st.set_page_config(page_title="一人公司虚拟团队", layout="wide")
st.title("🤖 虚拟团队群聊室")

with st.sidebar:
    st.header("👥 选择部门")
    if AGENTS:
        selected_agent = st.radio("点选一个AI员工对话", list(AGENTS.keys()), index=0)
    else:
        st.warning("未找到任何Agent项目，请先创建。")
        st.stop()
    st.divider()
    st.caption("在下方输入框发送消息，当前选中的AI员工会回复")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f"对 {selected_agent} 说点什么..."):
    st.session_state.messages.append({"role": "user", "content": f"@{selected_agent}：{prompt}", "agent": selected_agent})
    with st.chat_message("user"):
        st.markdown(f"@{selected_agent}：{prompt}")

    system_prompt = AGENTS[selected_agent]
    with st.chat_message("assistant"):
        with st.spinner(f"{selected_agent} 正在思考..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    stream=False
                )
                ai_text = response.choices[0].message.content
                st.markdown(ai_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_text, "agent": selected_agent})
            except Exception as e:
                st.error(f"出错了：{str(e)}")
