import json
import os
from datetime import datetime
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

# 初始化客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 加载 System Prompt
with open("prompts/media_expert.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

def generate_media_content(user_request: str) -> dict:
    """将用户需求发给大模型，返回解析后的JSON"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request}
        ],
        temperature=0.8
    )
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"raw_output": content}
    return result

def save_to_markdown(data: dict, topic: str):
    """保存为时间戳命名的 markdown 文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/{timestamp}_{topic[:20]}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 选题: {topic}\n\n")
        f.write(f"**思路分析:** {data.get('thinking', '')}\n\n")
        f.write(f"**标题:** {data.get('title', '')}\n\n")
        f.write(f"**正文/脚本:**\n{data.get('content', '')}\n\n")
        f.write(f"**标签:** {' '.join(data.get('hashtags', []))}\n\n")
        f.write(f"**回复模板:**\n")
        for t in data.get('reply_templates', []):
            f.write(f"- {t}\n")
    print(f"✅ 已保存到: {filename}")

if __name__ == "__main__":
    task = "写一篇小红书笔记，主题是：95后摆摊卖卤味月入3万的真实经历。要求贴近生活，有干货，带emoji和热门标签。"
    print("🤖 正在生成自媒体内容...\n")
    result = generate_media_content(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    save_to_markdown(result, "卤味创业笔记")