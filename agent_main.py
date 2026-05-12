import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from pypdf import PdfReader

# ---------- 配置 ----------
load_dotenv(Path(__file__).parent / ".env")  # 从项目根目录的 .env 读密钥
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME", "glm-4-flash")
EMBED_MODEL = "embedding-2"

# ---------- 向量数据库 ----------
PERSIST_DIR = str(Path(__file__).parent / ".chroma_db")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name="my_library")

# ---------- 工具函数 ----------
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
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def add_file_to_knowledge(file_path):
    """将 PDF/TXT 文件加入知识库"""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在：{file_path}")
        return

    print(f"正在解析 {file_path.name} ...")
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_path.suffix.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("❌ 不支持的文件格式，只支持 PDF 和 TXT")
        return

    chunks = split_text(text)
    file_id = hashlib.md5(text.encode()).hexdigest()
    print(f"正在向量化 {len(chunks)} 个文本块...")
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        emb = get_embedding(chunk)
        collection.add(
            embeddings=[emb],
            documents=[chunk],
            ids=[f"{file_id}_{i}"],
            metadatas=[{"source": file_path.name}],
        )
    print(f"✅ 已添加《{file_path.name}》到知识库（{len(chunks)} 个片段）")

def ask_question(question, top_k=3):
    """向知识库提问"""
    if collection.count() == 0:
        print("❌ 知识库是空的，请先添加文件。")
        return
    q_emb = get_embedding(question)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results["documents"][0]
    if not docs:
        print("⚠️ 知识库中没有找到相关内容。")
        return

    context = "\n\n".join(docs)
    prompt = f"""你是一个知识渊博的助手，根据以下资料回答问题。如果资料中没有相关信息，请如实回答“资料中未提及”。

资料：
{context}

问题：{question}
回答："""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    answer = response.choices[0].message.content
    print("\n📖 回答：")
    print(answer)
    print("\n📚 参考资料片段：")
    for i, doc in enumerate(docs, 1):
        print(f"--- 片段 {i} ---")
        print(doc[:200] + "..." if len(doc) > 200 else doc)

def clear_knowledge():
    count = collection.count()
    if count == 0:
        print("知识库已经是空的。")
        return
    collection.delete(where={})
    print(f"🗑 已清空知识库（移除了 {count} 条记录）")

def list_files():
    all_data = collection.get()
    if not all_data["metadatas"]:
        print("知识库中没有任何文件。")
        return
    sources = set()
    for meta in all_data["metadatas"]:
        sources.add(meta.get("source", "未知文件"))
    print("📚 知识库中的文件：")
    for src in sorted(sources):
        print(f"  • {src}")

def main():
    print("=" * 50)
    print("🤖 一人公司 · 知识库助手 (命令行版)")
    print("=" * 50)
    print("可用命令：")
    print("  add <文件路径>     - 添加 PDF/TXT 到知识库")
    print("  ask <问题>         - 向知识库提问")
    print("  list               - 列出知识库文件")
    print("  clear              - 清空知识库")
    print("  quit / exit        - 退出")
    print("-" * 50)

    while True:
        try:
            cmd_line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not cmd_line:
            continue

        if cmd_line.lower() in ("quit", "exit"):
            print("👋 再见！")
            break
        elif cmd_line.lower() == "list":
            list_files()
        elif cmd_line.lower() == "clear":
            clear_knowledge()
        elif cmd_line.lower().startswith("add "):
            file_path = cmd_line[4:].strip()
            add_file_to_knowledge(file_path)
        elif cmd_line.lower().startswith("ask "):
            question = cmd_line[4:].strip()
            ask_question(question)
        else:
            print("❌ 未知命令，请输入 add / ask / list / clear / quit")

if __name__ == "__main__":
    main()
