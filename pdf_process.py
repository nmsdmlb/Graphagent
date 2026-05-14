import fitz
import networkx as nx
import json
import time
import os
import re  # 新增正则提取
import pandas as pd
from openai import OpenAI
import pickle

client = OpenAI(
    api_key="sk-3d3a867afba84f16b282decc55d1c7a3",
    base_url="https://api.deepseek.com"
)


def get_triples_from_llm(text_chunk):
    system_prompt = """
    你是一个中医知识图谱专家。请从文本中提取 [实体, 关系, 实体]。
    必须以 JSON 格式输出: {"data": [[实体, 关系, 实体], ...]}
    关系限定: [包含, 功效, 主治, 属于, 性味, 归经, 配伍]
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_chunk},
            ],
            response_format={'type': 'json_object'}
        )
        content = response.choices[0].message.content

        # --- 增强的 JSON 清理逻辑 ---
        # 移除可能的 Markdown 代码块标记回车
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return data.get('data', [])
    except Exception as e:
        # 如果 JSON 解析失败，尝试用正则暴力提取（保底方案）
        print(f"⚠️ JSON 解析失败，尝试暴力提取: {e}")
        found = re.findall(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', content)
        return found


def process_single_pdf(pdf_path, G):
    print(f"\n📖 深度处理 PDF: {os.path.basename(pdf_path)}")
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if len(text.strip()) < 30: continue

            # --- 优化分块逻辑：每页分两段处理，保证覆盖率 ---
            mid = len(text) // 2
            chunks = [text[:mid + 100], text[mid - 100:]]  # 略微重叠防止截断

            for i, chunk in enumerate(chunks):
                print(f"  > 正在分析第 {page_num + 1} 页第 {i + 1} 部分...")
                triples = get_triples_from_llm(chunk)
                for item in triples:
                    if len(item) == 3:
                        u, r, v = [str(x).strip() for x in item]
                        # 使用 MultiDiGraph 允许存储多重关系
                        G.add_edge(u, v, label=r, source=os.path.basename(pdf_path))
                time.sleep(0.5)  # 控制速率
        doc.close()
    except Exception as e:
        print(f"❌ PDF 处理出错: {e}")


def inject_excel_attributes(G, excel_path):
    print(f"\n🧪 注入 Excel 数据...")
    df = pd.read_excel(excel_path)
    for _, row in df.iterrows():
        name = str(row['中药名称']).strip()
        if name == 'nan': continue

        # 注入节点属性
        G.add_node(name,
                   药性=str(row.get('性质', '')),
                   归经=str(row.get('归经', '')),
                   功效分类=str(row.get('功效分类', '')),
                   主要功效=str(row.get('主要功效', '')))

        # 显式建立分类边（非常重要！）
        cat = str(row.get('功效分类', '')).strip()
        if cat != 'nan':
            G.add_edge(name, cat, label="属于分类")
    print(f"✅ Excel 注入完成。")


if __name__ == "__main__":
    PDF_FOLDER = r"F:\Agent_project\data"
    EXCEL_PATH = r"F:\Agent_project\pdf_process\中医药数据.xlsx"

    # 【改动】使用 MultiDiGraph 允许同一对节点间有多个方向或标注
    master_graph = nx.MultiDiGraph()

    # 1. 注入 Excel
    inject_excel_attributes(master_graph, EXCEL_PATH)

    # 2. 处理 PDF
    if os.path.exists(PDF_FOLDER):
        for f in os.listdir(PDF_FOLDER):
            if f.endswith(".pdf"):
                process_single_pdf(os.path.join(PDF_FOLDER, f), master_graph)

    # 3. 保存
    save_path = "advanced_medical_graph.gpickle"
    with open(save_path, "wb") as f:
        pickle.dump(master_graph, f)

    print(f"\n📊 最终统计报告：")
    print(f"   - 节点总数: {master_graph.number_of_nodes()}")
    print(f"   - 关系总数: {master_graph.number_of_edges()}")