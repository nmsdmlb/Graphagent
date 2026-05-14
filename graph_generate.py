import networkx as nx
from openai import OpenAI
import numpy as np
import os
import pickle
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

# 1. 初始化配置
client = OpenAI(
    api_key="sk-3d3a867afba84f16b282decc55d1c7a3",
    base_url="https://api.deepseek.com"
)

# 加载你生成的图谱
with open(r"F:\Agent_project\pdf_process\advanced_medical_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# --- 新增：初始化向量模型 ---
print("正在初始化向量引擎以支持模糊语义匹配...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 新增：预计算图谱节点的向量索引 ---
all_nodes = list(G.nodes())
node_embeddings = embed_model.encode(all_nodes, convert_to_tensor=True)
# 注：将所有节点名称转为向量，存入内存，供后面比对

def get_semantic_node(user_keyword, all_nodes, node_embeddings, threshold=0.6):
    """
    语义找词：如果用户说的词不在图谱里，找一个最接近的词
    """
    from sentence_transformers import util
    query_embedding = embed_model.encode([user_keyword], convert_to_tensor=True)
    # 计算余弦相似度
    cos_scores = util.cos_sim(query_embedding, node_embeddings)[0]
    # 找到最匹配的索引
    import numpy as np
    top_result = np.argmax(cos_scores.cpu())
    score = cos_scores[top_result].item()

    if score >= threshold:
        return all_nodes[top_result]
    return None


def get_graph_context(user_query, max_hops=2):
    # 1. 提取关键词
    prompt = f"从文本中提取所有中医方剂名或中药名，用逗号分隔，不要多余文字: {user_query}"
    resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
    keywords = [k.strip() for k in resp.choices[0].message.content.split(",") if k.strip()]

    structured_context = {}
    visited_edges = set()  # (u, v, label)

    for word in keywords:
        # 语义找词（all_nodes 建议在函数外定义好作为全局变量，不要每次都算）
        target = word if word in G else get_semantic_node(word, all_nodes, node_embeddings)
        if not target: continue

        if target not in structured_context:
            structured_context[target] = []

        # --- A. 提取目标节点自身的属性 (Excel 注入的数据) ---
        target_attrs = {k: v for k, v in G.nodes[target].items() if k != 'label'}
        if target_attrs:
            attr_desc = " | ".join([f"{k}: {v}" for k, v in target_attrs.items()])
            structured_context[target].append(f"【{target}】的百科属性: {attr_desc}")

        # --- B. 遍历邻居 (处理 MultiDiGraph) ---
        # 针对 MultiDiGraph，需要通过 G.edges(data=True) 或者 G[u][v] 的 keys 遍历
        for neighbor in G.neighbors(target):
            edge_data = G.get_edge_data(target, neighbor)
            # 因为是 MultiDiGraph，edge_data 是一个字典 {0: {...}, 1: {...}}
            for idx in edge_data:
                rel = edge_data[idx].get('label', '相关')
                structured_context[target].append(f"直接关系: {target} --({rel})--> {neighbor}")

                # --- C. 二跳深度检索 (解决 Q2, Q3 的核心提升) ---
                if max_hops >= 2:
                    # 1. 抓取邻居的属性 (这是解决 Q2 '化痰药' 归类的关键)
                    n_attrs = {k: v for k, v in G.nodes[neighbor].items() if k != 'label'}
                    if n_attrs:
                        n_attr_desc = " | ".join([f"{k}: {v}" for k, v in n_attrs.items()])
                        structured_context[target].append(f"  └─ 成分【{neighbor}】的详细药性: {n_attr_desc}")

                    # 2. 抓取邻居的再下一级关系
                    for deep_n in G.neighbors(neighbor):
                        if deep_n == target: continue
                        deep_edge_data = G.get_edge_data(neighbor, deep_n)
                        for d_idx in deep_edge_data:
                            rel2 = deep_edge_data[d_idx].get('label', '具有功效')
                            edge_key = tuple(sorted((neighbor, deep_n, rel2)))
                            if edge_key not in visited_edges:
                                structured_context[target].append(f"  └─ 逻辑链: {neighbor} --({rel2})--> {deep_n}")
                                visited_edges.add(edge_key)

    # 格式化输出
    final_output = ""
    for entity, infos in structured_context.items():
        final_output += f"\n### [{entity}] 相关的研判证据 ###\n"
        # 加上去重处理，防止同一关系多次出现
        unique_infos = list(dict.fromkeys(infos))
        final_output += "\n".join(unique_infos[:60]) + "\n"

    return final_output

def medical_agent_chat(user_input, max_hops=2):
    context = get_graph_context(user_input, max_hops=max_hops)

    # 调整规则：不仅看直接描述，还要看“属性发现”里的药性进行组合推理
    system_prompt = f"""你是一个专业的中医临床导师。请基于提供的【证据链】回答用户问题。

    【推理指引】：
    1. **对比逻辑**：如果涉及两个方剂对比，请分别总结它们的药性分布（例如：A方中有X味寒性药，B方中有Y味温性药）。
    2. **脏腑定位**：如果问题涉及“脾胃”或具体的身体部位，请重点检查药材属性中的“归经”字段。例如，归经包含“胃”或“脾”的药材对脾胃影响更直接。
    3. **证据至上**：如果证据链中存在信息，禁止说“证据不足”。请计算证据中寒凉药物与温热药物的比例来得出结论。

    【已知核心证据】：
    {context}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


# --- 运行测试 ---
if __name__ == "__main__":
    print("AI: 你好，我是基于《中成药治疗指南》的智能助手，请问有什么可以帮您？")
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ['exit', 'quit', '再见']: break

        answer = medical_agent_chat(user_input)
        print(f"\nAI: {answer}\n")