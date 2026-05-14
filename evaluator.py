import pandas as pd
import json
from graph_generate import medical_agent_chat, client  # 复用主程序里的 client 进行打分

test_cases = [
    {"type": "深度关联", "query": "荆防败毒散里用到了川芎，川芎的药性和归经是如何配合该方‘散寒祛湿’的？"},
    {"type": "深度属性", "query": "银翘散中是否含有属于“化痰止咳平喘药”分类的药物？其具体功效是什么？"},
    {"type": "症状溯源", "query": "针对‘颈项强痛’，荆防败毒散里的哪些药在起作用？"},
    {"type": "用药禁忌", "query": "如果患者脾胃虚寒，在使用银翘散时需要注意哪味药？为什么？"},
    {"type": "多门类对比","query": "对比银翘散和荆防败毒散，哪一个更适合治疗带有‘热象’的感冒？请根据成分的温凉性质给出证据。"},
]


def run_evaluation():
    results = []

    for case in test_cases:
        query = case['query']
        print(f"\n>>> 正在评估问题: {query}")

        # 1. 获取三组不同深度的回答
        print("正在获取 [0跳/无图谱] 回答...")
        ans_0 = medical_agent_chat(query, max_hops=0)

        print("正在获取 [1跳图谱] 回答...")
        ans_1 = medical_agent_chat(query, max_hops=1)

        print("正在获取 [2跳图谱] 回答...")
        ans_2 = medical_agent_chat(query, max_hops=2)

        # 2. 调用裁判打分
        print("正在请求裁判打分...")
        score_result = judge_performance_three_way(query, ans_0, ans_1, ans_2)

        results.append({
            "问题": query,
            "无图谱分数": score_result.get('score_0'),
            "一跳分数": score_result.get('score_1'),
            "二跳分数": score_result.get('score_2'),
            "二跳带来的核心提升": score_result.get('improvement')
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("three_way_eval_report.csv", index=False, encoding='utf-8-sig')
    print("\n评估完成！请查看 three_way_eval_report.csv")


def judge_performance_three_way(query, ans0, ans1, ans2):
    prompt = f"""
    你是一名专业的中医导师。请对比以下三个 AI 回答的质量并给出评分（0-10分）。

    【问题】: {query}

    【回答 A (无外部知识)】: {ans0}
    【回答 B (仅一级关联)】: {ans1}
    【回答 C (深度两级关联)】: {ans2}

    评分标准：
    1. 准确性：是否基于事实？
    2. 深度：回答 C 是否通过“第二步联系”发现了一些 B 没发现的隐藏信息？
    3. 幻觉：AI 是否在没有依据的情况下胡乱推论？

    请严格输出 JSON 格式（不要有其他文字）：
    {{ "score_0": 0-10, "score_1": 0-10, "score_2": 0-10, "improvement": "说明 C 相比于 B 发现了哪些更深层的信息" }}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={'type': 'json_object'}
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    run_evaluation()