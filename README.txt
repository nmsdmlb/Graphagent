# TCM-GraphAgent: 知识图谱驱动的中医临床推理智能体

##项目介绍
本项目是一个专为中医领域设计的图谱增强型智能体，通过将大规模异构医学知识图谱融入 Agent 推理环路，解决了传统 RAG 在处理中医配伍逻辑、归经推理时“看得到，想不深”的痛点

##项目核心架构
1.图谱构建 → 异构数据融合（PDF 关系提取 + Excel 属性注入）
2.知识存储 → NetworkX MultiDiGraph（多维有向图存储）
3.语义对齐 → SentenceTransformer 向量化（解决实体别名与孤岛问题）
4.意图解析 → LLM 关键词提取（从中医查询中识别方剂、药名）
5.语义找词 → 相似度计算（将非标准术语映射至图谱标准节点）
6.多步检索 → 2-Hop 子图遍历（自动化抓取“关系链”与“底层药性属性”）
7.逻辑组装 → 推理证据链构建（去重、截断并结构化上下文）
8.Agent 决策 → 思维链推理（基于性味归经比例进行临床研判）
9.结果生成 → 结构化、可溯源的导师级中医建议

##核心功能
1.异构图谱引擎：支持同一实体间的多种逻辑连接（包含，药效，禁忌）。
2.语义找词系统：利用向量库纠正非规范节点名，确保属性注入链路完整。
3.2-Hop 推理增强：Agent 自动深入图谱二层逻辑，挖掘隐含的药性关联。
4.逻辑评分保障：针对复杂判断题（如：药性对脏腑平衡的影响）实现高分突破。

##技术栈
推理大脑：DeepSeek-V3 (DeepSeek-Chat)
图引擎：NetworkX (MultiDiGraph)
语义对齐：Sentence-Transformers (paraphrase-multilingual)
数据操作：Pandas / Pickle / Openpyxl
基础环境：Python 3.9+ / Torch

##项目亮点：
1.突破 1-Hop 局限：实现多级邻居属性聚合，支撑复杂临床逻辑推演。
2.实体对齐优化：自研语义找词模块，将 PDF 提取的非标准实体与结构化词库完美对齐。
3.逻辑可靠性强：基于图谱事实回填，有效抑制大模型的“医学幻觉”。
4.架构模组化：解耦了知识提取、图谱构建与智能体检索，支持快速迁移至其他垂直领域。

##快速启动
1.环境配置：
conda create -n graphagent python=3.9
pip install -r requirements.tx
2.配置 API Key：DEEPSEEK_API_KEY = "sk-3d3a867afba84f16b282decc55d1c7a3"
3.图谱初始化：运行 PDF 和excel处理脚本提取基础关系：python pdf_process.py
4.执行主程序进入对话：python graph_generate.py
5.系统评估：运行评估脚本，查看 无图谱vs1-Hop vs 2-Hop 的逻辑得分改进：python graph_evaluator.py