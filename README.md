<div align="center">

# 📊 EconWritingBench

**评测大模型在经济学论文写作辅助上的表现**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/harryhu321/EconWritingBench?style=social)](https://github.com/harryhu321/EconWritingBench)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](#english) | 中文

</div>

## 🎯 项目介绍

EconWritingBench 是一个专为**经济学科研场景**设计的大模型写作辅助评测基准。

我们把经济学论文写作拆解为 5 个高频任务，用真实经济学场景的输入，系统对比 GPT-4o、Claude、Gemini、DeepSeek 等主流模型的输出质量——关注的不是"能写出来"，而是"写出来能不能用"。

> 📌 **核心问题**：当你把回归表格扔给 AI，它会不会把系数抄错？会不会把相关性写成因果？

---

## 🏆 Leaderboard（v0.1，持续更新）

| 模型 | Overall | Task1 Related Work | Task2 回归解读 | Task3 Introduction | Task4 摘要润色 | Task5 研究设计 | 版本 | 日期 |
|------|---------|-------------------|--------------|-------------------|--------------|--------------|------|------|
| 🥇 Doubao-Pro | **4.82** | 5.00 | 4.41 | 5.00 | 4.90 | 4.89 | rubric-v0.1 | 2026-07-17 |
| 🥈 deepseek-chat | **4.62** | 4.93 | 4.75 | 5.00 | 4.42 | 4.00 | rubric-v0.1 | 2026-07-17 |
| 🥉 GLM-Z1-5.2 | **4.41** | 4.81 | 4.81 | 4.53 | 3.78 | 3.89 | rubric-v1.0 | 2026-07-17 |

> 评分范围 1–5，Overall 为各任务加权平均。完整评分细则见 [evaluation/rubrics.json](evaluation/rubrics.json)。

---

## 📋 5 个评测任务

### Task 1：Related Work 生成
- **输入**：5 篇论文标题 + 摘要
- **输出**：一段完整的文献综述（约 200–300 词）
- **评测重点**：是否虚构文献、是否能综合归纳而非逐篇罗列、学术语气

### Task 2：计量结果解读
- **输入**：回归系数表（含变量名、系数、标准误、显著性）
- **输出**：结果分析段落
- **评测重点**：数值准确性、是否区分相关/因果、统计显著性表述

### Task 3：Introduction Motivation 写作
- **输入**：研究问题 + 背景信息
- **输出**：Introduction 中的 motivation 段落
- **评测重点**：问题重要性论证、文献缺口识别、钩子设计

### Task 4：摘要顶刊风格润色
- **输入**：初稿摘要（可能含中式英语、结构松散）
- **输出**：顶刊风格修订版
- **评测重点**：语言地道性、结构规范性、信息完整性

### Task 5：研究设计建议
- **输入**：一个经济学研究想法（现象 + 初步问题）
- **输出**：实证策略建议 + 数据来源推荐
- **评测重点**：识别策略质量、可行性、威胁讨论

---

## 🚀 快速开始

```bash
git clone https://github.com/harryhu321/EconWritingBench.git
cd EconWritingBench
pip install -r requirements.txt

# 运行评测（需配置 API Key）
python scripts/run_inference.py --model gpt-4o --task all
python scripts/run_judge.py
python scripts/aggregate_scores.py
```

---

## 📁 数据集

评测数据位于 `data/samples/`，每个任务 3 条示例样本，格式为 JSON。

欢迎通过 PR 贡献更多高质量样本！

---

## 🤝 贡献

欢迎贡献新样本、新模型评测结果、评分标准改进！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📄 License

MIT License © 2025 harryhu321

---

<a name="english"></a>
## English

**EconWritingBench** is a benchmark for evaluating LLMs on economics academic writing assistance tasks.

We decompose economics paper writing into 5 high-frequency tasks and systematically compare GPT-4o, Claude, Gemini, DeepSeek, and other models — focusing not on whether they can produce output, but on whether the output is actually usable.

See the Chinese section above for full task descriptions and leaderboard.
