# Corpus 语料库总说明

本目录 `corpus/` 汇集用于模型训练的多来源中文语料。

## 1. LCCC 中文对话语料（HuggingFace 下载）
- 位置：`corpus/lccc/`
- 详见 `lccc/README.md`。
- 概要：从 HuggingFace 下载的大规模中文对话数据集。其中 `lccc_base_train.jsonl`（约 872MB）、`lccc_base_train.jsonl.gz`（约 353MB）因单个体积超过 100MB，已从 git 跟踪中移除并加入 `.gitignore` 忽略，仅保留本地文件。

## 2. 模型编写的教材（Kimi / GLM / Hy3）
- 位置：`教材/`、`义务教育教材/`
- 由 **Kimi（K2.7 Code）、GLM（5.2）、Hy3** 三个模型共同编写，覆盖**幼儿园、小学、初中、高中**各学段教材内容（含语文、数学、英语、物理、化学、生物、历史、地理、政治等学科，以及幼儿成长日志等大量 `.md` 文本）。
- 各模型分别承担不同学段 / 学科的编写工作。

## 版本控制约定
- 大体积训练语料（如 LCCC 的 jsonl / gz，>100MB）不纳入 git 提交，已忽略。
- 教材类 `.md` 文本体积小，正常纳入版本控制。
- 各语料具体说明见对应子目录文档（如 `lccc/README.md`）。
