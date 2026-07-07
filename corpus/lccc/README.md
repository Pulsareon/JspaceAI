# LCCC 训练数据说明

本目录 `corpus/lccc/` 存放 LCCC 中文对话语料相关的**训练数据**：

> **数据来源**：本目录语料为从 HuggingFace 下载的中文对话语料（LCCC，大规模中文对话数据集），仅作本地训练使用。

- `lccc_base_train.jsonl` / `lccc_base_train.jsonl.gz`：基础训练语料（对话数据）
- `lccc_dialogues.txt`：对话文本导出

数据处理脚本 `scripts/prepare_lccc.py` 位于仓库 `scripts/` 目录，属于代码，**正常纳入版本控制**。

## 版本控制说明

LCCC 训练数据为生成式语料、体积较大，**不纳入 git 提交 / 跟踪**。

已通过 `git rm --cached` 取消跟踪（仅保留本地文件，不删除磁盘数据），并在仓库根 `.gitignore` 中追加了忽略条目。请勿将其提交到远程仓库。

### 已从 git 移除的语料（体积大于 100MB）

以下语料因单个文件体积超过 100MB，已从 git 跟踪中移除，仅保留在本地磁盘，不进入版本库 / 远程仓库：

| 文件 | 体积 | 内容说明 |
| --- | --- | --- |
| `lccc_base_train.jsonl` | 约 872 MB | LCCC 基础训练语料（原始 JSONL 对话数据，未压缩） |
| `lccc_base_train.jsonl.gz` | 约 353 MB | 上述语料的 gzip 压缩版本 |

这两个文件已执行 `git rm --cached` 取消跟踪，并在仓库根 `.gitignore` 中显式忽略；全仓库扫描确认已无大于 100MB 的被跟踪文件。

### 仍纳入版本控制的文件

- `scripts/prepare_lccc.py`：数据处理脚本（位于仓库 `scripts/` 目录），正常跟踪。
- 本说明文档 `README.md`。

如需在本地使用训练语料，直接读取本目录下的 `lccc_base_train.jsonl` / `lccc_base_train.jsonl.gz` 文件即可。
