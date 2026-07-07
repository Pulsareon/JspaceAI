#!/usr/bin/env python3
"""从 LCCC 对话语料提取训练文本。

LCCC 格式：每行一个 JSON list，包含多轮对话，词间有空格。
输出：纯文本（去词间空格），繁简双版本，保存到 corpus/lccc_dialogues.txt

用法：
  python scripts/prepare_lccc.py [--max_lines 500000]
"""
import json
import gzip
import sys
import re
from pathlib import Path
from opencc import OpenCC

INPUT = Path(__file__).parent.parent / 'corpus' / 'lccc' / 'lccc_base_train.jsonl'
INPUT_GZ = INPUT.with_suffix('.jsonl.gz')
OUTPUT = Path(__file__).parent.parent / 'corpus' / 'lccc_dialogues.txt'

cc_t2s = OpenCC('t2s')
cc_s2t = OpenCC('s2t')


def clean_utterance(text: str) -> str:
    """清洗单句话：去词间空格、去特殊字符"""
    # 去词间空格（LCCC 用空格分词了）
    text = re.sub(r'\s+', '', text)
    # 去网址
    text = re.sub(r'https?://\S+', '', text)
    # 去多余标点
    text = re.sub(r'(.)\1{5,}', r'\1\1', text)  # 去超长重复
    return text.strip()


def main():
    max_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 500000

    # 选择输入文件
    if INPUT.exists():
        f = open(INPUT, 'r', encoding='utf-8')
    elif INPUT_GZ.exists():
        f = gzip.open(INPUT_GZ, 'rt', encoding='utf-8')
    else:
        print(f"找不到数据文件: {INPUT} 或 {INPUT_GZ}")
        return

    cc_t2s = OpenCC('t2s')
    cc_s2t = OpenCC('s2t')

    count = 0
    char_count = 0
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for line in f:
            if count >= max_lines:
                break
            try:
                dialogue = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(dialogue, list) or len(dialogue) < 2:
                continue

            # 清洗每句话
            utterances = [clean_utterance(u) for u in dialogue]
            utterances = [u for u in utterances if len(u) >= 2]
            if len(utterances) < 2:
                continue

            # 拼成一段对话文本（用换行分隔每轮）
            text_simp = '\n'.join(utterances)
            text_simp = cc_t2s.convert(text_simp)  # 转简体
            text_trad = cc_s2t.convert(text_simp)  # 转繁体

            # 写入简体版
            out.write(text_simp + '\n\n')
            # 写入繁体版
            out.write(text_trad + '\n\n')

            count += 1
            char_count += len(text_simp) + len(text_trad) + 4

            if count % 50000 == 0:
                print(f"  已处理 {count} 段对话, {char_count/10000:.0f} 万字符")

    f.close()
    print(f"\n完成: {count} 段对话, {char_count} 字符")
    print(f"保存到: {OUTPUT}")


if __name__ == '__main__':
    main()
