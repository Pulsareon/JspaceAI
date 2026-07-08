#!/usr/bin/env python3
"""从 ChinaTextbook 仓库下载人教版核心教材 PDF 并提取文本。

策略：
  1. 选 30 本最重要的人教版教材（小学语文数学 + 初中核心科目）
  2. 下载 PDF（用 raw.githubusercontent.com）
  3. 用 pypdf 提取文本
  4. 保存到 corpus/china_textbook/<科目>_<年级>.txt
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from pypdf import PdfReader
import io

OUT_DIR = Path(__file__).parent.parent / 'corpus' / 'china_textbook'
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO = 'TapXWorld/ChinaTextbook'
BRANCH = 'master'
RAW_BASE = f'https://raw.githubusercontent.com/{REPO}/{BRANCH}/'

# 30 本核心教材（人教版/统编版）
SELECTED_PDFS = [
    # 小学语文
    '小学/语文/统编版-人民教育出版社/一年级/义务教育教科书·语文一年级上册.pdf',
    '小学/语文/统编版-人民教育出版社/一年级/义务教育教科书·语文一年级下册.pdf',
    '小学/语文/统编版-人民教育出版社/二年级/义务教育教科书·语文二年级上册.pdf',
    '小学/语文/统编版-人民教育出版社/二年级/义务教育教科书·语文二年级下册.pdf',
    '小学/语文/统编版-人民教育出版社/三年级/义务教育教科书·语文三年级上册.pdf',
    '小学/语文/统编版-人民教育出版社/三年级/义务教育教科书·语文三年级下册.pdf',
    '小学/语文/统编版-人民教育出版社/四年级/义务教育教科书·语文四年级上册.pdf',
    '小学/语文/统编版-人民教育出版社/五年级/义务教育教科书·语文五年级上册.pdf',
    '小学/语文/统编版-人民教育出版社/六年级/义务教育教科书·语文六年级上册.pdf',
    # 小学数学
    '小学/数学/人教版-人民教育出版社/一年级/义务教育教科书·数学一年级上册.pdf',
    '小学/数学/人教版-人民教育出版社/二年级/义务教育教科书·数学二年级上册.pdf',
    '小学/数学/人教版-人民教育出版社/三年级/义务教育教科书·数学三年级上册.pdf',
    '小学/数学/人教版-人民教育出版社/四年级/义务教育教科书·数学四年级上册.pdf',
    '小学/数学/人教版-人民教育出版社/五年级/义务教育教科书·数学五年级上册.pdf',
    '小学/数学/人教版-人民教育出版社/六年级/义务教育教科书·数学六年级上册.pdf',
    # 小学英语
    '小学/英语/人教版-人民教育出版社/三年级/义务教育教科书·英语三年级上册.pdf',
    '小学/英语/人教版-人民教育出版社/四年级/义务教育教科书·英语四年级上册.pdf',
    '小学/英语/人教版-人民教育出版社/五年级/义务教育教科书·英语五年级上册.pdf',
    # 小学科学
    '小学/科学/人教版-人民教育出版社/一年级/义务教育教科书·科学一年级上册.pdf',
    '小学/科学/人教版-人民教育出版社/三年级/义务教育教科书·科学三年级上册.pdf',
    # 初中
    '初中/语文/统编版-人民教育出版社/七年级/义务教育教科书·语文七年级上册.pdf',
    '初中/语文/统编版-人民教育出版社/八年级/义务教育教科书·语文八年级上册.pdf',
    '初中/语文/统编版-人民教育出版社/九年级/义务教育教科书·语文九年级上册.pdf',
    '初中/数学/人教版-人民教育出版社/七年级/义务教育教科书·数学七年级上册.pdf',
    '初中/数学/人教版-人民教育出版社/八年级/义务教育教科书·数学八年级上册.pdf',
    '初中/英语/人教版-人民教育出版社/七年级/义务教育教科书·英语七年级上册.pdf',
    '初中/物理/人教版-人民教育出版社/八年级/义务教育教科书·物理八年级上册.pdf',
    '初中/化学/人教版-人民教育出版社/九年级/义务教育教科书·化学九年级上册.pdf',
    '初中/生物学/人教版-人民教育出版社/七年级/义务教育教科书·生物学七年级上册.pdf',
    '初中/历史/统编版-人民教育出版社/七年级/义务教育教科书·中国历史七年级上册.pdf',
]


def download_pdf(url: str) -> bytes:
    """下载 PDF 返回 bytes"""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (educational corpus fetcher)'
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def extract_text(pdf_bytes: bytes) -> str:
    """从 PDF bytes 提取文本"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texts.append(t.strip())
        return '\n\n'.join(texts)
    except Exception as e:
        print(f"    提取失败: {e}")
        return ''


def main():
    total_chars = 0
    success = 0
    for i, pdf_path in enumerate(SELECTED_PDFS):
        # 生成输出文件名
        name = pdf_path.split('/')[-1].replace('.pdf', '').replace('义务教育教科书·', '')
        out_file = OUT_DIR / f"{name}.txt"

        if out_file.exists() and out_file.stat().st_size > 100:
            print(f"[{i+1}/{len(SELECTED_PDFS)}] 跳过（已存在）: {name}")
            total_chars += out_file.stat().st_size
            success += 1
            continue

        url = RAW_BASE + urllib.request.quote(pdf_path, safe='/')
        print(f"[{i+1}/{len(SELECTED_PDFS)}] 下载: {name}...", end=' ', flush=True)
        try:
            pdf_bytes = download_pdf(url)
            print(f"{len(pdf_bytes)//1024}KB", end=' ', flush=True)
            text = extract_text(pdf_bytes)
            if text:
                out_file.write_text(text, encoding='utf-8')
                print(f"→ {len(text)} 字符")
                total_chars += len(text)
                success += 1
            else:
                print("→ 无文本（可能是扫描PDF）")
        except Exception as e:
            print(f"失败: {e}")
        time.sleep(1)  # 礼貌延时

    print(f"\n完成: {success}/{len(SELECTED_PDFS)} 成功, {total_chars} 字符")
    print(f"保存到: {OUT_DIR}")


if __name__ == '__main__':
    main()
