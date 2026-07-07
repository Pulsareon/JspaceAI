#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从电子课本网（dzkbw.com）抓取义务教育阶段教材目录，并按阶段/年级/科目/版本建立文件夹。

说明：
- dzkbw.com 本身是导航站，教材正文托管在国家中小学智慧教育平台（basic.smartedu.cn）。
- 本脚本会解析出每个章节的官方直达链接，写入各本书的 Markdown 索引。
- 可开启 --search，使用 DuckDuckGo Lite 按章节标题搜索网络补充资料。
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "http://www.dzkbw.com"
SMARTEDU_BASE = "https://basic.smartedu.cn/tchMaterial/detail"
DDG_URL = "https://lite.duckduckgo.com/lite/"
JINA_BASE = "https://r.jina.ai/http://"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

PREFERRED_VERSIONS = [
    "人教版", "部编版", "北师大版", "苏教版",
    "外研版", "译林版", "教科版", "青岛版", "北京版",
    "沪教版", "冀教版", "鲁教版", "湘教版", "浙教版",
]

GRADE_RE = re.compile(r"(五四制)?\s*([一二三四五六七八九十]+年级|高[一二三]|初[一二三])([上下]?)册?")
SAFE_RE = re.compile(r'[\\/*?:"<>|]+')

KINDERGARTEN_STAGES = ["小班", "中班", "大班"]
KINDERGARTEN_SUBJECTS = ["语言", "健康", "社会", "科学", "艺术", "数学启蒙", "英语启蒙"]


class Cache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, url: str):
        return self.data.get(url)

    def set(self, url: str, value: str):
        self.data[url] = value
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch_text(url: str, session: requests.Session, cache: Cache, delay: float = 0.3) -> str:
    cached = cache.get(url)
    if cached is not None:
        return cached
    if delay > 0:
        time.sleep(delay)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    if "dzkbw.com" in url:
        text = resp.content.decode("gbk", errors="ignore")
    else:
        text = resp.text
    cache.set(url, text)
    return text


def dzkbw_decode(encoded: str) -> str:
    n = len(encoded)
    k = 3 + n % 4
    acc = n * 7
    half = n // 2
    encoded = encoded[n - half:] + encoded[:n - half]
    out = []
    for i in range(1, n + 1):
        ch = encoded[i - 1]
        code = ord(ch) + acc % k + k // 2
        out.append(chr(code))
        acc = ord(out[-1])
    return "".join(out)


def extract_content_id(chapter_html: str) -> str | None:
    m = re.search(r'<a\s+id=[\'"]gourl[\'"]\s+[^>]*?href="(/go/[^"]+)"', chapter_html, re.I)
    if not m:
        return None
    # 必须直接解析原始 href，否则 url 参数里的 %23 会被提前解码为 #，导致查询字符串被截断。
    href = m.group(1)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    encoded = q.get("url", [""])[0]
    if not encoded:
        return None
    encoded = urllib.parse.unquote(encoded)
    decoded = dzkbw_decode(encoded)
    cm = re.search(r"contentId=([0-9a-f\-]+)", decoded)
    return cm.group(1) if cm else None


def smartedu_url(content_id: str, page_code: str) -> str:
    page = page_code.zfill(3)
    return (
        f"{SMARTEDU_BASE}?contentType=assets_document"
        f"&contentId={content_id}&catalogType=tchMaterial"
        f"&subCatalog=tchMaterial&page={page}"
    )


def safe_name(s: str) -> str:
    return SAFE_RE.sub("_", s).strip(" ._")


def parse_home(html: str):
    """返回 [(stage, stage_slug, subject_slug, subject_name), ...]"""
    # 支持按阶段找课本和年级找课本两个导航区
    patterns = [
        (r'按阶段找课本.*?</li>', True),
        (r'<li class="xl">按年级找课本.*?</li>', True),
    ]
    seen = set()
    items = []
    for pat, use_in_list in patterns:
        m = re.search(pat, html, re.S | re.I)
        if not m:
            continue
        nav = m.group(0)
        for a in re.finditer(
            r'<a\s+href="/books/(xiaoxue|chuzhong|gaozhong)-([^"/]+)/"\s+title="([^"]+)"[^>]*>([^<]+)</a>',
            nav,
            re.I,
        ):
            stage_slug, subject_slug, title_attr, text = a.groups()
            stage_map = {"xiaoxue": "小学", "chuzhong": "初中", "gaozhong": "高中"}
            stage = stage_map[stage_slug]
            subject_name = text.replace(stage, "").strip()
            key = (stage, subject_slug)
            if key in seen:
                continue
            seen.add(key)
            items.append((stage, stage_slug, subject_slug, subject_name))
    return items


def parse_subject_page(html: str, stage: str, subject_name: str, only_preferred: bool = True):
    """返回图书列表，按版本分组过滤。"""
    sections = re.split(r'<DIV\s+class=i_d_tc[^>]*>', html, flags=re.I)
    books = []
    for sec in sections[1:]:
        header_match = re.match(r"([^<]+)</DIV>", sec, re.I)
        if not header_match:
            continue
        version_heading = header_match.group(1).strip()
        if only_preferred and not any(v in version_heading for v in PREFERRED_VERSIONS):
            continue
        ul_part = sec.split("</UL>", 1)[0]
        for li in re.finditer(r"<LI[^>]*>(.*?)</LI>", ul_part, re.S | re.I):
            li_html = li.group(1)
            m1 = re.search(r'<A\s+class="mba"\s+href="([^"]+)"', li_html, re.I)
            m2 = re.search(
                r'<A\s+class="ih3"\s+href="[^"]+"\s+title="([^"]+)"[^>]*>([^<]*)</A>',
                li_html,
                re.I,
            )
            if not (m1 and m2):
                continue
            book_url = m1.group(1)
            book_title = m2.group(1).strip()
            gm = GRADE_RE.search(book_title)
            grade = gm.group(2) if gm else "其他"
            books.append(
                {
                    "version_heading": version_heading,
                    "url": book_url,
                    "title": book_title,
                    "grade": grade,
                }
            )
    return books


def parse_book_page(html: str, book_url: str):
    title_match = re.search(r"<title>([^<]+)</title>", html, re.I)
    book_title = title_match.group(1).strip() if title_match else ""

    book_path = book_url.rstrip("/")

    # 封面图：优先匹配本书路径下的 lazy 图片
    cover_match = re.search(
        rf'<img[^>]*?class=[\'"]lazy[\'"][^>]*?data-original="({re.escape(book_path)}/cover\.jpg)"[^>]*?>',
        html,
        re.I,
    )
    cover = cover_match.group(1) if cover_match else ""

    # 目录：本书目录页链接统一为 /books/.../{book_id}/{code}.htm
    toc = []
    seen = set()
    pattern = rf'<A\s+[^>]*?href="({re.escape(book_path)}/([0-9]*)\.htm)"[^>]*>(.*?)</A>'
    for a in re.finditer(pattern, html, re.S | re.I):
        href, code, txt = a.groups()
        code = (code or "000").zfill(3)
        if href in seen:
            continue
        seen.add(href)
        title = re.sub(r"<[^>]+>", "", txt).strip()
        if title:
            toc.append(
                {
                    "code": code,
                    "title": title,
                    "dzkbw_url": BASE_URL + href,
                }
            )
    return {"title": book_title, "cover": cover, "toc": toc}


def search_ddg(query: str, max_results: int = 3):
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "text/html",
        "Referer": "https://lite.duckduckgo.com/lite/",
    }
    try:
        resp = requests.post(
            DDG_URL, data={"q": query, "kl": "zh-CN"}, headers=headers, timeout=20
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  [search error] {e}")
        return []
    results = []
    for m in re.finditer(
        r'<a\s+rel="nofollow"\s+href="([^"]+)"\s+class=[\'"]result-link[\'"]\s*>([^<]+)</a>',
        resp.text,
        re.I,
    ):
        url = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not title or not url.startswith("http"):
            continue
        results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break
    return results


def write_book_markdown(
    output_root: Path,
    stage: str,
    grade: str,
    subject_name: str,
    book: dict,
    info: dict,
    supplements: dict,
):
    folder = output_root / safe_name(stage) / safe_name(grade) / safe_name(subject_name)
    folder.mkdir(parents=True, exist_ok=True)

    version = safe_name(book["version_heading"].replace(stage, "").replace(subject_name, "").strip() or "未知版本")
    filename = f"{version}_{safe_name(book['title'])}.md"
    filepath = folder / filename

    lines = []
    lines.append(f"# {book['title']}")
    lines.append("")
    lines.append(f"- **学段**：{stage}")
    lines.append(f"- **年级**：{book['grade']}")
    lines.append(f"- **科目**：{subject_name}")
    if book.get("version_heading"):
        lines.append(f"- **版本**：{book['version_heading']}")
    if book.get("url"):
        lines.append(f"- **电子课本网目录**：{BASE_URL}{book['url']}")
    else:
        lines.append(f"- **资料来源**：网络搜索整理")
    if info.get("cover"):
        lines.append(f"- **封面图**：{BASE_URL}{info['cover']}")
    lines.append("")
    lines.append("## 目录与官方阅读链接")
    lines.append("")
    lines.append("| 序号 | 章节 | 电子课本网 | 智慧教育平台 |")
    lines.append("|------|------|------------|--------------|")
    for ch in info["toc"]:
        smart = ch.get("smartedu_url", "")
        dzkbw = ch.get("dzkbw_url", "")
        dzkbw_link = f"[查看]({dzkbw})" if dzkbw else "-"
        smart_link = f"[官方阅读]({smart})" if smart else "-"
        lines.append(
            f"| {ch['code']} | {ch['title']} | {dzkbw_link} | {smart_link} |"
        )
    lines.append("")

    if supplements:
        lines.append("## 网络补充资料")
        lines.append("")
        for ch_title, results in supplements.items():
            lines.append(f"### {ch_title}")
            if results:
                for r in results:
                    lines.append(f"- [{r['title']}]({r['url']})")
            else:
                lines.append("- 未找到结果")
            lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="构建义务教育阶段教材目录语料")
    parser.add_argument("--output", default="corpus/义务教育教材", help="输出根目录")
    parser.add_argument("--cache", default=".cache/dzkbw_cache.json", help="HTTP 缓存文件")
    parser.add_argument("--delay", type=float, default=0.25, help="请求间隔（秒）")
    parser.add_argument("--max-books-per-subject", type=int, default=0, help="每科目最多处理的书本数，0 为不限")
    parser.add_argument("--only-preferred", action="store_true", default=True, help="仅保留主流版本")
    parser.add_argument("--search", action="store_true", help="是否按章节搜索网络补充资料")
    parser.add_argument("--search-chapters", type=int, default=5, help="每本书搜索网络补充资料的章节数")
    parser.add_argument("--search-results", type=int, default=3, help="每章节保留的搜索结果数")
    args = parser.parse_args()

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache = Cache(Path(args.cache).resolve())
    session = get_session()

    print(f"输出目录：{output_root}")
    print("正在解析 dzkbw.com 首页...")
    home_html = fetch_text(BASE_URL + "/", session, cache, delay=args.delay)
    subjects = parse_home(home_html)
    print(f"发现义务教育阶段科目：{len(subjects)} 个")

    total_books = 0
    for stage, stage_slug, subject_slug, subject_name in subjects:
        subject_url = f"{BASE_URL}/books/{stage_slug}-{subject_slug}/"
        print(f"\n[{stage}] {subject_name} -> {subject_url}")
        try:
            subj_html = fetch_text(subject_url, session, cache, delay=args.delay)
        except Exception as e:
            print(f"  跳过（获取失败）：{e}")
            continue

        books = parse_subject_page(subj_html, stage, subject_name, args.only_preferred)
        if not books:
            print("  没有匹配到书本")
            continue
        if args.max_books_per_subject:
            books = books[: args.max_books_per_subject]

        for book in books:
            print(f"  - {book['title']} ({book['version_heading']})")
            try:
                book_html = fetch_text(
                    BASE_URL + book["url"], session, cache, delay=args.delay
                )
                info = parse_book_page(book_html, book["url"])
                if not info["toc"]:
                    print("    未解析到目录，跳过")
                    continue

                first_chapter_html = fetch_text(
                    info["toc"][0]["dzkbw_url"], session, cache, delay=args.delay
                )
                content_id = extract_content_id(first_chapter_html)
                if content_id:
                    for ch in info["toc"]:
                        ch["smartedu_url"] = smartedu_url(content_id, ch["code"])

                supplements = {}
                if args.search:
                    for ch in info["toc"][: args.search_chapters]:
                        q = f"{book['title']} {ch['title']}"
                        supplements[ch["title"]] = search_ddg(
                            q, max_results=args.search_results
                        )
                        if args.delay > 0:
                            time.sleep(args.delay)

                filepath = write_book_markdown(
                    output_root, stage, book["grade"], subject_name, book, info, supplements
                )
                total_books += 1
                print(f"    已生成：{filepath.relative_to(output_root)}")
            except Exception as e:
                print(f"    处理失败：{e}")
                continue

    print(f"\n完成，共生成 {total_books} 本书的索引。")


if __name__ == "__main__":
    main()
