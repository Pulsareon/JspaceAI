#!/usr/bin/env python3
"""从维基百科 API 抓取九年义务教育全科目中文语料。

科目覆盖：语文(古诗词/文言文) 数学 英语 化学 政治 历史 地理 物理 生物
每个科目抓多个核心词条的纯文本摘要，落地到 corpus/<subject>.txt
"""
import json
import urllib.request
import urllib.parse
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / 'corpus'
OUT_DIR.mkdir(exist_ok=True)

# 各科目核心词条（九年义务教育范围）
SUBJECTS = {
    '语文': [
        '唐诗', '宋词', '元曲', '文言文', '李白', '杜甫', '白居易',
        '苏轼', '李清照', '辛弃疾', '鲁迅', '朱自清', '老舍',
        '三字经', '弟子规', '千字文', '百家姓', '论语', '孟子',
    ],
    '数学': [
        '加减乘除', '分数', '小数', '百分比', '方程', '几何',
        '三角形', '圆', '正方形', '长方形', '平行四边形',
        '面积', '体积', '整数', '有理数', '无理数',
        '代数', '函数', '勾股定理', '圆周率',
    ],
    '英语': [
        '英文字母', '英语语法', '英语动词', '英语名词',
        '英语形容词', '英语时态', '现在进行时', '一般现在时',
        '一般过去时', '一般将来时', '英语单词',
    ],
    '化学': [
        '化学元素', '原子', '分子', '离子', '化合物', '混合物',
        '氧气', '氢气', '二氧化碳', '水', '酸', '碱', '盐',
        '化学反应', '氧化还原反应', '燃烧', '溶液', '金属',
        '元素周期表', '碳', '铁', '铜', '铝',
    ],
    '政治': [
        '公民', '权利', '义务', '法律', '宪法', '人民代表大会制度',
        '中国共产党', '社会主义', '改革开放', '社会主义核心价值观',
        '国家安全', '消费者权益', '未成年人保护法',
    ],
    '历史': [
        '中国历史', '夏朝', '商朝', '周朝', '秦朝', '汉朝',
        '唐朝', '宋朝', '元朝', '明朝', '清朝',
        '鸦片战争', '辛亥革命', '中华人民共和国', '抗日战争',
        '丝绸之路', '四大发明', '孔子', '秦始皇', '汉武帝',
    ],
    '地理': [
        '地球', '大气层', '水循环', '气候', '地形', '平原',
        '高原', '山地', '盆地', '丘陵', '河流', '湖泊',
        '海洋', '中国地理', '长江', '黄河', '珠江', '青藏高原',
        '塔里木盆地', '华北平原', '中国省级行政区', '北京', '上海',
    ],
    '物理': [
        '力', '运动', '速度', '加速度', '牛顿运动定律',
        '重力', '摩擦力', '弹力', '浮力', '压强',
        '功', '能量', '动能', '势能', '机械能守恒',
        '电', '电流', '电压', '电阻', '欧姆定律',
        '磁', '电磁感应', '光', '反射', '折射', '声音',
        '热传导', '温度', '比热容',
    ],
    '生物': [
        '细胞', '细胞膜', '细胞核', '细胞质', 'DNA',
        '光合作用', '呼吸作用', '新陈代谢',
        '植物', '动物', '细菌', '病毒', '真菌',
        '生态系统', '食物链', '生物多样性',
        '人体', '心脏', '肺', '肝脏', '胃',
        '血液循环', '神经系统', '消化系统', '生殖', '遗传',
    ],
}


def fetch_extract(title: str) -> str:
    """从维基百科 API 获取词条纯文本摘要"""
    api = "https://zh.wikipedia.org/w/api.php"
    params = urllib.parse.urlencode({
        'action': 'query',
        'titles': title,
        'prop': 'extracts',
        'explaintext': '1',
        'exsectionformat': 'plain',
        'format': 'json',
        'redirects': '1',
    })
    url = f"{api}?{params}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'JspaceAI/1.0 (educational corpus)'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        pages = data.get('query', {}).get('pages', {})
        for pid, page in pages.items():
            if pid != '-1' and 'extract' in page:
                return page['extract']
    except Exception as e:
        print(f"  [warn] {title}: {e}")
    return ''


def main():
    total_chars = 0
    for subject, titles in SUBJECTS.items():
        out_file = OUT_DIR / f"{subject}.txt"
        texts = []
        for title in titles:
            print(f"  抓取 [{subject}] {title}...", end=' ', flush=True)
            text = fetch_extract(title)
            if text:
                # 截取前 2000 字符避免过长
                text = text[:2000]
                texts.append(f"## {title}\n{text}")
                print(f"{len(text)} 字符")
            else:
                print("无")
            time.sleep(2.0)  # 礼貌延时，避免 429

        content = f"\n\n=== {subject} ===\n\n" + "\n\n".join(texts)
        out_file.write_text(content, encoding='utf-8')
        print(f"[{subject}] 保存到 {out_file.name}: {len(content)} 字符")
        total_chars += len(content)

    print(f"\n完成，共 {total_chars} 字符，{len(SUBJECTS)} 个科目文件")


if __name__ == '__main__':
    main()
