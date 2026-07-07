"""
字符级 tokenizer + 数据集

为什么字符级而不是 subword：
    - 词汇表小（~100），模型可以小，验证架构用
    - 字符级有明确的"风格"信号（拼写、标点、节奏）
    - 持续学习场景下，subword 词汇表会变，字符级稳定

数据：用经典 Shakespeare 文本作为持续学习的语料。
也可以换成任何 UTF-8 文本。
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class CharTokenizer:
    """字符级 tokenizer，支持训练时新字符的动态加入

    未知字符统一映射到 idx 0（unk）。构建时 idx 0 固定是 '<unk>'。
    """
    chars: list[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @classmethod
    def from_text(cls, text: str, max_vocab: int | None = None) -> "CharTokenizer":
        """从文本构建 tokenizer。

        Args:
            text: 训练文本
            max_vocab: 最大词汇表大小。如果字符种类超过此数，
                       按字符频率排序只保留前 max_vocab-1 个高频字符，
                       其余映射到 <unk>。None 则保留全部。
        """
        from collections import Counter
        freq = Counter(text)
        all_chars = sorted(freq.keys(), key=lambda c: (-freq[c], c))

        if max_vocab is not None and len(all_chars) >= max_vocab:
            # 保留高频字符，idx 0 是 <unk>
            kept = all_chars[:max_vocab - 1]
        else:
            kept = all_chars

        chars = ['<unk>'] + kept
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}
        return cls(chars, char_to_idx, idx_to_char)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # 未知字符映射到 0 (<unk>)
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "") for i in ids)


class CharDataset(Dataset):
    """字符级 next-char 预测数据集

    每个样本：(input_seq, target_seq) 长度均为 seq_len
    target[i] = input[i+1]
    """

    def __init__(self, text: str, seq_len: int = 64, tokenizer: CharTokenizer | None = None):
        self.seq_len = seq_len
        if tokenizer is None:
            self.tokenizer = CharTokenizer.from_text(text)
        else:
            self.tokenizer = tokenizer
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_chinese_corpus(data_dir: Path | None = None) -> str:
    """加载中文语料（唐诗宋词 + 论语片段，公共领域）。

    如果 data_dir 有 chinese.txt 则从文件加载，否则用内嵌样本。
    """
    if data_dir is not None:
        text_path = data_dir / "chinese.txt"
        if text_path.exists():
            return text_path.read_text(encoding="utf-8")

    return """学而时习之，不亦说乎？有朋自远方来，不亦乐乎？
人不知而不愠，不亦君子乎？
吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？
温故而知新，可以为师矣。
学而不思则罔，思而不学则殆。
知之为知之，不知为不知，是知也。
三人行，必有我师焉。择其善者而从之，其不善者而改之。
己所不欲，勿施于人。
巧言令色，鲜矣仁。
君子坦荡荡，小人长戚戚。
敏而好学，不耻下问，是以谓之文也。
知者乐水，仁者乐山。知者动，仁者静。知者乐，仁者寿。
质胜文则野，文胜质则史。文质彬彬，然后君子。
朝闻道，夕死可矣。
德不孤，必有邻。

床前明月光，疑是地上霜。
举头望明月，低头思故乡。

白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。

春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。

红豆生南国，春来发几枝。
愿君多采撷，此物最相思。

千山鸟飞绝，万径人踪灭。
孤舟蓑笠翁，独钓寒江雪。

离离原上草，一岁一枯荣。
野火烧不尽，春风吹又生。

锄禾日当午，汗滴禾下土。
谁知盘中餐，粒粒皆辛苦。

故人西辞黄鹤楼，烟花三月下扬州。
孤帆远影碧空尽，唯见长江天际流。

朝辞白帝彩云间，千里江陵一日还。
两岸猿声啼不住，轻舟已过万重山。

日照香炉生紫烟，遥看瀑布挂前川。
飞流直下三千尺，疑是银河落九天。

两个黄鹂鸣翠柳，一行白鹭上青天。
窗含西岭千秋雪，门泊东吴万里船。

月落乌啼霜满天，江枫渔火对愁眠。
姑苏城外寒山寺，夜半钟声到客船。

清明时节雨纷纷，路上行人欲断魂。
借问酒家何处有，牧童遥指杏花村。

向晚意不适，驱车登古原。
夕阳无限好，只是近黄昏。

大江东去，浪淘尽，千古风流人物。
故垒西边，人道是，三国周郎赤壁。
乱石穿空，惊涛拍岸，卷起千堆雪。
江山如画，一时多少豪杰。

明月几时有，把酒问青天。
不知天上宫阙，今夕是何年。
但愿人长久，千里共婵娟。

寻寻觅觅，冷冷清清，凄凄惨惨戚戚。
乍暖还寒时候，最难将息。

落霞与孤鹜齐飞，秋水共长天一色。
渔舟唱晚，响穷彭蠡之滨。

天行健，君子以自强不息。
地势坤，君子以厚德载物。

知足者富，强行者有志。
不失其所者久，死而不亡者寿。

上善若水。水善利万物而不争。
处众人之所恶，故几于道。

道可道，非常道。名可名，非常名。
无名天地之始，有名万物之母。

千里之行，始于足下。
祸兮福之所倚，福兮祸之所伏。

君子之交淡如水，小人之交甘若醴。
玉不琢，不成器；人不学，不知道。

路漫漫其修远兮，吾将上下而求索。
亦余心之所善兮，虽九死其犹未悔。

莫等闲，白了少年头，空悲切。
三十功名尘与土，八千里路云和月。

人生自古谁无死，留取丹心照汗青。
臣心一片磁针石，不指南方不肯休。

山重水复疑无路，柳暗花明又一村。
纸上得来终觉浅，绝知此事要躬行。

问渠那得清如许，为有源头活水来。
等闲识得东风面，万紫千红总是春。

海内存知己，天涯若比邻。
同是天涯沦落人，相逢何必曾相识。

会当凌绝顶，一览众山小。
读书破万卷，下笔如有神。

天生我材必有用，千金散尽还复来。
长风破浪会有时，直挂云帆济沧海。
"""


def load_shakespeare(data_dir: Path | None = None) -> str:
    """加载 Shakespeare 文本。

    如果 data_dir 不存在或没有文本，返回一个内嵌的小样本。
    """
    if data_dir is not None:
        text_path = data_dir / "shakespeare.txt"
        if text_path.exists():
            return text_path.read_text(encoding="utf-8")

    # 内嵌样本（足够小但能展现语言结构）
    return """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

O Romeo, Romeo! wherefore art thou Romeo?
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? It is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.

The course of true love never did run smooth;
But, either it was different in blood,
Or else misgraffed in respect of years.

If music be the food of love, play on,
Give me excess of it, that, surfeiting,
The appetite may sicken, and so die.
That strain again! it had a dying fall:
O, it came o'er my ear like the sweet sound
That breathes upon a bank of violets,
Stealing and giving odour.

All the world's a stage,
And all the men and women merely players:
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
"""


def load_textbook_corpus(corpus_dir: Path | None = None) -> str:
    """加载九年义务教育课本语料（从 corpus/*.txt 读取）。

    语料由 scripts/fetch_corpus_incremental.py 从维基百科 API 抓取，
    覆盖语文/数学/英语/化学/政治/历史/地理/物理/生物九个科目。
    维基百科返回繁体中文，这里统一转成简体。

    如果 corpus_dir 不存在或为空，返回空字符串。
    """
    if corpus_dir is None:
        corpus_dir = Path(__file__).parent.parent / 'corpus'

    if not corpus_dir.exists():
        return ''

    # 繁简转换
    try:
        from opencc import OpenCC
        cc = OpenCC('t2s')  # 繁体转简体
        convert = cc.convert
    except ImportError:
        convert = lambda x: x  # 无 opencc 则不转换

    parts = []
    for txt_file in sorted(corpus_dir.glob('*.txt')):
        content = txt_file.read_text(encoding='utf-8').strip()
        if content:
            parts.append(convert(content))
    return '\n\n'.join(parts)


def load_corpus(data_dir: Path | None = None,
                corpus_dir: Path | None = None) -> str:
    """加载完整训练语料（Shakespeare + 中文经典 + 九年义务教育课本）。

    用于构建包含中英文字符的 tokenizer 和训练。
    """
    parts = [
        load_shakespeare(data_dir),
        load_chinese_corpus(data_dir),
    ]
    textbook = load_textbook_corpus(corpus_dir)
    if textbook:
        parts.append(textbook)
    return '\n\n'.join(parts)
