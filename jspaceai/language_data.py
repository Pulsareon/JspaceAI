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
    """字符级 tokenizer，支持训练时新字符的动态加入"""
    chars: list[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}
        return cls(chars, char_to_idx, idx_to_char)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # 未知字符用 0（假设第一个字符是常见的，或后续加 unk token）
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
