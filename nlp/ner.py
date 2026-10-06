"""命名实体识别（NER）。

三路结合：
1. **字典匹配**：用内置实体词典（人名 / 地名 / 机构）做最长匹配；
2. **正则规则**：识别时间、日期、数字、金额、百分比；
3. **结构规则**：姓氏 + 名字组合识别人名，后缀识别机构 / 地名。

实体类别：PERSON 人名、LOCATION 地名、ORGANIZATION 机构、TIME 时间、
DATE 日期、NUMBER 数字、MONEY 金额、PERCENT 百分比。
"""

from __future__ import annotations

import re
from typing import Optional

from .lexicon import (LOCATIONS, ORGANIZATIONS, PERSONS, SURNAMES,
                      ORG_SUFFIXES, LOC_SUFFIXES, GIVEN_NAME_CHARS, load_dictionary)
from .segmenter import Segmenter


ENTITY_TYPE_NAMES = {
    "PERSON": "人名", "LOCATION": "地名", "ORGANIZATION": "机构",
    "TIME": "时间", "DATE": "日期", "NUMBER": "数字", "MONEY": "金额",
    "PERCENT": "百分比",
}


# 英文/中英混合组织名（大写英文词 + 可选机构后缀）
_EN_ORG_RE = re.compile(
    r"[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Za-z][A-Za-z0-9&.\-]*){0,3}"
    r"(?:公司|集团|银行|大学|学院|实验室|研究院|研究所)?")
# 常见英文机构后缀（出现大写词 + 这类后缀时判为机构）
_EN_ORG_SUFFIX_HINT = {"Inc", "Corp", "Ltd", "LLC", "Group", "Bank",
                       "Labs", "Lab", "Motors", "Company", "Co",
                       "SolarCity", "SpaceX", "YouTube", "OpenAI",
                       "DeepMind", "Meta"}
# 显式已知的英文组织名（无后缀时只认这些，避免句首大写词误报）
_EN_ORG_LEXICON = {"SolarCity", "SpaceX", "YouTube", "OpenAI", "DeepMind",
                   "Google", "Microsoft", "Amazon", "Apple", "Tesla",
                   "Meta", "Nvidia", "Intel", "IBM", "Oracle", "SAP",
                   "Samsung", "Sony", "Toyota", "Boeing", "Alibaba"}


# 正则规则：按优先级排列
_REGEX_RULES = [
    ("DATE", re.compile(r"\d{4}[年/\-]\d{1,2}[月/\-]\d{1,2}日?")),
    ("DATE", re.compile(r"\d{4}年(?:\d{1,2}月)?")),
    ("DATE", re.compile(r"\d{1,2}月\d{1,2}日")),
    ("TIME", re.compile(r"\d{1,2}[:：]\d{2}(?:[:：]\d{2})?")),
    ("MONEY", re.compile(r"\d+(?:\.\d+)?(?:万元|亿元|人民币|美元|港元|港币|欧元|日元|英镑|元)")),
    ("PERCENT", re.compile(r"百分之[零一二三四五六七八九十百]+|\d+(?:\.\d+)?%")),
    ("NUMBER", re.compile(r"\d+(?:\.\d+)?")),
]


class NERExtractor:
    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        self.dictionary = load_dictionary()
        # 合并实体词典：实体名 -> 类型
        self._entity_dict: dict[str, str] = {}
        for name in PERSONS:
            self._entity_dict[name] = "PERSON"
        for name in LOCATIONS:
            self._entity_dict[name] = "LOCATION"
        for name in ORGANIZATIONS:
            self._entity_dict[name] = "ORGANIZATION"
        self._max_entity_len = max((len(k) for k in self._entity_dict), default=6)

    # -- 对外接口 ---------------------------------------------------------
    def recognize(self, text: str) -> list[dict]:
        """返回实体列表，每个实体含 ``start/end/text/type``（字符偏移）。"""
        entities: list[dict] = []
        consumed: list[tuple[int, int]] = []

        # 1. 正则（时间/日期/数字/金额/百分比）
        for etype, pattern in _REGEX_RULES:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if self._overlaps(span, consumed):
                    continue
                consumed.append(span)
                entities.append({
                    "start": m.start(), "end": m.end(),
                    "text": m.group(), "type": etype,
                })

        # 2. 字典最长匹配（人名/地名/机构）
        dict_entities = self._dict_match(text)
        for ent in dict_entities:
            span = (ent["start"], ent["end"])
            if self._overlaps(span, consumed):
                continue
            consumed.append(span)
            entities.append(ent)

        # 3. 结构规则（姓氏+名 / 后缀）
        rule_entities = self._rule_match(text)
        for ent in rule_entities:
            span = (ent["start"], ent["end"])
            if self._overlaps(span, consumed):
                continue
            consumed.append(span)
            entities.append(ent)

        # 4. 英文 / 中英混合组织名（SolarCity、Alibaba 集团）
        for ent in self._english_orgs(text):
            span = (ent["start"], ent["end"])
            if self._overlaps(span, consumed):
                continue
            consumed.append(span)
            entities.append(ent)

        entities.sort(key=lambda e: e["start"])
        return entities

    # -- 英文组织 ---------------------------------------------------------
    def _english_orgs(self, text: str) -> list[dict]:
        results = []
        for m in _EN_ORG_RE.finditer(text):
            word = m.group().strip()
            if len(word) < 2:
                continue
            parts = re.split(r"[\s.]+", word)
            suffix_hit = any(p in _EN_ORG_SUFFIX_HINT for p in parts)
            cjk = any("一" <= c <= "鿿" for c in word)
            known = word in _EN_ORG_LEXICON or word in self._entity_dict
            # 纯英文必须有后缀线索或命中显式词典；中英混合要求带中文后缀
            if not (suffix_hit or known or
                    (cjk and any(word.endswith(s) for s in ORG_SUFFIXES))):
                continue
            # 紧跟中文机构后缀时把后缀一并纳入
            end = m.end()
            for suffix in sorted(ORG_SUFFIXES, key=len, reverse=True):
                if text.startswith(suffix, end):
                    end += len(suffix)
                    break
            results.append({"start": m.start(), "end": end,
                            "text": text[m.start():end],
                            "type": "ORGANIZATION"})
        return results

    # -- 字典匹配 ---------------------------------------------------------
    def _dict_match(self, text: str) -> list[dict]:
        results = []
        n = len(text)
        i = 0
        while i < n:
            if not ("一" <= text[i] <= "鿿"):
                i += 1
                continue
            matched = None
            for length in range(min(self._max_entity_len, n - i), 0, -1):
                word = text[i:i + length]
                if word in self._entity_dict:
                    matched = (word, self._entity_dict[word], length)
                    break
            if matched:
                word, etype, length = matched
                results.append({
                    "start": i, "end": i + length, "text": word, "type": etype,
                })
                i += length
            else:
                i += 1
        return results

    # -- 结构规则 ---------------------------------------------------------
    def _rule_match(self, text: str) -> list[dict]:
        results: list[dict] = []
        tokens = self._tokenize_with_offsets(text)
        # 常见称谓后缀，用于剔除「王先生」这类误报
        title_suffixes = ("先生", "女士", "小姐", "同志", "老师", "教授",
                          "博士", "经理", "局长", "主席", "书记")
        # 常见动作/职务词，防止把「任职」「董事」误判为人名
        non_person_words = {"任职", "就职", "供职", "同事", "工作", "出任",
                            "担任", "创业", "就业", "失业", "请假", "出差",
                            "董事", "股东", "法人", "代表", "委员", "秘书",
                            "主任", "处长", "科长", "部长", "总理", "经理",
                            "总裁", "总监", "主管", "工程师", "院士"}

        for word, start, end in tokens:
            if end - start < 2:
                continue
            if not all("一" <= c <= "鿿" for c in word):
                continue
            matched = False
            # 机构后缀
            for suffix in sorted(ORG_SUFFIXES, key=len, reverse=True):
                if word.endswith(suffix) and len(word) >= len(suffix) + 1:
                    results.append({"start": start, "end": end,
                                    "text": word, "type": "ORGANIZATION"})
                    matched = True
                    break
            if matched:
                continue
            # 地名后缀
            for suffix in sorted(LOC_SUFFIXES, key=len, reverse=True):
                if word.endswith(suffix) and len(word) >= len(suffix) + 1:
                    results.append({"start": start, "end": end,
                                    "text": word, "type": "LOCATION"})
                    matched = True
                    break
            if matched:
                continue
            # 人名：整词 2~3 字、以姓氏开头、且不是词典词（避免误报）
            if (2 <= len(word) <= 3 and word[0] in SURNAMES
                    and word not in self.dictionary
                    and word not in non_person_words
                    and not word.endswith(title_suffixes)
                    and (len(word) == 2 or word[1] in GIVEN_NAME_CHARS)):
                results.append({"start": start, "end": end,
                                "text": word, "type": "PERSON"})

        return results

    # -- 工具 -------------------------------------------------------------
    def _tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        words = self.segmenter.cut(text)
        tokens = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            if idx < 0:
                idx = pos
            tokens.append((word, idx, idx + len(word)))
            pos = idx + len(word)
        return tokens

    @staticmethod
    def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
        s, e = span
        for (a, b) in spans:
            if s < b and a < e:
                return True
        return False


def annotate(text: str, entities: list[dict]) -> str:
    """把实体标注回文本，用 XML 风格标签包裹（用于导出/展示）。"""
    parts = []
    last = 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        parts.append(text[last:ent["start"]])
        parts.append(f"<{ent['type']}>{text[ent['start']:ent['end']]}</{ent['type']}>")
        last = ent["end"]
    parts.append(text[last:])
    return "".join(parts)
