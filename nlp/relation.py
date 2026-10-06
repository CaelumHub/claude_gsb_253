"""实体关系联合抽取。

在 :mod:`nlp.ner` 已识别实体的基础上，**以触发词为必要条件**抽取实体之间
有方向的关系。核心原则：

* **宁缺毋滥**：两个实体在同一句中共现并不构成关系，只有在它们之间的
  文本里找到该关系的触发词（如「任职」「发布」「属于」）才产出一条边；
  抽取不到就不产出，绝不硬造。
* **有方向**：每条关系明确 ``head -> tail``，另存反向叙述名用于展示。
* **带溯源**：每条关系记录句内字符偏移与原句，前端可直接定位回原文。
* **开放宾语**：NER 只识别人名/地名/机构/时间等，对「产品」「品类」这类
  开放域名词，抽取器在触发词上下文里按需切出名词块充当节点
  （如「华为发布了 Mate60 手机」中的「Mate60 手机」）。
* **事件性状语**：时间/地点作为关系的 ``qualifiers`` 挂在边上，同时补出
  主体 -> 时间 的 ``occurred_at`` 边，支持「谁在何时做了什么事」。

输出的实体在 NER 原类别之外扩展两类：``PRODUCT`` 产品、``CATEGORY`` 品类。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .ner import NERExtractor
from .pos import POSTagger
from .segmenter import Segmenter
from .text import split_sentences


# ---------------------------------------------------------------------------
# 关系模式（schema）
# ---------------------------------------------------------------------------

#: 关系类型 -> （中文名，反向叙述名，主体类型，客体类型）
RELATION_SCHEMA = {
    "works_at":         ("任职于", "雇员", "PERSON", "ORGANIZATION"),
    "leads":            ("担任领导", "领导为", "PERSON", "ORGANIZATION"),
    "founded":          ("创立", "创立者", "PERSON", "ORGANIZATION"),
    "located_in":       ("位于", "所在地", "ORGANIZATION", "LOCATION"),
    "headquartered_in": ("总部位于", "总部所在地", "ORGANIZATION", "LOCATION"),
    "born_in":          ("出生于", "出生地", "PERSON", "LOCATION"),
    "launches":         ("发布/推出", "发布方", "ORGANIZATION", "PRODUCT"),
    "produces":         ("生产/制造", "生产商", "ORGANIZATION", "PRODUCT"),
    "belongs_to":       ("属于品类", "包含", "ENTITY", "CATEGORY"),
    "acquires":         ("收购", "被收购方", "ORGANIZATION", "ORGANIZATION"),
    "invests":          ("投资", "投资方", "ORGANIZATION", "ORGANIZATION"),
    "cooperates":       ("合作", "合作方", "ORGANIZATION", "ORGANIZATION"),
    "subsidiary_of":    ("子公司/品牌属于", "母公司", "ORGANIZATION",
                         "ORGANIZATION"),
    "occurred_at":      ("发生于", "事件主体", "ENTITY", "DATE"),
}

RELATION_NAMES = {k: v[0] for k, v in RELATION_SCHEMA.items()}

#: 扩展实体类型（叠加在 NER 的 ENTITY_TYPE_NAMES 之上）
ENTITY_TYPE_NAMES = {"PRODUCT": "产品", "CATEGORY": "品类"}

# ---------------------------------------------------------------------------
# 触发词词典
# ---------------------------------------------------------------------------

_WORKS_AT_TRIGGERS = (
    "工作于", "任职于", "就职于", "供职于", "受聘于",
    "加入", "出任", "就任", "担任", "任职", "就职", "供职", "受聘",
    "工作", "效力",
)
_LEADS_TRIGGERS = (
    "创始人", "联合创始人", "董事长", "首席执行官", "总裁", "总经理",
    "创办者", "掌舵",
)
_FOUNDED_TRIGGERS = ("创立", "创办", "成立", "创建", "组建")
_LOCATED_TRIGGERS = ("位于", "地处", "落户", "坐落于", "设在", "驻扎于", "扎根")
_HQ_TRIGGERS = ("总部位于", "总部设在", "总部坐落于", "总部位于", "总部在")
_BORN_TRIGGERS = ("出生于", "生于", "出生地")
_LAUNCH_TRIGGERS = (
    "发布", "推出", "发售", "上市", "正式发布", "召开发布会", "亮相",
)
_PRODUCE_TRIGGERS = ("生产", "制造", "研发", "开发", "出品", "打造", "量产")
_BELONG_TRIGGERS = ("属于", "隶属于", "归属于", "是一种", "是一类")
_ACQUIRE_TRIGGERS = ("收购", "并购", "买下")
_INVEST_TRIGGERS = ("投资", "注资", "入股")
_COOPERATE_TRIGGERS = ("合作", "联手", "携手", "达成合作", "战略合作")
_SUBSIDIARY_TRIGGERS = ("旗下", "子公司", "全资子公司", "控股", "母公司")

#: 模式表：(关系, 触发词组, 方向, 两实体间最大词距, 置信度, 触发位置)
#: direction = +1：靠前的实体是主体；-1：靠后的实体是主体（被动/倒装）。
#: zone = "between"：触发词在两实体之间；"after"：触发词在客体之后
#: （如「腾讯向美团投资」，此时要求两实体之间出现 向/对/给/和 等搭配标记）。
_PATTERNS = [
    ("works_at", _WORKS_AT_TRIGGERS, +1, 8, 0.9, "between"),
    ("leads", _LEADS_TRIGGERS, +1, 4, 0.9, "between"),
    ("leads", _LEADS_TRIGGERS, -1, 4, 0.7, "between"),
    ("founded", _FOUNDED_TRIGGERS, +1, 8, 0.9, "between"),
    ("founded", _FOUNDED_TRIGGERS, -1, 8, 0.7, "between"),
    ("headquartered_in", _HQ_TRIGGERS, +1, 6, 0.95, "between"),
    ("located_in", _LOCATED_TRIGGERS, +1, 8, 0.9, "between"),
    ("born_in", _BORN_TRIGGERS, +1, 6, 0.95, "between"),
    ("launches", _LAUNCH_TRIGGERS, +1, 10, 0.9, "between"),
    ("produces", _PRODUCE_TRIGGERS, +1, 10, 0.85, "between"),
    ("acquires", _ACQUIRE_TRIGGERS, +1, 10, 0.9, "between"),
    ("acquires", ("收购", "并购"), -1, 10, 0.75, "between"),
    ("invests", _INVEST_TRIGGERS, +1, 10, 0.85, "between"),
    ("invests", ("投资", "注资", "入股"), -1, 10, 0.65, "between"),
    ("cooperates", _COOPERATE_TRIGGERS, +1, 10, 0.8, "between"),
    ("cooperates", _COOPERATE_TRIGGERS, -1, 10, 0.7, "between"),
    # 「X 向/对/给 Y 投资」「X 和 Y 合作」：触发词在客体之后
    ("invests", ("投资", "注资", "入股"), +1, 8, 0.85, "after"),
    ("cooperates", ("合作", "联手", "携手"), +1, 8, 0.8, "after"),
    # 「X 在 Y 任职/工作」
    ("works_at", ("任职", "工作", "就职", "供职"), +1, 8, 0.85, "after"),
]

#: zone="after" 时，两实体之间必须出现的搭配标记
_AFTER_MARKERS = {"invests": ("向", "对", "给", "在"),
                  "cooperates": ("和", "与", "跟", "同"),
                  "works_at": ("在",)}
#: 反向（direction=-1）规则要求两实体之间出现被动标记
_PASSIVE_MARKERS = ("被", "由", "受", "获")

# ---------------------------------------------------------------------------
# 开放域名词块（产品 / 品类）资源
# ---------------------------------------------------------------------------

PRODUCT_HEADS = (
    "手机", "电脑", "笔记本", "平板", "手表", "耳机", "芯片", "处理器",
    "显卡", "汽车", "轿车", "电动车", "车型", "系统", "操作系统", "软件",
    "应用", "APP", "app", "模型", "机器人", "设备", "装置",
    "相机", "电视", "空调", "冰箱", "机型", "药物", "疫苗", "饮料",
)
#: 无明确中心语时的兜底产品词
_GENERIC_PRODUCT_HEADS = ("产品", "新品", "新款", "品牌")
CATEGORY_HEADS = (
    "品类", "类别", "种类", "类目", "门类", "分类", "行业", "领域",
    "赛道", "市场",
)
_BRAND_TERMS = (
    "华为", "苹果", "小米", "特斯拉", "比亚迪", "三星", "谷歌", "微软",
    "亚马逊", "英伟达", "腾讯", "阿里巴巴", "阿里", "百度", "字节跳动",
    "京东", "美团", "OPPO", "vivo", "荣耀", "理想", "蔚来", "小鹏",
    "茅台",
)
_DETERMINERS = {"全新", "新款", "新一代", "首款", "旗舰", "高端", "智能",
                "电动", "新能源", "新", "整个", "整个"}

_NOUN_TAGS = {"n", "ns", "nr", "nt"}
_ALNUM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")

#: 小句切分（关系只在同一小句内配对，不跨逗号乱连）
_CLAUSE_SPLIT_RE = re.compile(r"[，,；;。！？!?]")
#: 名词块切分时跳过的功能词
_SKIP_CHUNK_WORDS = {"的", "了", "之"}


@dataclass
class _Token:
    word: str
    tag: str
    start: int
    end: int


@dataclass
class RelationMention:
    """一条关系提及（一次出现）。"""
    head: dict
    tail: dict
    relation: str
    relation_name: str
    action: str = ""        # 触发词原文（边上「做了什么」）
    confidence: float = 0.0
    qualifiers: list[dict] = field(default_factory=list)
    sentence: str = ""
    sent_start: int = 0
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict:
        return {
            "head": self.head, "tail": self.tail,
            "relation": self.relation, "relation_name": self.relation_name,
            "action": self.action, "confidence": round(self.confidence, 3),
            "qualifiers": self.qualifiers,
            "sentence": self.sentence, "sent_start": self.sent_start,
            "start": self.start, "end": self.end,
        }


class RelationExtractor:
    """实体关系联合抽取器。"""

    def __init__(self, ner: Optional[NERExtractor] = None,
                 tagger: Optional[POSTagger] = None,
                 segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        self.ner = ner or NERExtractor(self.segmenter)
        self.tagger = tagger or POSTagger(self.segmenter)

    # -- 对外接口 ---------------------------------------------------------
    def extract(self, text: str) -> dict:
        """从一段文本同时抽取实体与关系。

        返回::

            {
              "entities":  [ {text,type,start,end,occurrences}, ... ],
              "relations": [ RelationMention.to_dict(), ... ],
              "sentences": [ {text,start,end}, ... ],
            }
        """
        sentences = self._split_sentences_with_offsets(text)
        all_entities: list[dict] = []
        all_relations: list[RelationMention] = []

        for sent in sentences:
            sent_text, s_off = sent["text"], sent["start"]
            tokens = self._tokenize(sent_text, s_off)
            base_entities = self._recognize_entities(sent_text, s_off)
            product_spans = self._find_noun_spans(
                tokens, PRODUCT_HEADS + _GENERIC_PRODUCT_HEADS, "PRODUCT")
            category_spans = self._find_noun_spans(
                tokens, CATEGORY_HEADS, "CATEGORY")

            relations, spawned = self._extract_open_domain(
                sent_text, s_off, tokens, base_entities,
                product_spans, category_spans)
            # 被产品/品类块覆盖的数字等基础实体剔除（如 Mate60 里的「60」）
            cover = product_spans + category_spans
            entities = [e for e in base_entities
                        if not self._overlaps_any(e, cover)] + spawned
            relations += self._extract_entity_pairs(
                sent_text, s_off, tokens, entities)
            relations = self._deduplicate(relations)
            relations = self._attach_qualifiers(relations, entities)

            all_entities.extend(entities)
            all_relations.extend(relations)

        return {
            "entities": self._merge_entity_mentions(all_entities),
            "relations": [r.to_dict() for r in all_relations],
            "sentences": sentences,
            "relation_names": RELATION_NAMES,
            "entity_type_names": ENTITY_TYPE_NAMES,
        }

    # -- 句子 / 分词 / 实体 -----------------------------------------------
    @staticmethod
    def _split_sentences_with_offsets(text: str) -> list[dict]:
        """句子切分并给出字符偏移（顺序 find 定位，避免重复子句错位）。"""
        result = []
        cursor = 0
        for s in split_sentences(text):
            idx = text.find(s, cursor)
            if idx < 0:
                idx = cursor
            result.append({"text": s, "start": idx, "end": idx + len(s)})
            cursor = idx + len(s)
        return result

    def _tokenize(self, sent_text: str, offset: int) -> list[_Token]:
        words = self.segmenter.cut(sent_text)
        tokens: list[_Token] = []
        pos = 0
        for word, tag in self.tagger.tag(words):
            idx = sent_text.find(word, pos)
            if idx < 0:
                idx = pos
            tokens.append(_Token(word, tag, offset + idx,
                                 offset + idx + len(word)))
            pos = idx + len(word)
        return tokens

    def _recognize_entities(self, sent_text: str, offset: int) -> list[dict]:
        ents = self.ner.recognize(sent_text)
        for e in ents:
            e["start"] += offset
            e["end"] += offset
        return ents

    @staticmethod
    def _merge_entity_mentions(entities: list[dict]) -> list[dict]:
        """同一 (text,type) 的多次提及合并，偏移保留全部出现位置。"""
        merged: dict[tuple, dict] = {}
        order = []
        for e in entities:
            key = (e["text"], e["type"])
            if key not in merged:
                merged[key] = {
                    "text": e["text"], "type": e["type"],
                    "start": e["start"], "end": e["end"],
                    "occurrences": [[e["start"], e["end"]]],
                    "generic": bool(e.get("generic")),
                }
                order.append(key)
            else:
                merged[key]["occurrences"].append([e["start"], e["end"]])
                merged[key]["generic"] = (
                    merged[key]["generic"] and bool(e.get("generic")))
        return [merged[k] for k in sorted(order, key=lambda k: merged[k]["start"])]

    # -- 名词块 -----------------------------------------------------------
    def _find_noun_spans(self, tokens: list[_Token], heads: tuple[str, ...],
                         etype: str) -> list[dict]:
        """找以 ``heads`` 任一为中心语的名词块（中心语 + 左侧名词/修饰）。"""
        spans: list[dict] = []
        for i, tok in enumerate(tokens):
            if not any(tok.word == h or tok.word.endswith(h) for h in heads):
                continue
            start_i = i
            j = i - 1
            while j >= 0 and i - j <= 5:
                t = tokens[j]
                if t.tag in _NOUN_TAGS or t.word in _DETERMINERS \
                        or _ALNUM_RE.fullmatch(t.word) \
                        or t.word in _BRAND_TERMS:
                    start_i = j
                    j -= 1
                elif t.word in _SKIP_CHUNK_WORDS:
                    j -= 1  # 「Mate60 的 手机」允许跨过「的」
                else:
                    break
            if start_i > 0 and tokens[start_i - 1].word in _BRAND_TERMS:
                start_i -= 1
            text = "".join(t.word for t in tokens[start_i:i + 1])
            if len(text) >= 2:
                spans.append({
                    "text": text, "type": etype,
                    "start": tokens[start_i].start, "end": tokens[i].end,
                    "generic": True,
                })
        # 去重叠，保留较长块
        spans.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        unique: list[dict] = []
        for s in spans:
            if unique and s["start"] < unique[-1]["end"]:
                if s["end"] - s["start"] > unique[-1]["end"] - unique[-1]["start"]:
                    unique[-1] = s
                continue
            unique.append(s)
        return unique

    @staticmethod
    def _clause_bounds(sent_text: str, offset: int) -> list[tuple[int, int]]:
        """按逗号/分号切小句，返回全局字符区间。"""
        spans = []
        start = 0
        for m in _CLAUSE_SPLIT_RE.finditer(sent_text):
            if m.start() > start:
                spans.append((offset + start, offset + m.start()))
            start = m.end()
        if start < len(sent_text):
            spans.append((offset + start, offset + len(sent_text)))
        return spans or [(offset, offset + len(sent_text))]

    # -- 实体对 + 触发词模式 ----------------------------------------------
    def _extract_entity_pairs(self, sent_text, offset, tokens, entities):
        relations: list[RelationMention] = []
        seen: set = set()
        for c_start, c_end in self._clause_bounds(sent_text, offset):
            members = sorted(
                (e for e in entities
                 if e["start"] >= c_start and e["end"] <= c_end),
                key=lambda e: e["start"])
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    earlier, later = members[x], members[y]
                    gap = self._token_distance(tokens, earlier, later)
                    # 两实体之间的「字符级原文」，弱分词也能匹配触发词
                    between = sent_text[earlier["end"] - offset:
                                        later["start"] - offset]
                    for (rel, triggers, direction, window, conf,
                         zone) in _PATTERNS:
                        if gap > window:
                            continue
                        if zone == "after":
                            # 「X 向 Y 投资」：触发词紧挨客体之后，
                            # 两实体之间必须有 向/对/给/和 等搭配标记
                            markers = _AFTER_MARKERS.get(rel, ())
                            if not any(mk in between for mk in markers):
                                continue
                            tail_text = self._after_text(
                                sent_text, offset, later, length=6)
                            trigger = self._match_trigger(tail_text, triggers)
                        else:
                            trigger = self._match_trigger(between, triggers)
                        if not trigger:
                            continue
                        # 总部触发词与普通「位于」同时命中时，只保留更具体的
                        if (rel == "located_in"
                                and any(h in between for h in _HQ_TRIGGERS)):
                            continue
                        head, tail = ((earlier, later) if direction > 0
                                      else (later, earlier))
                        # 反向（被动/倒装）规则必须带 被/由/受 等标记
                        if direction < 0 and not any(
                                mk in between for mk in _PASSIVE_MARKERS):
                            continue
                        if not self._type_ok(rel, head, tail):
                            continue
                        key = (rel, head["start"], tail["start"], zone)
                        if key in seen:
                            continue
                        seen.add(key)
                        relations.append(self._make_relation(
                            head, tail, rel, trigger, conf,
                            sent_text, offset))
        return relations

    @staticmethod
    def _type_ok(rel: str, head: dict, tail: dict) -> bool:
        _, _, want_h, want_t = RELATION_SCHEMA[rel]
        head_ok = head["type"] in ("PERSON", "ORGANIZATION", "LOCATION") \
            if want_h == "ENTITY" else head["type"] == want_h
        tail_ok = tail["type"] in ("DATE", "TIME") \
            if want_t == "DATE" else (
                tail["type"] in ("PRODUCT", "CATEGORY", "PERSON",
                                 "ORGANIZATION", "LOCATION")
                if want_t == "ENTITY" else tail["type"] == want_t)
        return head_ok and tail_ok

    @staticmethod
    def _after_text(sent_text: str, offset: int, ent: dict,
                    length: int = 6) -> str:
        start = ent["end"] - offset
        return sent_text[start:start + length]

    @staticmethod
    def _token_distance(tokens: list[_Token], a: dict, b: dict) -> int:
        ia = next((i for i, t in enumerate(tokens) if t.start == a["start"]), None)
        ib = next((i for i, t in enumerate(tokens) if t.start == b["start"]), None)
        if ia is None or ib is None:
            return 99
        return abs(ia - ib)

    @staticmethod
    def _match_trigger(text: str, triggers: tuple[str, ...]) -> str:
        """两实体之间的文本里是否包含任一触发词，长词优先。"""
        for trig in sorted(triggers, key=len, reverse=True):
            if trig in text:
                return trig
        return ""

    def _make_relation(self, head, tail, rel, trigger, conf,
                       sent_text, offset) -> RelationMention:
        return RelationMention(
            head=self._public_entity(head),
            tail=self._public_entity(tail),
            relation=rel,
            relation_name=RELATION_SCHEMA[rel][0],
            action=trigger, confidence=conf,
            sentence=sent_text, sent_start=offset,
            start=min(head["start"], tail["start"]),
            end=max(head["end"], tail["end"]))

    # -- 开放域：产品 / 品类 / 隶属 ---------------------------------------
    def _extract_open_domain(self, sent_text, offset, tokens, base_entities,
                             product_spans, category_spans):
        relations: list[RelationMention] = []
        used_spans: list[dict] = []
        clauses = self._clause_bounds(sent_text, offset)

        def clause_of(pos: int):
            for c_start, c_end in clauses:
                if c_start <= pos < c_end:
                    return c_start, c_end
            return offset, offset + len(sent_text)

        def in_clause(ent, pos):
            c_start, c_end = clause_of(pos)
            return ent and c_start <= ent["start"] and ent["end"] <= c_end

        def emit(head, tail, rel, trigger, conf):
            relations.append(RelationMention(
                head=self._public_entity(head),
                tail=self._public_entity(tail),
                relation=rel,
                relation_name=RELATION_SCHEMA[rel][0],
                action=trigger, confidence=conf,
                sentence=sent_text, sent_start=offset,
                start=min(head["start"], tail["start"]),
                end=max(head["end"], tail["end"])))
            for e in (head, tail):
                if e.get("generic") and not self._overlaps_any(
                        e, base_entities) and not any(
                        e["start"] == u["start"] for u in used_spans):
                    used_spans.append(e)

        # 1) 产品 -> 品类：「X 属于 / 是一种 Y」
        for trig in sorted(_BELONG_TRIGGERS, key=len, reverse=True):
            for i, tok in enumerate(tokens):
                if trig not in tok.word:
                    continue
                c_start, c_end = clause_of(tok.start)
                # 优先取触发词所在小句内的产品；其次小句紧邻前文
                head = (self._nearest_entity(base_entities, tok.start, "left",
                                             ("PRODUCT", "ORGANIZATION",
                                              "PERSON", "LOCATION"),
                                             bounds=(c_start, c_end))
                        or self._nearest_span(product_spans, tok.start, "left",
                                              bounds=(c_start, c_end))
                        or self._chunk_near(tokens, i, "left", "PRODUCT"))
                tail = (self._nearest_span(category_spans, tok.start, "right",
                                           bounds=(c_start, c_end))
                        or self._chunk_near(tokens, i, "right", "CATEGORY"))
                # 只在同一小句内配对，避免跨逗号张冠李戴
                if (head and tail and head["start"] != tail["start"]
                        and in_clause(head, tok.start)
                        and in_clause(tail, tok.start)):
                    emit(head, tail, "belongs_to", trig, 0.8)

        # 2) 机构 -> 产品：「X 发布/推出/生产 Y」
        for triggers, rel in ((_LAUNCH_TRIGGERS, "launches"),
                              (_PRODUCE_TRIGGERS, "produces")):
            for trig in sorted(triggers, key=len, reverse=True):
                for i, tok in enumerate(tokens):
                    if trig not in tok.word:
                        continue
                    c_start, c_end = clause_of(tok.start)
                    org = self._nearest_entity(base_entities, tok.start,
                                               "left", ("ORGANIZATION",))
                    if org and in_clause(org, tok.start):
                        conf = 0.85
                    elif org:
                        # 主语承前省略（同句紧邻小句回溯），降权处理
                        conf = 0.7
                    else:
                        continue
                    prod = (self._nearest_span(product_spans, tok.start,
                                               "right",
                                               bounds=(c_start, c_end))
                            or self._chunk_near(tokens, i, "right", "PRODUCT"))
                    if (prod and in_clause(prod, tok.start)
                            and self._token_distance_by_index(
                            tokens, org["start"], prod["start"], 14)):
                        emit(org, prod, rel, trig, conf)

        # 3) 隶属：「X 旗下 Y」「Y 的母公司是 X」
        for i, tok in enumerate(tokens):
            if tok.word == "旗下":
                parent = self._nearest_entity(base_entities, tok.start,
                                              "left", ("ORGANIZATION",))
                child = (self._nearest_entity(base_entities, tok.end, "right",
                                              ("ORGANIZATION", "PRODUCT"))
                         or self._nearest_span(product_spans, tok.end, "right")
                         or self._chunk_near(tokens, i, "right", "ORGANIZATION"))
                if (parent and child and child is not parent
                        and in_clause(parent, tok.start)
                        and in_clause(child, tok.start)):
                    emit(child, parent, "subsidiary_of", "旗下", 0.8)
            elif tok.word in ("子公司", "全资子公司"):
                parent = self._nearest_entity(base_entities, tok.end, "right",
                                              ("ORGANIZATION",))
                child = self._nearest_entity(base_entities, tok.start, "left",
                                             ("ORGANIZATION",))
                if (parent and child and parent is not child
                        and in_clause(parent, tok.start)
                        and in_clause(child, tok.start)):
                    emit(child, parent, "subsidiary_of", tok.word, 0.8)
            elif tok.word == "母公司":
                child = self._nearest_entity(base_entities, tok.start, "left",
                                             ("ORGANIZATION",))
                parent = self._nearest_entity(base_entities, tok.end, "right",
                                              ("ORGANIZATION",))
                if (parent and child and parent is not child
                        and in_clause(parent, tok.start)
                        and in_clause(child, tok.start)):
                    emit(child, parent, "subsidiary_of", "母公司", 0.8)

        return relations, used_spans

    # -- 邻近查询 / 兜底名词块 --------------------------------------------
    @staticmethod
    def _nearest_entity(entities, pos, side, types, bounds=None):
        cand = [e for e in entities if e["type"] in types]
        return RelationExtractor._nearest_span(cand, pos, side, bounds)

    @staticmethod
    def _nearest_span(spans, pos, side, bounds=None):
        if side == "left":
            cand = [s for s in spans if s["end"] <= pos]
            if bounds:
                cand = [s for s in cand if s["start"] >= bounds[0]]
            return cand[-1] if cand else None
        cand = [s for s in spans if s["start"] >= pos]
        if bounds:
            cand = [s for s in cand if s["end"] <= bounds[1]]
        return cand[0] if cand else None

    def _chunk_near(self, tokens: list[_Token], tok_i: int,
                    side: str, etype: str):
        """触发词紧邻位置上的兜底名词块（要求至少一个名词性 token）。"""
        heads = (PRODUCT_HEADS + _GENERIC_PRODUCT_HEADS if etype == "PRODUCT"
                 else CATEGORY_HEADS)
        step = 1 if side == "right" else -1
        j = tok_i + step
        idxs: list[int] = []
        noun_seen = False
        while 0 <= j < len(tokens) and len(idxs) < 4:
            t = tokens[j]
            if t.word in _SKIP_CHUNK_WORDS:
                j += step
                continue
            is_alnum = bool(_ALNUM_RE.fullmatch(t.word))
            is_noun = t.tag in _NOUN_TAGS or is_alnum
            is_head = any(t.word == h or t.word.endswith(h) for h in heads)
            # 左向扩展时，紧邻触发词的裸字母数字名（如 Mate60）也接受
            if is_noun or t.word in _DETERMINERS or is_head:
                idxs.append(j)
                noun_seen = noun_seen or is_noun or is_head
                j += step
            else:
                break
        if not noun_seen:
            return None
        idxs.sort()
        text = "".join(tokens[k].word for k in idxs)
        if len(text) < 2:
            return None
        return {"text": text, "type": etype,
                "start": tokens[idxs[0]].start, "end": tokens[idxs[-1]].end,
                "generic": True}

    @staticmethod
    def _token_distance_by_index(tokens, a_start, b_start, limit) -> bool:
        ia = next((i for i, t in enumerate(tokens) if t.start == a_start), None)
        ib = next((i for i, t in enumerate(tokens) if t.start == b_start), None)
        if ia is None or ib is None:
            return abs(a_start - b_start) <= 40
        return abs(ia - ib) <= limit

    @staticmethod
    def _overlaps_any(span, entities) -> bool:
        return any(span["start"] < e["end"] and e["start"] < span["end"]
                   for e in entities)

    # -- 后处理：去重、时间地点限定、occurred_at ---------------------------
    @staticmethod
    def _deduplicate(relations: list[RelationMention]) -> list[RelationMention]:
        """同句同 (关系, head 偏移, tail 偏移) 只保留置信度最高的一条。"""
        best: dict = {}
        for r in relations:
            key = (r.sent_start, r.relation,
                   r.head["start"], r.tail["start"])
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r
        return sorted(best.values(), key=lambda r: (r.sent_start, r.start))

    def _attach_qualifiers(self, relations, entities) -> list[RelationMention]:
        """把同小句中的时间/地点实体挂到事件关系上，并补 occurred_at 边。"""
        event_rels = {"works_at", "founded", "located_in",
                      "headquartered_in", "launches", "produces",
                      "acquires", "invests", "cooperates", "born_in",
                      "subsidiary_of"}
        extra: list[RelationMention] = []
        for r in relations:
            # 限定语取关系所在小句
            rel_clause = self._relation_clause(r)
            for e in entities:
                if e["type"] not in ("DATE", "TIME", "LOCATION"):
                    continue
                if not (rel_clause[0] <= e["start"] < rel_clause[1]):
                    continue
                if e["start"] == r.head["start"] or e["start"] == r.tail["start"]:
                    continue
                r.qualifiers.append({
                    "text": e["text"], "type": e["type"],
                    "start": e["start"], "end": e["end"]})
            if r.relation in event_rels:
                for q in r.qualifiers:
                    if q["type"] not in ("DATE", "TIME"):
                        continue
                    extra.append(RelationMention(
                        head=dict(r.head),
                        tail={"text": q["text"], "type": q["type"],
                              "start": q["start"], "end": q["end"]},
                        relation="occurred_at",
                        relation_name=RELATION_SCHEMA["occurred_at"][0],
                        action=r.action,
                        confidence=min(0.8, r.confidence - 0.1),
                        sentence=r.sentence, sent_start=r.sent_start,
                        start=min(r.start, q["start"]),
                        end=max(r.end, q["end"])))
        # occurred_at 同 (head,tail) 去重，保留置信度最高且 action 最具事件性的
        best: dict = {}
        for r in extra:
            key = (r.sent_start, r.head["start"], r.tail["start"])
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r
        relations.extend(best.values())
        return relations

    def _relation_clause(self, r: RelationMention) -> tuple[int, int]:
        """定位关系所在小句：找包含触发词、且覆盖 head/tail 区间的小句。"""
        sent_text = r.sentence
        offset = r.sent_start
        clauses = self._clause_bounds(sent_text, offset)

        def clause_index(pos):
            for k, (c_start, c_end) in enumerate(clauses):
                if c_start <= pos < c_end:
                    return k
            return len(clauses) - 1

        lo = min(r.head["start"], r.tail["start"])
        hi = max(r.head["end"], r.tail["end"])
        # 在 head/tail 覆盖的小句区间里，找文字中真正含触发词的那个
        for k in range(clause_index(lo), clause_index(hi - 1) + 1):
            c_start, c_end = clauses[k]
            window = sent_text[c_start - offset:c_end - offset]
            if r.action and r.action in window:
                return c_start, c_end
        # 触发词在客体之后（「向美团投资」）：用客体所在小句
        return clauses[clause_index(r.tail["start"])]

    @staticmethod
    def _public_entity(e: dict) -> dict:
        return {
            "text": e["text"], "type": e["type"],
            "start": e["start"], "end": e["end"],
            "generic": bool(e.get("generic")),
        }


def extract_relations(text: str) -> dict:
    """便捷函数：一次性抽取实体 + 关系。"""
    return RelationExtractor().extract(text)
