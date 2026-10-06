"""实体关系抽取与三元组证据管理。

该模块坚持「有证据才输出」：所有关系都必须来自同一个句子，并保留方向、
谓词、原文句子、字符偏移和匹配规则。跨文档节点对齐由
:class:`storage.knowledge_graph.KnowledgeGraphStore` 负责，抽取层只提供
保守的规范化，不做高风险的模糊合并。

支持的高频关系：
- PERSON --WORKS_AT/FOUNDED/SERVED_AS--> ORGANIZATION / TITLE
- ORGANIZATION --LOCATED_IN--> LOCATION
- ORGANIZATION --PRODUCES/LAUNCHES--> PRODUCT
- PRODUCT --BELONGS_TO--> CATEGORY
- PRODUCT --PRICED_AT--> MONEY
- PERSON/ORGANIZATION --PERFORMED--> EVENT --OCCURRED_AT--> DATE
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from .ner import NERExtractor, ENTITY_TYPE_NAMES

# 额外关系标签（实体类型标签仍放在 ner.py）
RELATION_NAMES = {
    "WORKS_AT": "任职于",
    "FOUNDED": "创办",
    "SERVED_AS": "担任",
    "LOCATED_IN": "位于",
    "PRODUCES": "生产",
    "LAUNCHES": "发布",
    "BELONGS_TO": "属于品类",
    "PRICED_AT": "价格为",
    "PERFORMED": "参与事件",
    "OCCURRED_AT": "发生于",
}

# 关系允许的主体/客体类型，用于防止张冠李戴
RELATION_DOMAINS = {
    "WORKS_AT": (("PERSON",), ("ORGANIZATION",)),
    "FOUNDED": (("PERSON",), ("ORGANIZATION",)),
    "SERVED_AS": (("PERSON",), ("TITLE",)),
    "LOCATED_IN": (("ORGANIZATION",), ("LOCATION",)),
    "PRODUCES": (("ORGANIZATION",), ("PRODUCT",)),
    "LAUNCHES": (("ORGANIZATION", "PERSON"), ("PRODUCT",)),
    "BELONGS_TO": (("PRODUCT",), ("CATEGORY",)),
    "PRICED_AT": (("PRODUCT", "EVENT"), ("MONEY",)),
    "PERFORMED": (("PERSON", "ORGANIZATION"), ("EVENT",)),
    "OCCURRED_AT": (("EVENT",), ("DATE", "TIME")),
}

_WORDS_CUES = ("任职于", "就职于", "供职于", "加入", "入职", "供职", "工作于",
               "在", "受雇于")
_FOUND_CUES = ("创办", "创立", "创建", "成立", "联合创立", "联合创办")
_TITLE_CUES = ("担任", "出任", "任职", "作为", "职位是")
_BELONGS_CUES = ("属于", "隶属", "归属于", "是一种", "是一款", "是一个")
_LAUNCH_CUES = ("发布", "发布了", "推出", "推出了", "发售")
_LOCATION_CUES = ("位于", "总部位于", "坐落于", "在", "成立于")

_CATEGORY_HEADS = ("品类", "类别", "分类", "产品类目")
_CATEGORY_TERMS = ("手机", "电脑", "汽车", "电动汽车", "新能源汽车", "平板",
                   "手表", "耳机", "软件", "系统", "芯片", "服务器", "机器人",
                   "冰箱", "电视", "相机", "无人机", "音箱")
_TITLE_SUFFIXES = ("工程师", "经理", "总监", "总裁", "主任", "教授", "老师",
                   "分析师", "设计师", "研究员", "科学家", "顾问", "主管",
                   "董事长", "首席执行官", "CEO", "CTO", "CFO", "COO")

# 已知的英文/数字型号；未知型号只在紧邻品类词等强模式下识别，避免乱抽。
_KNOWN_PRODUCTS = {"iPhone", "iPad", "MacBook", "Mate", "Pura", "Model",
                   "Mi", "Redmi", "Galaxy", "Pixel", "Windows", "Office",
                   "ChatGPT"}
_PRODUCT_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"[A-Z][A-Za-z]{1,15}(?:\s+[A-Z0-9][A-Za-z0-9+.\-]*){0,3}"
    r"|(?:iPhone|iPad|Mate|Pura|Model|Mi|Redmi|Galaxy|Pixel)\s*"
    r"[A-Z0-9][A-Za-z0-9+.\- ]{0,20}"
    r")(?=\s*(?:手机|电话|电脑|汽车|平板|手表|耳机|芯片|服务器|机器人|"
    r"电视|相机|无人机|音箱|软件|系统|$|[，。,.！!？?；;]))"
)
_CLAUSE_SPLIT_RE = re.compile(r"[，,；;。！？!?\n]+")


class RelationExtractor:
    """基于实体 + 强触发模式的高精度关系抽取器。"""

    def __init__(self, ner: Optional[NERExtractor] = None):
        self.ner = ner or NERExtractor()

    # -- 对外接口 ---------------------------------------------------------
    def extract(self, text: str, known_entities: Optional[list[dict]] = None) -> dict:
        """从一段文本抽取实体、关系和逐句证据。

        ``known_entities`` 用于注入跨文档人工确认的别名实体，例如把“阿里”
        预先标注为 ORGANIZATION，使后续关系规则能够在本句中识别它。
        """
        entities = self.ner.recognize(text)
        for ent in known_entities or []:
            if not self._overlaps_entity(ent, entities):
                entities.append(ent)
        entities = self._add_domain_entities(text, entities)
        # 人工别名优先于通用 NER 的重叠结果，避免同一字符片段出现两个类型。
        entities = self._prefer_manual_entities(entities, known_entities or [])
        sentences = self._sentences_with_offsets(text)

        relations: list[dict] = []
        for sent, s_start, s_end in sentences:
            sent_entities = [e for e in entities
                             if e["start"] >= s_start and e["end"] <= s_end]
            relations.extend(self._extract_sentence(
                sent, sent_entities, s_start, s_end))

        relations = self._deduplicate(relations)
        return {
            "entities": sorted(entities, key=lambda e: e["start"]),
            "relations": relations,
            "entity_type_names": ENTITY_TYPE_NAMES,
            "relation_names": RELATION_NAMES,
        }

    # -- 句子和实体补充 ---------------------------------------------------
    @staticmethod
    def _prefer_manual_entities(entities: list[dict],
                                known: list[dict]) -> list[dict]:
        manual = [e for e in known if e.get("rule") == "manual_alias"]
        result = [e for e in entities
                  if not any(e.get("rule") != "manual_alias"
                             and e["start"] < m["end"]
                             and m["start"] < e["end"] for m in manual)]
        result.extend(manual)
        return sorted({(e["start"], e["end"]): e for e in result}.values(),
                      key=lambda e: e["start"])

    def _sentences_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        result = []
        # split_sentences 会吞掉标点，这里保留可回溯的字符偏移。
        for match in re.finditer(r".+?(?:[。！？!?；;\n]+|$)", text, re.S):
            raw = match.group()
            sentence = raw.strip(" 　\t\r\n")
            leading = len(raw) - len(raw.lstrip(" 　\t\r\n"))
            if not sentence:
                continue
            offset = text.index(sentence, match.start() + leading, match.end())
            result.append((sentence, offset, offset + len(sentence)))
        return result or [(text, 0, len(text))]

    def _add_domain_entities(self, text: str, entities: list[dict]) -> list[dict]:
        entities = list(entities)
        products = self._find_products(text, entities)
        # 型号中的数字应解释为产品名的一部分，而不是独立 NUMBER。
        entities = [e for e in entities
                    if not (e["type"] in ("NUMBER",)
                            and self._overlaps_entity(e, products))]
        for ent in products:
            if not self._overlaps_entity(ent, entities):
                entities.append(ent)
        for ent in self._find_categories(text, entities):
            if not self._overlaps_entity(ent, entities):
                entities.append(ent)
        for ent in self._find_titles(text):
            if not self._overlaps_entity(ent, entities):
                entities.append(ent)
        return sorted(entities, key=lambda e: e["start"])

    def _find_products(self, text: str, entities: list[dict]) -> list[dict]:
        results: list[dict] = []

        for m in _PRODUCT_RE.finditer(text):
            name = m.group(1).strip()
            start = m.start(1) + (m.group(1).find(name) if name else 0)
            if self._valid_product(name, text[m.end():m.end() + 8]):
                results.append({"start": start, "end": start + len(name),
                                "text": name, "type": "PRODUCT",
                                "rule": "latin_model"})

        # “华为发布了 Mate 60 Pro 手机”等中英混合表达，拉丁型号本身可能无
        # 空格边界；用已知型号前缀 + 紧邻品类词做强约束。
        for known in _KNOWN_PRODUCTS:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(known)}\s*"
                r"[A-Z0-9][A-Za-z0-9+.\- ]{{0,20}}?"
                r"(?=\s*(?:手机|电话|电脑|汽车|平板|手表|耳机|芯片|"
                r"服务器|机器人|电视|相机|无人机|音箱|软件|系统))"
            )
            for m in pattern.finditer(text):
                name = m.group().strip()
                if self._valid_product(name, text[m.end():m.end() + 8]):
                    results.append(self._entity(m, name, "PRODUCT", "known_model"))

        # “Model Y 是一款电动汽车”：型号后没有紧邻品类，但是强判断句。
        for known in _KNOWN_PRODUCTS:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(known)}\s+[A-Z0-9][A-Z0-9+.\-]*"
                r"(?=\s*(?:是一款|是一种|属于|作为))"
            )
            for m in pattern.finditer(text):
                results.append(self._entity(m, m.group().strip(),
                                            "PRODUCT", "model_apposition"))

        # 纯中文产品只接受“名称 + 品类词”的完整短语，不把“新产品”当产品。
        for term in _CATEGORY_TERMS:
            for m in re.finditer(rf"([一-鿿]{{2,8}}?){re.escape(term)}", text):
                head = m.group(1)
                if not head or head in ("一款", "一个", "这个", "那个", "新",
                                        "这款", "该款", "所有", "其他"):
                    continue
                if head.endswith(("的", "了", "一款", "一个")):
                    continue
                if head[0] in ("是", "属", "在", "和", "与"):
                    continue
                # “智能手机品类”中“智能手机”是品类，不是产品。
                following = text[m.end():m.end() + 3]
                if following.startswith(_CATEGORY_HEADS):
                    continue
                results.append(self._entity(m, m.group(), "PRODUCT",
                                            "zh_product_category"))
        return self._dedupe_entities(results)

    @staticmethod
    def _valid_product(name: str, following: str) -> bool:
        clean = name.strip()
        if len(clean) < 2 or clean.lower() in {"the", "this", "that"}:
            return False
        if re.fullmatch(r"[A-Z]?\d+(?:\.\d+)?", clean):
            return False
        if following and re.match(r"^[A-Za-z]{3,}", following.lstrip()):
            return False
        return True

    def _find_categories(self, text: str, entities: list[dict]) -> list[dict]:
        results: list[dict] = []
        for head in _CATEGORY_HEADS:
            for m in re.finditer(rf"([一-鿿]{{2,10}}?){head}", text):
                label = m.group()
                for prefix in _BELONGS_CUES:
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                label = re.sub(r"^(?:一款|一种|一个|是)", "", label)
                # “产品品类”过泛；至少要有具体品类名。
                if label in {"产品品类", "商品品类", "品类"} or len(label) < 2:
                    continue
                start = m.start() + (len(m.group()) - len(label))
                results.append({"start": start, "end": start + len(label),
                                "text": label, "type": "CATEGORY",
                                "rule": "explicit_head"})

        # “智能手机 / 电动汽车”常作为品类判断的宾语。
        for m in re.finditer(r"(?:是一款|是一种|是一个|属于)\s*"
                             r"([一-鿿]{2,12}?)(?=[。！？!?，,；;；\s]|$)", text):
            label = m.group(1)
            if label in _CATEGORY_TERMS or any(label.endswith(t) for t in _CATEGORY_TERMS):
                start = m.start(1)
                results.append({"start": start, "end": start + len(label),
                                "text": label, "type": "CATEGORY",
                                "rule": "category_predicate"})
        return self._dedupe_entities(results)

    def _find_titles(self, text: str) -> list[dict]:
        results = []
        for suffix in _TITLE_SUFFIXES:
            for m in re.finditer(rf"[一-鿿A-Za-z]{{0,10}}{re.escape(suffix)}",
                                 text):
                label = m.group()
                for cue in ("担任", "出任", "任职", "作为", "职位是"):
                    if label.startswith(cue):
                        label = label[len(cue):]
                        break
                if len(label) < 2 or label in ("的" + suffix, suffix):
                    continue
                start = m.end() - len(label)
                results.append({"start": start, "end": m.end(),
                                "text": label, "type": "TITLE",
                                "rule": "title_suffix"})
        return self._dedupe_entities(results)

    @staticmethod
    def _entity(match: re.Match, text: str, etype: str, rule: str) -> dict:
        start = match.start(1) if match.groups() else match.start()
        return {"start": start, "end": start + len(text), "text": text,
                "type": etype, "rule": rule}

    # -- 单句关系 ---------------------------------------------------------
    def _extract_sentence(self, sentence: str, entities: list[dict],
                          s_start: int, s_end: int) -> list[dict]:
        relations: list[dict] = []
        local = sentence
        people = self._by_type(entities, "PERSON")
        orgs = self._by_type(entities, "ORGANIZATION")
        locations = self._by_type(entities, "LOCATION")
        dates = self._by_type(entities, "DATE", "TIME")
        products = self._by_type(entities, "PRODUCT")
        categories = self._by_type(entities, "CATEGORY")
        money = self._by_type(entities, "MONEY")
        titles = self._by_type(entities, "TITLE")
        self._add_employment(local, s_start, people, orgs, dates, relations)
        self._add_founding(local, s_start, people, orgs, dates, relations)
        self._add_title(local, s_start, people, titles, relations)
        self._add_org_location(local, s_start, orgs, locations, dates, relations)
        self._add_product_category(local, s_start, products, categories, relations)
        self._add_org_product(local, s_start, people, orgs, products, dates,
                              relations)
        self._add_price(local, s_start, products, money, relations)
        self._add_events(local, s_start, people, orgs, dates, relations)
        return relations

    def _add_employment(self, sentence: str, offset: int, people, orgs,
                        dates, relations) -> None:
        for person in people:
            for cue in _WORDS_CUES:
                cue_rel = self._cue_position(sentence, person, cue)
                if cue_rel is None:
                    continue
                clause, clause_rel = self._clause_at(sentence, cue_rel)
                clause_span = (offset + clause_rel,
                               offset + clause_rel + len(clause))
                person_abs = offset + sentence.index(person["text"])
                org = self._nearest_entity_after(orgs, person_abs, clause_span)
                if org:
                    date = self._date_for_cue(dates, offset + cue_rel,
                                              clause_span)
                    self._append(relations, person, "WORKS_AT", org,
                                 sentence, offset, "employment_cue",
                                 trigger=cue, qualifiers=self._qualifiers(date))
                    break

    def _add_founding(self, sentence: str, offset: int, people, orgs,
                      dates, relations) -> None:
        for person in people:
            for cue in _FOUND_CUES:
                cue_rel = self._cue_position(sentence, person, cue)
                if cue_rel is None:
                    continue
                clause_span = self._span_from_rel(sentence, cue_rel, offset)
                person_abs = offset + sentence.index(person["text"])
                org = self._nearest_entity_after(orgs, person_abs, clause_span)
                if org:
                    date = self._date_for_cue(dates, offset + cue_rel,
                                              clause_span)
                    self._append(relations, person, "FOUNDED", org,
                                 sentence, offset, "founding_cue",
                                 trigger=cue,
                                 qualifiers=self._qualifiers(date))
                    break

    def _add_title(self, sentence: str, offset: int, people, titles,
                   relations) -> None:
        for person in people:
            for cue in _TITLE_CUES:
                cue_rel = self._cue_position(sentence, person, cue)
                if cue_rel is None:
                    continue
                person_abs = offset + sentence.index(person["text"])
                clause_span = self._span_from_rel(sentence, cue_rel, offset)
                title = self._nearest_entity_after(titles, person_abs,
                                                   clause_span)
                if title:
                    self._append(relations, person, "SERVED_AS", title,
                                 sentence, offset, "title_cue", trigger=cue)
                    break

    def _add_org_location(self, sentence: str, offset: int, orgs, locations,
                          dates, relations) -> None:
        for org in orgs:
            for cue in _LOCATION_CUES:
                if cue not in sentence:
                    continue
                # “成立于2024年”不应错误关联到地名；“位于”才直接使用。
                for loc in locations:
                    if cue in ("位于", "总部位于", "坐落于"):
                        if loc["start"] > org["end"]:
                            self._append(relations, org, "LOCATED_IN", loc,
                                         sentence, offset, "location_cue",
                                         trigger=cue)
                    elif (cue == "在" and org["start"] <= sentence.find(cue)
                          and loc["start"] < org["start"]):
                        # “北京的阿里巴巴公司”由后续 WORKS_AT 负责，不抽机构位于。
                        continue

    def _add_product_category(self, sentence: str, offset: int, products,
                              categories, relations) -> None:
        for product in products:
            # 紧邻：Mate 60 Pro 手机品类
            for category in categories:
                if category["start"] == product["end"] or \
                        (category["start"] > product["end"]
                         and category["start"] - product["end"] <= 2):
                    self._append(relations, product, "BELONGS_TO", category,
                                 sentence, offset, "adjacent_category")
                    break
            else:
                # 判断句：Model Y 是一款电动汽车
                tail = sentence[product["end"] - offset:]
                if any(cue in tail for cue in _BELONGS_CUES):
                    sentence_span = (offset, offset + len(sentence))
                    category = self._nearest_entity_after(
                        categories, product["end"], sentence_span)
                    if category:
                        self._append(relations, product, "BELONGS_TO",
                                     category, sentence, offset,
                                     "category_predicate")

    def _add_org_product(self, sentence: str, offset: int, people, orgs,
                         products, dates, relations) -> None:
        for product in products:
            pred = "PRODUCES"
            cue_found = None
            for cue in _LAUNCH_CUES:
                if cue in sentence:
                    pred = "LAUNCHES" if "发布" in cue or "推出" in cue or "发售" in cue else "PRODUCES"
                    cue_found = cue
                    break
            if cue_found is None:
                for cue in ("生产", "制造"):
                    if cue in sentence:
                        pred, cue_found = "PRODUCES", cue
                        break
            if not cue_found:
                continue

            actor = self._nearest_entity_before(orgs, product["start"])
            if actor:
                date = self._date_for_cue(dates, offset + sentence.find(cue_found))
                self._append(relations, actor, pred, product, sentence,
                             offset, "org_product_cue", trigger=cue_found,
                             qualifiers=self._qualifiers(date))
                continue

            actor = self._nearest_entity_before(people, product["start"])
            if actor and pred == "LAUNCHES":
                date = self._date_for_cue(dates, offset + sentence.find(cue_found))
                self._append(relations, actor, pred, product, sentence,
                             offset, "person_product_cue", trigger=cue_found,
                             qualifiers=self._qualifiers(date))

    def _add_price(self, sentence: str, offset: int, products, money,
                   relations) -> None:
        for amount in money:
            if not any(k in sentence for k in ("价格", "售价", "卖", "定价", "元", "美元")):
                continue
            product = self._nearest_entity_before(products, amount["start"])
            if product:
                self._append(relations, product, "PRICED_AT", amount,
                             sentence, offset, "price_pattern")

    def _add_events(self, sentence: str, offset: int, people, orgs, dates,
                    relations) -> None:
        # 只抽取“主体 + 明确动作 + 名词客体”的事件，动作客体过泛时放弃。
        actors = people + orgs
        for actor in actors:
            actor_idx = sentence.find(actor["text"])
            tail = sentence[actor_idx + len(actor["text"]):]
            # 不跨越逗号/分号把上一个主语绑定到另一个小句，避免无主语时硬猜。
            boundary = _CLAUSE_SPLIT_RE.search(tail)
            if boundary:
                tail = tail[:boundary.start()]
            m = re.search(r"(发布了?|推出了?|生产了?|制造了?|创办了?|创立了?|"
                          r"完成了?|收购了?|投资了?|签署了?|发布了?)\s*"
                          r"([一-鿿A-Za-z0-9]{2,20})", tail)
            if not m:
                continue
            verb, obj = m.group(1), m.group(2)
            if obj in {"新产品", "产品", "系统", "一个"} or \
                    self._looks_like_entity(obj):
                if obj in {"新产品", "产品", "系统"}:
                    # 允许为“新产品”创建事件，但事件名保留为动作短语。
                    pass
                elif self._looks_like_entity(obj):
                    continue
            # 如果已经抽成机构-产品直接关系，则不重复造事件。
            if any(r["relation"] in ("PRODUCES", "LAUNCHES", "FOUNDED", "WORKS_AT")
                   and r["source"]["text"] == actor["text"]
                   and r["target"]["text"] in sentence for r in relations):
                continue
            event_text = f"{verb}{obj}"
            date = self._date_for_cue(
                dates, offset + actor_idx + len(actor["text"]) + m.start(1),
                (offset, offset + len(sentence)))
            start = offset + actor_idx + len(actor["text"]) + m.start(2)
            event = {"start": start, "end": start + len(obj),
                     "text": event_text, "type": "EVENT", "rule": "svo_event",
                     "identity_text": f"{event_text}@"
                                      f"{date['text'] if date else '未知日期'}"}
            self._append(relations, actor, "PERFORMED", event, sentence,
                         offset, "svo_event", trigger=verb,
                         qualifiers=self._qualifiers(date))
            if date:
                self._append(relations, event, "OCCURRED_AT", date, sentence,
                             offset, "event_date",
                             qualifiers=self._qualifiers(date))

    # -- 构造三元组 -------------------------------------------------------
    def _append(self, relations: list[dict], source: dict, predicate: str,
                target: dict, sentence: str, offset: int, rule: str,
                trigger: str = "", qualifiers: Optional[dict] = None,
                confidence: float = 0.82) -> None:
        if not self._valid_direction(predicate, source["type"], target["type"]):
            return
        # 目标若是事件，关系构造时给出的偏移是对象偏移，名称为动作+对象。
        if target["type"] == "EVENT":
            verb_len = max(0, len(target["text"]) - (target["end"] - target["start"]))
            identity = target.get("identity_text")
            target = dict(target)
            target["start"] -= verb_len
            if identity:
                target["identity_text"] = identity
        if source["start"] == target["start"] and source["end"] == target["end"]:
            return
        rid = hashlib.sha1(
            f"{predicate}|{source['type']}:{source['text']}|"
            f"{target['type']}:{target['text']}".encode("utf-8")
        ).hexdigest()[:16]
        relations.append({
            "id": rid,
            "source": self._ref(source),
            "relation": predicate,
            "relation_name": RELATION_NAMES[predicate],
            "target": self._ref(target),
            "confidence": round(confidence, 3),
            "trigger": trigger,
            "rule": rule,
            "qualifiers": qualifiers or {},
            "evidence": {
                "sentence": sentence,
                "start": offset,
                "end": offset + len(sentence),
            },
        })

    @staticmethod
    def _ref(entity: dict) -> dict:
        return {"text": entity["text"], "type": entity["type"],
                "start": entity["start"], "end": entity["end"]}

    @staticmethod
    def _valid_direction(predicate: str, source_type: str,
                         target_type: str) -> bool:
        sources, targets = RELATION_DOMAINS.get(predicate, ((), ()))
        return source_type in sources and target_type in targets

    # -- 查找工具 ---------------------------------------------------------
    @staticmethod
    def _by_type(entities: Iterable[dict], *types: str) -> list[dict]:
        return [e for e in entities if e["type"] in types]

    @staticmethod
    def _cue_position(sentence: str, entity: dict, cue: str) -> Optional[int]:
        idx = sentence.find(entity["text"])
        if idx < 0:
            return None
        tail_start = idx + len(entity["text"])
        tail = sentence[tail_start:tail_start + 18]
        pos = tail.find(cue)
        return tail_start + pos if pos >= 0 else None

    @staticmethod
    def _clause_at(sentence: str, pos: int) -> tuple[str, int]:
        parts = list(_CLAUSE_SPLIT_RE.finditer(sentence))
        start, end = 0, len(sentence)
        for m in parts:
            if m.start() <= pos:
                start = m.end()
            elif m.start() > pos:
                end = m.start()
                break
        if start > pos:
            start = 0
        return sentence[start:end], start

    @staticmethod
    def _span_from_rel(sentence: str, pos: int, offset: int) -> tuple[int, int]:
        _, rel_start = RelationExtractor._clause_at(sentence, pos)
        end = len(sentence)
        for m in _CLAUSE_SPLIT_RE.finditer(sentence):
            if m.start() > pos:
                end = m.start()
                break
        return offset + rel_start, offset + end

    @staticmethod
    def _nearest_entity_after(entities: list[dict], absolute_after: int,
                              span: tuple[int, int] | None = None):
        candidates = list(entities)
        if span:
            start, end = span
            candidates = [e for e in entities if start <= e["start"] < end]
        candidates = [e for e in candidates if e["start"] >= absolute_after]
        return min(candidates, key=lambda e: e["start"], default=None)

    @staticmethod
    def _nearest_entity_before(entities: list[dict],
                               absolute_before: int):
        candidates = [e for e in entities if e["end"] <= absolute_before]
        return max(candidates, key=lambda e: e["end"], default=None)

    @staticmethod
    def _date_for_cue(dates: list[dict], cue_abs: int,
                      span: tuple[int, int] | None = None):
        candidates = list(dates)
        if span:
            start, end = span
            candidates = [d for d in dates if start <= d["start"] < end]
        before = [d for d in candidates if d["end"] <= cue_abs + 2]
        after = [d for d in candidates if d["start"] >= cue_abs]
        if before:
            return max(before, key=lambda d: d["end"])
        if after:
            return min(after, key=lambda d: d["start"])
        return None

    @staticmethod
    def _qualifiers(date: Optional[dict]) -> dict:
        if not date:
            return {}
        return {"time": {"text": date["text"], "type": date["type"],
                         "start": date["start"], "end": date["end"]}}

    @staticmethod
    def _looks_like_entity(text: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:元|年|月|日|%|万|亿)?", text))

    # -- 去重与规范化 ------------------------------------------------------
    def _deduplicate(self, relations: list[dict]) -> list[dict]:
        best: dict[tuple, dict] = {}
        for rel in relations:
            key = (rel["relation"], rel["source"]["type"],
                   rel["source"]["text"], rel["target"]["type"],
                   rel["target"]["text"], rel["evidence"]["sentence"])
            if key not in best or rel["confidence"] > best[key]["confidence"]:
                best[key] = rel
        return sorted(best.values(),
                      key=lambda r: (r["evidence"]["start"], r["source"]["start"]))

    @staticmethod
    def _dedupe_entities(entities: list[dict]) -> list[dict]:
        result: dict[tuple[int, int], dict] = {}
        for ent in entities:
            key = (ent["start"], ent["end"])
            result.setdefault(key, ent)
        return list(result.values())

    @staticmethod
    def _overlaps_entity(entity: dict, entities: list[dict]) -> bool:
        return any(entity["start"] < e["end"] and e["start"] < entity["end"]
                   for e in entities)


def normalize_date(text: str) -> str:
    """把常见中文日期统一为 YYYY-MM-DD / YYYY-MM，便于同值日期合并。"""
    value = re.sub(r"\s+", "", text)
    m = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", value)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    m = re.fullmatch(r"(\d{4})年(\d{1,2})月", value)
    if m:
        y, mo = m.groups()
        return f"{y}-{int(mo):02d}"
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})日?", value)
    if m:
        mo, d = m.groups()
        return f"{int(mo):02d}-{int(d):02d}"
    return text.strip()


def canonical_entity(entity_type: str, text: str) -> str:
    """保守生成跨文档对齐用的规范名。

    只做无歧义清理（空白、标点、公司/品类后缀等），不做同义词猜测。
    别名/简称合并由人工 alias 接口显式完成，避免“苹果（公司）”和
    “苹果（水果）”这类张冠李戴。
    """
    value = re.sub(r"\s+", "", text.strip(" 　\t\r\n，,。.!！?？；;"))
    if entity_type == "ORGANIZATION":
        for suffix in ("有限责任公司", "股份有限公司", "有限公司", "公司", "集团"):
            if value.endswith(suffix) and len(value) > len(suffix):
                value = value[:-len(suffix)]
                break
    elif entity_type == "CATEGORY":
        value = re.sub(r"(品类|类别|分类|产品类目)$", "", value)
    elif entity_type in ("DATE", "TIME"):
        value = normalize_date(value)
    elif entity_type == "PRODUCT":
        value = re.sub(r"\s+", " ", text.strip()).upper()
    elif entity_type == "EVENT":
        value = text.strip()
    return value or text.strip()


def node_id_for(entity_type: str, canonical: str) -> str:
    digest = hashlib.sha1(
        f"{entity_type}|{canonical}".encode("utf-8")
    ).hexdigest()[:16]
    return f"node_{digest}"
