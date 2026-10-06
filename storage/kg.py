"""知识图谱存储：实体对齐 + 有向关系合并 + 证据溯源。

在分片 JSON 存储（:class:`~storage.sharded.ShardedStore`）之上构建图谱层，
解决四件事：

1. **跨文档实体对齐（同一对象对上号）**：
   节点身份 = ``类型 + 规范名``。规范名做机构后缀归一（阿里巴巴集团→阿里巴巴）、
   繁简/空白归一、日期格式归一；同时维护别名表（阿里 ↔ 阿里巴巴集团），
   后续文档以别名出现时自动并入同一节点，绝不张冠李戴（类型不同绝不合并）。
2. **关系合并成一条边**：同一 (主体, 关系, 客体, 触发动作) 的多次提及
   （不同句子甚至不同文档）合并为一条边，边权重 = 提及次数，证据挂在边上。
3. **证据溯源**：每条证据保存文档 id、句子原文、句内字符偏移，
   点节点即可看到它参与的全部原句。
4. **增量更新不冲乱已有关系**：按文档幂等入库，重复入库只替换本文档的
   贡献，不产生重复边；删除文档时只移除它独有的证据与节点。

底层三类记录（分别落在 ``kg_node`` / ``kg_edge`` / ``kg_doc`` 三个任务目录）：

    node: {key, name, type, aliases[], mention_count, doc_ids[],
           first_seen, last_seen}
    edge: {key, head, relation, tail, action, weight, doc_ids[],
           evidence[{doc_id, doc_name, sentence, sent_start, head_text,
                     tail_text, action, qualifiers, confidence, created_at}]}
    doc:  {doc_id, name, text, node_keys[], edge_keys[], ingested_at}
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Optional

from .sharded import ShardedStore, StoreRegistry


# ---------------------------------------------------------------------------
# 实体规范与别名
# ---------------------------------------------------------------------------

#: 机构通名后缀：归一为核心专名（仅在存在更长核心名时剥离，避免「集团公司」变空）
_ORG_GENERIC_SUFFIXES = (
    "集团股份有限公司", "股份有限公司", "有限责任公司", "集团有限公司",
    "技术有限公司", "科技有限公司", "有限公司", "股份公司", "集团公司",
    "公司", "集团",
)
#: 地点通名（保留核心地名）
_LOC_SUFFIXES = ("市", "省", "自治区", "特别行政区")
#: 全局别名表：别名 -> 规范名（跨文档把不同写法对到同一节点）
_GLOBAL_ALIASES = {
    "阿里": "阿里巴巴",
    "阿里巴巴集团": "阿里巴巴",
    "阿里巴巴公司": "阿里巴巴",
    "阿里云": "阿里云",
    "腾讯公司": "腾讯",
    "腾讯集团": "腾讯",
    "华为公司": "华为",
    "华为技术有限公司": "华为",
    "小米公司": "小米",
    "小米集团": "小米",
    "苹果公司": "苹果",
    "谷歌公司": "谷歌",
    "微软公司": "微软",
    "比亚迪公司": "比亚迪",
    "比亚迪股份有限公司": "比亚迪",
}

#: 节点类型优先级（观察到多种类型时，取更「专名」的类型展示）
_TYPE_PRIORITY = {
    "PERSON": 0, "ORGANIZATION": 1, "LOCATION": 2,
    "PRODUCT": 3, "CATEGORY": 4, "DATE": 5, "TIME": 6,
    "MONEY": 7, "PERCENT": 8, "NUMBER": 9,
}

_DATE_FULL_RE = re.compile(
    r"^(\d{4})年(?:(\d{1,2})月)?(?:(\d{1,2})日)?$"
    r"|^(\d{4})[/-](\d{1,2})(?:[/-](\d{1,2}))?$")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[「」『』\"'（）()【】\[\]、，,。！？!?；;：:]+")


def canonical_name(text: str, etype: str) -> str:
    """把同一对象的不同写法归一为规范名。"""
    name = _WS_RE.sub("", (text or "").strip())
    if not name:
        return name
    if etype == "ORGANIZATION":
        if name in _GLOBAL_ALIASES:
            return _GLOBAL_ALIASES[name]
        core = name
        for suffix in _ORG_GENERIC_SUFFIXES:
            if core.endswith(suffix) and len(core) - len(suffix) >= 2:
                core = core[: -len(suffix)]
                break
        return _GLOBAL_ALIASES.get(core, core)
    if etype == "LOCATION":
        core = name
        for suffix in _LOC_SUFFIXES:
            if core.endswith(suffix) and len(core) - len(suffix) >= 2:
                core = core[: -len(suffix)]
                break
        return core
    if etype in ("DATE", "TIME"):
        return _normalize_date(name)
    if etype in ("MONEY", "NUMBER", "PERCENT"):
        return _normalize_number(name, etype)
    return name


def _normalize_date(name: str) -> str:
    """日期归一为 YYYY-MM-DD / YYYY-MM / YYYY，使不同写法指向同一节点。"""
    m = _DATE_FULL_RE.match(name)
    if not m:
        return name
    g = m.groups()
    if g[0] is not None:  # 中文 年/月/日
        year, month, day = g[0], g[1], g[2]
    else:                # 2023/09/01 或 2023-09
        year, month, day = g[3], g[4], g[5]
    if day:
        return f"{year}-{int(month):02d}-{int(day):02d}"
    if month:
        return f"{year}-{int(month):02d}"
    return year


def _normalize_number(name: str, etype: str) -> str:
    m = re.match(r"^([\d.]+)(.*)$", name)
    if not m:
        return name
    value = m.group(1)
    unit = m.group(2)
    try:
        value = str(float(value)).rstrip("0").rstrip(".")
    except ValueError:
        pass
    if etype == "PERCENT":
        unit = "%"
    return f"{value}{unit}"


def node_key(name: str, etype: str) -> str:
    """节点稳定身份：类型 + 规范名的短哈希（人类可读 + 防特殊字符）。"""
    canon = canonical_name(name, etype)
    digest = hashlib.md5(f"{etype}|{canon}".encode("utf-8")).hexdigest()[:12]
    return f"n_{etype.lower()}_{digest}"


def edge_key(head_key: str, relation: str, tail_key: str,
             action: str = "") -> str:
    """边稳定身份：同 (主体,关系,客体) 即同一条边。

    不同句子/文档里即便触发词写法不同（任职于 / 在…任职）也合并为一条；
    各种触发词在边上收集为 ``actions`` 标签。
    """
    raw = f"{head_key}|{relation}|{tail_key}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"e_{digest}"


def evidence_id(doc_id: str, head_key: str, tail_key: str,
                relation: str, sent_start: int) -> str:
    raw = f"{doc_id}|{head_key}|{tail_key}|{relation}|{sent_start}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:14]


# ---------------------------------------------------------------------------
# 图谱
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """跨文档知识图谱（线程安全；进程内缓存节点/边索引）。"""

    NODE_TASK = "kg_node"
    EDGE_TASK = "kg_edge"
    DOC_TASK = "kg_doc"

    def __init__(self, registry: StoreRegistry):
        self.registry = registry
        self._lock = threading.RLock()
        self._loaded = False
        # key -> node record
        self.nodes: dict[str, dict] = {}
        # edge key -> edge record
        self.edges: dict[str, dict] = {}
        # doc_id -> doc record
        self.docs: dict[str, dict] = {}
        # 别名 -> 节点 key（从已存节点的 aliases 构建）
        self._alias_index: dict[tuple[str, str], str] = {}

    # -- 存储句柄 ---------------------------------------------------------
    @property
    def node_store(self) -> ShardedStore:
        return self.registry.task(self.NODE_TASK)

    @property
    def edge_store(self) -> ShardedStore:
        return self.registry.task(self.EDGE_TASK)

    @property
    def doc_store(self) -> ShardedStore:
        return self.registry.task(self.DOC_TASK)

    # -- 载入 / 落盘 ------------------------------------------------------
    def load(self, force: bool = False) -> "KnowledgeGraph":
        with self._lock:
            if self._loaded and not force:
                return self
            self.nodes = {}
            self.edges = {}
            self.docs = {}
            self._alias_index = {}
            for r in self.node_store.all():
                if r.get("_deleted"):
                    continue
                self.nodes[r["key"]] = r
                self._index_node_aliases(r)
            for r in self.edge_store.all():
                if r.get("_deleted"):
                    continue
                self.edges[r["key"]] = r
            for r in self.doc_store.all():
                if r.get("_deleted"):
                    continue
                self.docs[r["doc_id"]] = r
            self._loaded = True
            return self

    def _index_node_aliases(self, node: dict) -> None:
        etype = node["type"]
        self._alias_index[(etype, node["name"])] = node["key"]
        for alias in node.get("aliases", []):
            self._alias_index[(etype, alias)] = node["key"]

    def _save_node(self, node: dict) -> None:
        self.node_store.upsert(node, key="key")

    def _save_edge(self, edge: dict) -> None:
        self.edge_store.upsert(edge, key="key")

    # -- 实体对齐 ---------------------------------------------------------
    def resolve(self, name: str, etype: str,
                learn_alias_from: Optional[str] = None) -> tuple[str, dict]:
        """把一次实体提及对齐到节点，返回 (node_key, node)。

        顺序：规范名精确命中 -> 别名命中 -> 新建。
        不同类型即使同名也绝不合并。
        """
        self.load()
        canon = canonical_name(name, etype)
        direct = self._alias_index.get((etype, canon))
        if direct:
            return direct, self.nodes[direct]
        if learn_alias_from:
            alias_canon = canonical_name(learn_alias_from, etype)
            hit = self._alias_index.get((etype, alias_canon))
            if hit:
                return hit, self.nodes[hit]
        # 新建节点
        key = node_key(name, etype)
        if key in self.nodes:
            return key, self.nodes[key]
        node = {
            "key": key,
            "name": canon,
            "type": etype,
            "aliases": sorted({a for a in (name.strip(), canon)
                               if a and a != canon}),
            "mention_count": 0,
            "doc_ids": [],
            "first_seen": time.time(),
            "last_seen": time.time(),
        }
        self.nodes[key] = node
        self._index_node_aliases(node)
        return key, node

    def _touch_node(self, key: str, surface: str, doc_id: str) -> None:
        node = self.nodes[key]
        surface = surface.strip()
        canon = node["name"]
        if surface and surface != canon and surface not in node["aliases"]:
            node["aliases"].append(surface)
            self._alias_index[(node["type"], surface)] = key
        node["mention_count"] = node.get("mention_count", 0) + 1
        if doc_id not in node["doc_ids"]:
            node["doc_ids"].append(doc_id)
        node["last_seen"] = time.time()

    # -- 文档入库（幂等） -------------------------------------------------
    def ingest(self, extraction: dict, doc_id: str,
               doc_name: str = "", text: str = "") -> dict:
        """把一篇文档的抽取结果并入图谱（幂等）。

        重复入库同一 ``doc_id`` 会先移除该文档此前的全部贡献再重新并入，
        已有其它文档的节点和边不受影响。
        """
        self.load()
        with self._lock:
            if doc_id in self.docs:
                self._remove_doc(doc_id)

            now = time.time()
            touched_nodes: set[str] = set()
            touched_edges: set[str] = set()

            # 1) 实体 -> 节点
            for ent in extraction.get("entities", []):
                key, _ = self.resolve(ent["text"], ent["type"])
                self._touch_node(key, ent["text"], doc_id)
                touched_nodes.add(key)

            # 2) 关系 -> 边（合并证据）
            for rel in extraction.get("relations", []):
                h, t = rel["head"], rel["tail"]
                hk, _ = self.resolve(h["text"], h["type"])
                tk, _ = self.resolve(t["text"], t["type"])
                self._touch_node(hk, h["text"], doc_id)
                self._touch_node(tk, t["text"], doc_id)
                touched_nodes.add(hk)
                touched_nodes.add(tk)

                action = (rel.get("action") or "").strip()
                ek = edge_key(hk, rel["relation"], tk, action)
                edge = self.edges.get(ek)
                if edge is None:
                    edge = {
                        "key": ek,
                        "head": hk,
                        "relation": rel["relation"],
                        "relation_name": rel.get(
                            "relation_name", rel["relation"]),
                        "tail": tk,
                        "action": action,
                        "actions": [action] if action else [],
                        "weight": 0,
                        "doc_ids": [],
                        "evidence": [],
                        "first_seen": now,
                    }
                    self.edges[ek] = edge
                if action and action not in edge["actions"]:
                    edge["actions"].append(action)
                ev = {
                    "eid": evidence_id(doc_id, hk, tk, rel["relation"],
                                       rel.get("sent_start", 0)),
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "sentence": rel.get("sentence", ""),
                    "sent_start": rel.get("sent_start", 0),
                    "head_text": h["text"],
                    "tail_text": t["text"],
                    "head_start": h.get("start"),
                    "head_end": h.get("end"),
                    "tail_start": t.get("start"),
                    "tail_end": t.get("end"),
                    "action": action,
                    "qualifiers": rel.get("qualifiers", []),
                    "confidence": rel.get("confidence", 0.0),
                    "created_at": now,
                }
                if not any(e["eid"] == ev["eid"] for e in edge["evidence"]):
                    edge["evidence"].append(ev)
                if doc_id not in edge["doc_ids"]:
                    edge["doc_ids"].append(doc_id)
                edge["weight"] = len(edge["evidence"])
                edge["last_seen"] = now
                touched_edges.add(ek)

            # 3) 文档记录
            doc_record = {
                "doc_id": doc_id,
                "name": doc_name or doc_id,
                "text": text,
                "node_keys": sorted(touched_nodes),
                "edge_keys": sorted(touched_edges),
                "node_count": len(touched_nodes),
                "edge_count": len(touched_edges),
                "ingested_at": now,
            }
            self.docs[doc_id] = doc_record
            doc_record["id"] = self.doc_store.upsert(doc_record, key="doc_id")

            # 4) 落盘
            for key in touched_nodes:
                stored_id = self.node_store.upsert(self.nodes[key], key="key")
                self.nodes[key]["id"] = stored_id
            for ek in touched_edges:
                stored_id = self.edge_store.upsert(self.edges[ek], key="key")
                self.edges[ek]["id"] = stored_id

            return {"doc_id": doc_id,
                    "nodes": len(touched_nodes),
                    "edges": len(touched_edges),
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges)}

    # -- 文档删除 ---------------------------------------------------------
    def remove_doc(self, doc_id: str) -> bool:
        self.load()
        with self._lock:
            if doc_id not in self.docs:
                return False
            self._remove_doc(doc_id)
            return True

    def _remove_doc(self, doc_id: str) -> None:
        """移除本文档对图谱的贡献；其它文档的证据原样保留。"""
        doc = self.docs.pop(doc_id, None)
        if doc is None:
            return

        # 边：删掉本文档证据；边若无其它证据则整条删除
        dead_edges: list[str] = []
        for ek in doc.get("edge_keys", []):
            edge = self.edges.get(ek)
            if edge is None:
                continue
            edge["evidence"] = [e for e in edge["evidence"]
                                if e["doc_id"] != doc_id]
            edge["doc_ids"] = [d for d in edge["doc_ids"] if d != doc_id]
            edge["weight"] = len(edge["evidence"])
            if edge["evidence"]:
                self._save_edge(edge)
            else:
                dead_edges.append(ek)

        # 节点：计数回滚；无其它文档引用则删除
        dead_nodes: list[str] = []
        for nk in doc.get("node_keys", []):
            node = self.nodes.get(nk)
            if node is None:
                continue
            node["doc_ids"] = [d for d in node["doc_ids"] if d != doc_id]
            # 提及计数无法精确按文档拆分时，用证据/文档数兜底修正
            still_in_edges = any(
                nk in (e["head"], e["tail"])
                for ek, e in self.edges.items() if ek not in dead_edges)
            if not node["doc_ids"] and not still_in_edges:
                dead_nodes.append(nk)
            else:
                node["mention_count"] = max(
                    0, node.get("mention_count", 1) - 1)
                self._save_node(node)

        for ek in dead_edges:
            self.edge_store.delete(self.edges[ek]["id"])
            del self.edges[ek]
        for nk in dead_nodes:
            node = self.nodes.pop(nk)
            self.node_store.delete(node["id"])
            for alias in [node["name"]] + node.get("aliases", []):
                self._alias_index.pop((node["type"], alias), None)

        self.doc_store.delete(doc["id"])

    # -- 查询 -------------------------------------------------------------
    def get_node(self, key: str) -> Optional[dict]:
        self.load()
        return self.nodes.get(key)

    def node_detail(self, key: str) -> Optional[dict]:
        """节点详情：节点本身 + 它参与的边（去重证据）+ 原句证据列表。"""
        self.load()
        node = self.nodes.get(key)
        if node is None:
            return None
        edges = []
        sentences: dict[str, dict] = {}
        for edge in self.edges.values():
            if key not in (edge["head"], edge["tail"]):
                continue
            edges.append(self._public_edge(edge))
            for ev in edge["evidence"]:
                sig = f"{ev['doc_id']}:{ev['sent_start']}"
                if sig not in sentences:
                    sentences[sig] = {
                        "doc_id": ev["doc_id"],
                        "doc_name": ev.get("doc_name", ""),
                        "sentence": ev["sentence"],
                        "sent_start": ev["sent_start"],
                        "relations": [],
                    }
                sentences[sig]["relations"].append({
                    "relation": edge["relation"],
                    "relation_name": edge["relation_name"],
                    "other": (self.nodes[edge["tail"]]["name"]
                              if edge["head"] == key
                              else self.nodes[edge["head"]]["name"]),
                    "direction": "out" if edge["head"] == key else "in",
                    "action": ev.get("action", ""),
                })
        node_out = dict(node)
        node_out["edges"] = edges
        node_out["sentences"] = sorted(
            sentences.values(), key=lambda s: s["sent_start"])
        return node_out

    def graph_view(self, relation_types: Optional[list[str]] = None,
                   doc_ids: Optional[list[str]] = None,
                   limit: int = 400) -> dict:
        """返回前端画图所需的 nodes/links（可按关系类型、文档过滤）。"""
        self.load()
        wanted_rels = set(relation_types) if relation_types else None
        wanted_docs = set(doc_ids) if doc_ids else None

        links = []
        used: set[str] = set()
        for edge in self.edges.values():
            if wanted_rels and edge["relation"] not in wanted_rels:
                continue
            evs = edge["evidence"]
            if wanted_docs:
                evs = [e for e in evs if e["doc_id"] in wanted_docs]
                if not evs:
                    continue
            links.append({
                "key": edge["key"],
                "source": edge["head"],
                "target": edge["tail"],
                "relation": edge["relation"],
                "relation_name": edge["relation_name"],
                "action": edge.get("action", ""),
                "weight": len(evs),
                "doc_count": len({e["doc_id"] for e in evs}),
            })
            used.add(edge["head"])
            used.add(edge["tail"])

        links.sort(key=lambda l: l["weight"], reverse=True)
        if len(links) > limit:
            links = links[:limit]
            used = {n for l in links for n in (l["source"], l["target"])}

        nodes = []
        for nk in used:
            node = self.nodes[nk]
            nodes.append({
                "key": nk,
                "name": node["name"],
                "type": node["type"],
                "mention_count": node.get("mention_count", 0),
                "doc_count": len(node.get("doc_ids", [])),
            })
        return {"nodes": nodes, "links": links}

    def _public_edge(self, edge: dict) -> dict:
        return {
            "key": edge["key"],
            "head": edge["head"],
            "head_name": self.nodes.get(edge["head"], {}).get("name", "?"),
            "tail": edge["tail"],
            "tail_name": self.nodes.get(edge["tail"], {}).get("name", "?"),
            "relation": edge["relation"],
            "relation_name": edge["relation_name"],
            "action": edge.get("action", ""),
            "weight": edge.get("weight", 0),
            "evidence": edge.get("evidence", []),
        }

    def list_edges(self, relation: Optional[str] = None) -> list[dict]:
        self.load()
        out = [self._public_edge(e) for e in self.edges.values()]
        if relation:
            out = [e for e in out if e["relation"] == relation]
        out.sort(key=lambda e: e["weight"], reverse=True)
        return out

    def list_docs(self) -> list[dict]:
        self.load()
        out = []
        for d in self.docs.values():
            out.append({k: v for k, v in d.items() if k != "text"})
        out.sort(key=lambda d: d.get("ingested_at", 0), reverse=True)
        return out

    def stats(self) -> dict:
        self.load()
        rel_counter: dict[str, int] = {}
        for edge in self.edges.values():
            rel_counter[edge["relation"]] = \
                rel_counter.get(edge["relation"], 0) + 1
        type_counter: dict[str, int] = {}
        for node in self.nodes.values():
            type_counter[node["type"]] = \
                type_counter.get(node["type"], 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "docs": len(self.docs),
            "evidence": sum(len(e["evidence"]) for e in self.edges.values()),
            "by_relation": rel_counter,
            "by_type": type_counter,
        }
