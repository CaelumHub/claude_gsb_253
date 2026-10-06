"""跨文档知识图谱存储。

抽取结果按三层保存：

- ``kg_node``：实体/事件节点，规范 ID 由「类型 + 规范名」决定；
- ``kg_edge``：有向关系边，ID 由「源节点 + 谓词 + 目标节点 + 时间限定」决定；
- ``kg_mention``：节点和边在原文句子中的证据。

节点/边是跨文档合并后的视图，mention 保留“这句话为什么这么连”。重新导入
同一篇文档时先删除该文档的旧证据，再刷新聚合计数，因此不会重复堆叠，也
不会因为新增文档冲乱已有 ID。别名合并必须显式指定，系统不做猜测式消歧。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Optional

from nlp.relation import (RelationExtractor, RELATION_NAMES, canonical_entity,
                          node_id_for)
from storage.lock import FileLock, lock_path_for
from storage.sharded import ShardedStore, StoreRegistry, _atomic_write_json, _read_json


NODE_TASK = "kg_node"
EDGE_TASK = "kg_edge"
MENTION_TASK = "kg_mention"
DOC_TASK = "kg_document"
ALIAS_FILE = "kg_aliases.json"


class KnowledgeGraphStore:
    """实体关系图谱的增量入库、对齐、查询和证据回溯。"""

    def __init__(self, registry: StoreRegistry,
                 extractor: Optional[RelationExtractor] = None):
        self.registry = registry
        self.root = registry.root
        self.nodes = registry.task(NODE_TASK)
        self.edges = registry.task(EDGE_TASK)
        self.mentions = registry.task(MENTION_TASK)
        self.documents = registry.task(DOC_TASK)
        self.extractor = extractor or RelationExtractor()
        self.alias_path = os.path.join(self.root, ALIAS_FILE)
        self._alias_lock = threading.RLock()

    # -- 导入 -------------------------------------------------------------
    def ingest_text(self, text: str, doc_id: Optional[str] = None,
                    name: str = "", corpus_id: Optional[str] = None,
                    metadata: Optional[dict] = None) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("文档内容不能为空")
        content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        doc_id = doc_id or corpus_id or f"kgdoc_{content_hash[:16]}"

        with FileLock(lock_path_for(os.path.join(self.root, ".kg.lock"))):
            self._delete_document_mentions(doc_id)
            aliases = self.load_aliases()
            extracted = self.extractor.extract(text)
            extracted = self._add_alias_entities(text, extracted, aliases)

            node_by_ref: dict[tuple[str, int, int], tuple[str, dict, dict]] = {}
            edge_records: list[dict] = []
            mention_records: list[dict] = []
            now = time.time()

            # 所有实体先成节点；没有关系的实体也保留，但不会硬造边。
            for entity in extracted["entities"]:
                node_id, node, mention = self._make_node_and_mention(
                    entity, doc_id, text, aliases, now)
                self.nodes.upsert(node)
                mention_records.append(mention)
                node_by_ref[(entity["type"], entity["start"], entity["end"])] = \
                    (node_id, node, entity)

            for relation in extracted["relations"]:
                source_id, source_node, source_ref = self._relation_endpoint(
                    relation["source"], doc_id, text, aliases, now,
                    mention_records)
                target_id, target_node, target_ref = self._relation_endpoint(
                    relation["target"], doc_id, text, aliases, now,
                    mention_records)
                if not source_id or not target_id or source_id == target_id:
                    continue

                edge = self._make_edge(relation, source_id, target_id,
                                       source_node, target_node, doc_id)
                edge_records.append(edge)
                mention_records.append(self._make_edge_mention(
                    relation, edge["id"], source_id, target_id, doc_id, name,
                    source_ref, target_ref))

            for edge in edge_records:
                self.edges.upsert(edge)
            for mention in mention_records:
                self.mentions.upsert(mention)

            doc = {
                "id": doc_id,
                "name": name or f"文档_{doc_id}",
                "corpus_id": corpus_id,
                "text": text,
                "length": len(text),
                "content_hash": content_hash,
                "entity_count": len(extracted["entities"]),
                "relation_count": len(extracted["relations"]),
                "metadata": metadata or {},
                "ingested_at": now,
            }
            self.documents.upsert(doc)
            self._refresh_counts()
            return {
                "doc_id": doc_id,
                "nodes": len(extracted["entities"]),
                "relations": len(extracted["relations"]),
                "mentions": len(mention_records),
                "extracted": extracted,
            }

    def delete_document(self, doc_id: str) -> bool:
        with FileLock(lock_path_for(os.path.join(self.root, ".kg.lock"))):
            existed = self.documents.delete(doc_id)
            self._delete_document_mentions(doc_id)
            self._refresh_counts()
            return existed

    def rebuild(self) -> dict:
        """从已保存文档重建全部节点、边和证据。"""
        with FileLock(lock_path_for(os.path.join(self.root, ".kg.lock"))):
            docs = [d for d in self.documents.all() if not d.get("_deleted")]
            for store in (self.mentions, self.edges, self.nodes):
                store.clear()
            result = {"documents": 0, "entities": 0, "relations": 0}
            for doc in docs:
                aliases = self.load_aliases()
                extracted = self.extractor.extract(doc["text"])
                extracted = self._add_alias_entities(doc["text"], extracted,
                                                     aliases)
                mentions: list[dict] = []
                node_by_ref: dict[tuple[str, int, int], tuple[str, dict]] = {}
                now = time.time()

                for entity in extracted["entities"]:
                    node_id, node, mention = self._make_node_and_mention(
                        entity, doc["id"], doc["text"], aliases, now)
                    self.nodes.upsert(node)
                    mentions.append(mention)
                    node_by_ref[(entity["type"], entity["start"],
                                 entity["end"])] = (node_id, node)

                for relation in extracted["relations"]:
                    src = self._resolved(relation["source"], aliases)
                    tgt = self._resolved(relation["target"], aliases)
                    if not src or not tgt or src[0] == tgt[0]:
                        continue
                    edge = self._make_edge(relation, src[0], tgt[0],
                                           {"name": src[2], "type": src[1]},
                                           {"name": tgt[2], "type": tgt[1]},
                                           doc["id"])
                    self.edges.upsert(edge)
                    mentions.append(self._make_edge_mention(
                        relation, edge["id"], src[0], tgt[0], doc["id"],
                        doc.get("name", ""), relation["source"],
                        relation["target"]))
                for mention in mentions:
                    self.mentions.upsert(mention)
                result["documents"] += 1
                result["entities"] += len(extracted["entities"])
                result["relations"] += len(extracted["relations"])
            self._refresh_counts()
            return result

    # -- 别名实体注入 -----------------------------------------------------
    def _add_alias_entities(self, text: str, extracted: dict,
                            alias_map: dict) -> dict:
        entities = list(extracted.get("entities", []))
        added_aliases = []
        for key, info in alias_map.items():
            if "|" not in key:
                continue
            etype, alias = key.split("|", 1)
            start = text.find(alias)
            while start >= 0:
                entity = {"start": start, "end": start + len(alias),
                          "text": alias, "type": etype, "rule": "manual_alias"}
                if not any(entity["start"] < e["end"] and e["start"] < entity["end"]
                           for e in entities):
                    entities.append(entity)
                    added_aliases.append(entity)
                start = text.find(alias, start + len(alias))
        if entities == extracted.get("entities", []):
            return extracted
        # 用“基础 NER + 人工别名”作为种子重新抽取，避免在已有实体之外再造一个
        # 同位置的产品/品类实体。
        reextracted = self.extractor.extract(text, known_entities=added_aliases)
        return {"entities": reextracted["entities"],
                "relations": reextracted["relations"],
                "entity_type_names": extracted.get("entity_type_names", {}),
                "relation_names": extracted.get("relation_names", {})}

    # -- 节点/边构造 ------------------------------------------------------
    def _make_node_and_mention(self, entity: dict, doc_id: str, text: str,
                               aliases: dict, now: float):
        node_id, canonical, etype = self._resolved(entity, aliases)
        raw = entity["text"]
        node = {
            "id": node_id,
            "name": raw if etype == "EVENT" else canonical,
            "canonical": canonical,
            "type": etype,
            "aliases": sorted({raw} if raw != canonical else set()),
            "doc_count": 0,
            "mention_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        mention = self._make_node_mention(node_id, entity, doc_id, text)
        return node_id, node, mention

    def _relation_endpoint(self, ref: dict, doc_id: str, text: str,
                           aliases: dict, now: float,
                           pending_mentions: list[dict]):
        node_id, canonical, etype = self._resolved(ref, aliases)
        node = {
            "id": node_id,
            "name": ref["text"] if etype == "EVENT" else canonical,
            "canonical": canonical,
            "type": etype,
            "aliases": sorted({ref["text"]} if ref["text"] != canonical else set()),
            "doc_count": 0,
            "mention_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        self.nodes.upsert(node)
        mention_ids = {m["id"] for m in pending_mentions}
        mention = self._make_node_mention(node_id, ref, doc_id, text)
        if mention["id"] not in mention_ids:
            pending_mentions.append(mention)
        return node_id, node, ref

    def _resolved(self, ref: dict, aliases: Optional[dict] = None):
        etype = ref["type"]
        raw = ref["text"]
        identity = ref.get("identity_text") if etype == "EVENT" else raw
        alias_key = f"{etype}|{raw.strip()}"
        if aliases and alias_key in aliases:
            canonical = aliases[alias_key]["canonical"]
        else:
            canonical = canonical_entity(etype, identity or raw)
        return node_id_for(etype, canonical), canonical, etype

    @staticmethod
    def _make_edge(relation: dict, source_id: str, target_id: str,
                   source_node: dict, target_node: dict, doc_id: str) -> dict:
        key_source = f"{source_id}|{relation['relation']}|{target_id}"
        edge_id = "edge_" + hashlib.sha1(
            key_source.encode("utf-8")).hexdigest()[:16]
        return {
            "id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation["relation"],
            "relation_name": RELATION_NAMES.get(
                relation["relation"], relation["relation"]),
            "source_name": source_node.get("name"),
            "target_name": target_node.get("name"),
            "source_type": source_node.get("type"),
            "target_type": target_node.get("type"),
            "qualifiers": relation.get("qualifiers", {}),
            "confidence": relation.get("confidence", 0.0),
            "rules": [relation.get("rule", "")] if relation.get("rule") else [],
            "doc_count": 0,
            "mention_count": 0,
            "first_doc": doc_id,
            "updated_at": time.time(),
        }

    def _make_node_mention(self, node_id: str, entity: dict, doc_id: str,
                           full_text: str) -> dict:
        sentence = self._sentence_at(full_text, entity.get("start", 0),
                                     entity.get("end", 0))
        raw = f"{node_id}|{doc_id}|{sentence}|{entity.get('start')}|{entity.get('end')}"
        return {
            "id": "mention_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16],
            "kind": "node",
            "node_id": node_id,
            "doc_id": doc_id,
            "sentence": sentence,
            "text": entity["text"],
            "entity_type": entity["type"],
            "start": entity.get("start"),
            "end": entity.get("end"),
            "created_at": time.time(),
        }

    def _make_edge_mention(self, relation: dict, edge_id: str,
                           source_id: str, target_id: str, doc_id: str,
                           doc_name: str, source_ref: dict,
                           target_ref: dict) -> dict:
        evidence = relation["evidence"]
        raw = (f"{edge_id}|{doc_id}|{evidence['sentence']}|"
               f"{source_ref.get('start')}|{target_ref.get('start')}|"
               f"{json.dumps(relation.get('qualifiers', {}), ensure_ascii=False)}")
        return {
            "id": "mention_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16],
            "kind": "edge",
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation["relation"],
            "relation_name": relation.get("relation_name"),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "sentence": evidence["sentence"],
            "start": evidence.get("start"),
            "end": evidence.get("end"),
            "trigger": relation.get("trigger", ""),
            "rule": relation.get("rule", ""),
            "confidence": relation.get("confidence", 0),
            "qualifiers": relation.get("qualifiers", {}),
            "source_text": source_ref.get("text"),
            "target_text": target_ref.get("text"),
            "created_at": time.time(),
        }

    # -- 删除与聚合 -------------------------------------------------------
    def _delete_document_mentions(self, doc_id: str) -> None:
        mentions = [m for m in self.mentions.all()
                    if not m.get("_deleted") and m.get("doc_id") == doc_id]
        if mentions:
            self.mentions.delete_many(m["id"] for m in mentions)

    def _refresh_counts(self) -> None:
        active = [m for m in self.mentions.all() if not m.get("_deleted")]
        node_mentions = [m for m in active if m["kind"] == "node"]
        edge_mentions = [m for m in active if m["kind"] == "edge"]
        self._refresh_nodes(node_mentions)
        self._refresh_edges(edge_mentions)

    def _refresh_nodes(self, mentions: list[dict]) -> None:
        by_node: dict[str, list[dict]] = {}
        for mention in mentions:
            by_node.setdefault(mention["node_id"], []).append(mention)

        keep_ids = set(by_node)
        for node in self.nodes.all():
            if node.get("_deleted") or node["id"] in keep_ids:
                continue
            self.nodes.delete(node["id"])

        for node_id, items in by_node.items():
            existing = self.nodes.get(node_id) or {}
            alias_map = self.load_aliases()
            aliases = set(existing.get("aliases", []))
            explicit = {key.split("|", 1)[1]
                        for key, value in alias_map.items()
                        if value.get("node_id") == node_id}
            aliases.update(m["text"] for m in items if m.get("text"))
            aliases.update(explicit)
            doc_count = len({m["doc_id"] for m in items})
            record = dict(existing)
            record.update({
                "id": node_id,
                "type": items[0]["entity_type"],
                "canonical": existing.get("canonical") or items[0]["text"],
                "name": existing.get("name") or self._display_name(
                    items[0]["entity_type"],
                    existing.get("canonical") or items[0]["text"],
                    items[0]["text"]),
                "aliases": sorted(a for a in aliases if a),
                "doc_count": doc_count,
                "mention_count": len(items),
                "updated_at": time.time(),
            })
            record.setdefault("created_at", time.time())
            self.nodes.upsert(record)

    def _refresh_edges(self, mentions: list[dict]) -> None:
        by_edge: dict[str, list[dict]] = {}
        for mention in mentions:
            by_edge.setdefault(mention["edge_id"], []).append(mention)

        keep_ids = set(by_edge)
        for edge in self.edges.all():
            if edge.get("_deleted") or edge["id"] in keep_ids:
                continue
            self.edges.delete(edge["id"])

        for edge_id, items in by_edge.items():
            existing = self.edges.get(edge_id) or {}
            times = []
            for item in items:
                value = (item.get("qualifiers") or {}).get("time")
                if value and value not in times:
                    times.append(value)
            rules = sorted({i["rule"] for i in items if i.get("rule")})
            record = dict(existing)
            record.update({
                "id": edge_id,
                "source_id": items[0]["source_id"],
                "target_id": items[0]["target_id"],
                "relation": items[0]["relation"],
                "relation_name": items[0]["relation_name"],
                "source_name": items[0].get("source_text") or existing.get("source_name"),
                "target_name": items[0].get("target_text") or existing.get("target_name"),
                "source_type": existing.get("source_type"),
                "target_type": existing.get("target_type"),
                "rules": rules,
                "confidence": max(float(i.get("confidence") or 0) for i in items),
                "doc_count": len({i["doc_id"] for i in items}),
                "mention_count": len(items),
                "updated_at": time.time(),
            })
            if times:
                record["qualifiers"] = {"time": times[0], "all_times": times}
            record.setdefault("first_doc", items[0]["doc_id"])
            record.setdefault("created_at", time.time())
            self.edges.upsert(record)

    # -- 查询 -------------------------------------------------------------
    def graph(self, limit_nodes: int = 300, limit_edges: int = 600,
              node_type: Optional[str] = None,
              relation: Optional[str] = None,
              query: Optional[str] = None) -> dict:
        mentions = [m for m in self.mentions.all() if not m.get("_deleted")]
        node_ids = None
        if relation:
            node_ids = set()
        edges = []
        for edge in self.edges.all():
            if edge.get("_deleted"):
                continue
            if relation and edge["relation"] != relation:
                continue
            edges.append(edge)
            if node_ids is not None:
                node_ids.update((edge["source_id"], edge["target_id"]))

        nodes = []
        q = (query or "").strip().lower()
        for node in self.nodes.all():
            if node.get("_deleted"):
                continue
            if node_type and node["type"] != node_type:
                continue
            if node_ids is not None and node["id"] not in node_ids:
                continue
            if q and not self._node_matches(node, q):
                continue
            nodes.append(node)
        nodes.sort(key=lambda n: n.get("mention_count", 0), reverse=True)
        selected_ids = {n["id"] for n in nodes[:limit_nodes]}
        if relation or node_type or q:
            edges = [e for e in edges
                     if e["source_id"] in selected_ids and e["target_id"] in selected_ids]
        return {
            "nodes": nodes[:limit_nodes],
            "edges": edges[:limit_edges],
            "stats": self.stats(),
        }

    def node_detail(self, node_id: str) -> Optional[dict]:
        node = self.nodes.get(node_id)
        if not node or node.get("_deleted"):
            return None
        mentions = [m for m in self.mentions.all()
                    if not m.get("_deleted")
                    and (m.get("node_id") == node_id
                         or m.get("source_id") == node_id
                         or m.get("target_id") == node_id)]
        edge_ids = {m["edge_id"] for m in mentions if m.get("edge_id")}
        related_edges = [e for e in self.edges.all()
                         if not e.get("_deleted") and e["id"] in edge_ids]
        return {"node": node, "edges": related_edges, "mentions": mentions}

    def edge_detail(self, edge_id: str) -> Optional[dict]:
        edge = self.edges.get(edge_id)
        if not edge or edge.get("_deleted"):
            return None
        mentions = [m for m in self.mentions.all()
                    if not m.get("_deleted") and m.get("edge_id") == edge_id]
        return {"edge": edge, "mentions": mentions}

    def stats(self) -> dict:
        nodes = [n for n in self.nodes.all() if not n.get("_deleted")]
        edges = [e for e in self.edges.all() if not e.get("_deleted")]
        docs = [d for d in self.documents.all() if not d.get("_deleted")]
        relation_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for edge in edges:
            relation_counts[edge["relation"]] = relation_counts.get(edge["relation"], 0) + 1
        for node in nodes:
            type_counts[node["type"]] = type_counts.get(node["type"], 0) + 1
        return {
            "documents": len(docs),
            "nodes": len(nodes),
            "edges": len(edges),
            "node_types": type_counts,
            "relations": relation_counts,
        }

    @staticmethod
    def _display_name(entity_type: str, canonical: str, raw: str = "") -> str:
        if entity_type == "PRODUCT" and raw:
            return re.sub(r"\s+", " ", raw.strip())
        return canonical

    @staticmethod
    def _node_matches(node: dict, q: str) -> bool:
        values = [node.get("name", ""), node.get("canonical", "")]
        values.extend(node.get("aliases", []))
        return any(q in str(v).lower() for v in values)

    @staticmethod
    def _sentence_at(text: str, start: int, end: int) -> str:
        start = max(0, int(start or 0))
        end = max(start, int(end or start))
        left = max(text.rfind("。", 0, start), text.rfind("！", 0, start),
                   text.rfind("？", 0, start), text.rfind("\n", 0, start),
                   text.rfind("；", 0, start))
        right_candidates = [text.find(p, end) for p in ("。", "！", "？", "\n", "；")]
        right_candidates = [p for p in right_candidates if p >= 0]
        right = min(right_candidates) + 1 if right_candidates else len(text)
        return text[left + 1:right].strip()

    # -- 别名 -------------------------------------------------------------
    def load_aliases(self) -> dict:
        with self._alias_lock:
            data = _read_json(self.alias_path, {})
            return data.get("aliases", {}) if isinstance(data, dict) else {}

    def add_alias(self, node_id: str, alias: str) -> dict:
        alias = alias.strip()
        if not alias:
            raise ValueError("别名不能为空")
        with FileLock(lock_path_for(self.alias_path)):
            data = _read_json(self.alias_path, {"version": 1, "aliases": {}})
            aliases = data.setdefault("aliases", {})
            node = self.nodes.get(node_id)
            if not node or node.get("_deleted"):
                raise ValueError("目标节点不存在")
            key = f"{node['type']}|{alias}"
            old = aliases.get(key)
            if old and old.get("node_id") != node_id:
                raise ValueError("该别名已经指向另一个节点")
            aliases[key] = {
                "node_id": node_id,
                "type": node["type"],
                "canonical": node["canonical"],
                "created_at": time.time(),
            }
            _atomic_write_json(self.alias_path, data)
        self.rebuild()
        return {"ok": True, "node_id": node_id, "alias": alias}

    def compact(self) -> dict:
        result = {}
        for name, store in (("nodes", self.nodes), ("edges", self.edges),
                            ("mentions", self.mentions),
                            ("documents", self.documents)):
            result[name] = store.compact()
        return result
