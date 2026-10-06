"""知识图谱：实体关系联合抽取 + 跨文档对齐 + 幂等合并。

运行：``python -m unittest tests.test_kg -v``

覆盖用户的核心要求：
1. 同时抽出实体和有方向关系（任职/品类/事件时间地点）；
2. 触发词是必要条件——无关系不硬造；
3. 跨文档把同一对象（不同写法）对齐到同一节点；
4. 同一对实体在不同句子/文档里的关系合并成一条边；
5. 节点可溯源到参与的原句；
6. 幂等入库、删除文档不冲乱其它文档的关系。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.relation import RelationExtractor, RELATION_SCHEMA
from storage import StoreRegistry, KnowledgeGraph, node_key


def _rels(result, relation=None):
    out = [(r["head"]["text"], r["relation"], r["tail"]["text"])
           for r in result["relations"]]
    if relation:
        out = [t for t in out if t[1] == relation]
    return out


class TestRelationExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = RelationExtractor()

    def test_person_works_at_org(self):
        r = self.ex.extract("马云任职于阿里巴巴集团。")
        self.assertIn(("马云", "works_at", "阿里巴巴"), _rels(r))
        # 关系有方向且带溯源
        rel = next(x for x in r["relations"] if x["relation"] == "works_at")
        self.assertEqual(rel["head"]["text"], "马云")
        self.assertEqual(rel["tail"]["text"], "阿里巴巴")
        self.assertEqual(rel["sentence"], "马云任职于阿里巴巴集团")
        self.assertTrue(rel["action"])

    def test_product_belongs_to_category(self):
        r = self.ex.extract("Mate60属于智能手机品类。")
        self.assertTrue(_rels(r, "belongs_to"))
        head_text, _, tail_text = _rels(r, "belongs_to")[0]
        self.assertIn("Mate60", head_text)
        self.assertIn("品类", tail_text)
        # 扩展出 PRODUCT / CATEGORY 实体
        types = {e["type"] for e in r["entities"]}
        self.assertIn("CATEGORY", types)

    def test_org_launches_product_and_event_time(self):
        r = self.ex.extract("华为在2023年9月发布了全新的Mate60手机。")
        self.assertTrue(_rels(r, "launches"))
        # 「谁在何时做了什么事」——补出 occurred_at 边
        events = _rels(r, "occurred_at")
        self.assertTrue(events, "应补出主体到时间的发生于边")
        self.assertEqual(events[0][0], "华为")
        self.assertIn("2023", events[0][2])

    def test_no_trigger_no_relation(self):
        """没有触发词，实体共现也不得硬造关系。"""
        r = self.ex.extract("马云和马化腾出现在同一个会场，旁边有百度的人。")
        self.assertTrue(r["entities"])
        self.assertEqual(r["relations"], [])

    def test_relation_direction(self):
        r = self.ex.extract("特斯拉收购了SolarCity。")
        rels = _rels(r, "acquires")
        self.assertTrue(rels)
        self.assertEqual(rels[0][0], "特斯拉")
        self.assertEqual(rels[0][2], "SolarCity")

    def test_invest_marker_after_object(self):
        r = self.ex.extract("腾讯向美团投资。")
        self.assertIn(("腾讯", "invests", "美团"), _rels(r))

    def test_headquarters_specific(self):
        r = self.ex.extract("华为总部位于深圳。")
        names = {x["relation"] for x in r["relations"]}
        self.assertIn("headquartered_in", names)
        self.assertNotIn("located_in", names)

    def test_passive_reverse_requires_marker(self):
        """反向规则必须带被动标记，避免正句被反向再连一次。"""
        r = self.ex.extract("特斯拉收购了SolarCity。")
        acq = [x for x in r["relations"] if x["relation"] == "acquires"]
        self.assertEqual(len(acq), 1)
        self.assertEqual(acq[0]["head"]["text"], "特斯拉")

    def test_no_cross_clause_mismatch(self):
        """逗号两侧不应张冠李戴：深圳不属于智能手机品类。"""
        r = self.ex.extract(
            "华为总部位于深圳，Mate60属于智能手机品类。")
        for h, rel, t in _rels(r):
            if rel == "belongs_to":
                self.assertNotIn("深圳", t)
                self.assertNotEqual(h, "深圳")

    def test_relation_schema_consistency(self):
        for key, (name, rev, h, t) in RELATION_SCHEMA.items():
            self.assertTrue(name and rev)


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kg_test_")
        self.reg = StoreRegistry(self.tmp, shard_size=20)
        self.kg = KnowledgeGraph(self.reg)
        self.ex = RelationExtractor()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ingest(self, text, doc_id, name=None):
        return self.kg.ingest(self.ex.extract(text), doc_id,
                              name or doc_id, text)

    def test_cross_doc_entity_alignment(self):
        """「阿里巴巴集团」与「阿里」应跨文档对齐为同一节点。"""
        self._ingest("马云任职于阿里巴巴集团。", "d1")
        self._ingest("阿里在杭州投资了达摩院。", "d2")
        ali_nodes = [n for n in self.kg.nodes.values()
                     if n["name"] == "阿里巴巴"]
        self.assertEqual(len(ali_nodes), 1, "不同写法必须合并成一个节点")
        node = ali_nodes[0]
        self.assertIn("阿里", node["aliases"])
        self.assertEqual(sorted(node["doc_ids"]), ["d1", "d2"])

    def test_relation_merge_across_docs(self):
        """同一对实体的同关系跨文档合并成一条边，证据数=2。"""
        self._ingest("马云任职于阿里巴巴集团。", "d1")
        self._ingest("马云在阿里继续任职。", "d2")
        edge = next(e for e in self.kg.edges.values()
                    if e["relation"] == "works_at")
        self.assertEqual(edge["weight"], 2)
        self.assertEqual(sorted(edge["doc_ids"]), ["d1", "d2"])
        self.assertEqual(len(edge["evidence"]), 2)

    def test_idempotent_reingest(self):
        self._ingest("马云任职于阿里巴巴集团。腾讯向美团投资。", "d1")
        before = self.kg.stats()
        self._ingest("马云任职于阿里巴巴集团。腾讯向美团投资。", "d1")
        after = self.kg.stats()
        self.assertEqual(before, after, "重复入库不得产生重复节点/边")

    def test_node_sentence_provenance(self):
        self._ingest("马云任职于阿里巴巴集团。", "d1", "报道一")
        mayun = next(k for k, n in self.kg.nodes.items()
                     if n["name"] == "马云")
        detail = self.kg.node_detail(mayun)
        sentences = {s["sentence"] for s in detail["sentences"]}
        self.assertIn("马云任职于阿里巴巴集团", sentences)

    def test_remove_doc_preserves_others(self):
        self._ingest("马云任职于阿里巴巴集团。", "d1")
        self._ingest("腾讯向美团投资。", "d2")
        self.assertTrue(self.kg.remove_doc("d2"))
        # d1 的关系原样保留
        rels = {e["relation"] for e in self.kg.edges.values()}
        self.assertIn("works_at", rels)
        self.assertNotIn("invests", rels)
        names = {n["name"] for n in self.kg.nodes.values()}
        self.assertIn("马云", names)
        self.assertNotIn("腾讯", names)
        self.assertNotIn("美团", names)

    def test_type_scoped_identity(self):
        """同名但类型不同的实体不能合并。"""
        k1 = node_key("苹果", "ORGANIZATION")
        k2 = node_key("苹果", "PRODUCT")
        self.assertNotEqual(k1, k2)

    def test_date_normalization(self):
        from storage.kg import canonical_name
        self.assertEqual(canonical_name("2023年9月", "DATE"), "2023-09")
        self.assertEqual(
            canonical_name("2023/09/01", "DATE"), "2023-09-01")
        self.assertEqual(
            canonical_name("阿里巴巴集团", "ORGANIZATION"), "阿里巴巴")

    def test_persistence_reload(self):
        self._ingest("马云任职于阿里巴巴集团。", "d1")
        reloaded = KnowledgeGraph(self.reg).load()
        self.assertEqual(reloaded.stats()["nodes"], self.kg.stats()["nodes"])
        self.assertEqual(reloaded.stats()["edges"], self.kg.stats()["edges"])

    def test_sharded_documents_uneven(self):
        """语料分片、文档多寡不一：逐篇入库后图谱完整。"""
        texts = [
            "华为总部位于深圳。",
            "华为在2023年9月发布了Mate60手机。",
            "腾讯向美团投资。特斯拉收购了SolarCity。",
            "比亚迪生产海豹电动车。",
            "雷军创立了小米。",
        ]
        for i, t in enumerate(texts):
            self._ingest(t, f"doc{i}")
        stats = self.kg.stats()
        self.assertEqual(stats["docs"], 5)
        self.assertGreaterEqual(stats["edges"], 5)
        # 华为在两篇文档里出现，但只有一个节点
        huawei = [n for n in self.kg.nodes.values() if n["name"] == "华为"]
        self.assertEqual(len(huawei), 1)
        self.assertEqual(len(huawei[0]["doc_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
