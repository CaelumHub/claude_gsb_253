"""NLP 平台单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：分词、词性、句法、NER、情感、摘要、翻译、关键词、词向量、
分片存储（含并发锁）、流水线引擎、HMM。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import (get_segmenter, get_tagger, get_parser, get_constituency_parser,
                 get_ner, get_sentiment, get_summarizer, get_translator,
                 get_keywords, get_embeddings, get_relation_extractor, TAGSET)
from nlp.hmm import HMM
from pipeline import PipelineEngine, PipelineError
from storage import ShardedStore, StoreRegistry, KnowledgeGraphStore


class TestSegmenter(unittest.TestCase):
    def test_basic(self):
        words = get_segmenter().cut("自然语言处理是人工智能的重要分支")
        self.assertIn("自然语言", words)
        self.assertIn("人工智能", words)
        self.assertIn("是", words)

    def test_english_number(self):
        words = get_segmenter().cut("我用Python写了100行代码")
        self.assertIn("Python", words)
        self.assertIn("100", words)


class TestPOSTagger(unittest.TestCase):
    def test_tags(self):
        pairs = get_tagger().tag("我学习自然语言处理")
        self.assertTrue(pairs)
        for word, tag in pairs:
            self.assertIn(tag, TAGSET, f"{word}:{tag}")

    def test_punct_as_other(self):
        pairs = get_tagger().tag("你好，世界。")
        tags = [t for _, t in pairs]
        for t in tags:
            if t in ("，", "。"):
                continue
        # 标点词本身应为 x
        for w, t in pairs:
            if w in ("，", "。"):
                self.assertEqual(t, "x")


class TestParser(unittest.TestCase):
    def test_dependency(self):
        dep = get_parser().parse("北京大学的研究团队开发了机器学习系统")
        self.assertEqual(len(dep["words"]), len(dep["heads"]))
        self.assertIn(-1, dep["heads"])  # 存在根
        # 每个 head 都是有效下标或 -1
        for h in dep["heads"]:
            self.assertTrue(h == -1 or 0 <= h < len(dep["words"]))

    def test_constituency_spans(self):
        c = get_constituency_parser().parse("北京大学的研究团队开发了系统")
        leaves = self._leaves(c["tree"])
        self.assertEqual("北京大学的研究团队开发了系统", leaves)

    @staticmethod
    def _leaves(tree):
        if not tree.get("children"):
            return tree.get("word", "")
        return "".join(TestParser._leaves(ch) for ch in tree["children"])


class TestNER(unittest.TestCase):
    def test_known_entities(self):
        ents = get_ner().recognize("马云在北京工作")
        types = {e["text"]: e["type"] for e in ents}
        self.assertEqual(types.get("马云"), "PERSON")
        self.assertEqual(types.get("北京"), "LOCATION")

    def test_date_money(self):
        ents = get_ner().recognize("2024年10月1日花了99.9元")
        texts = [e["text"] for e in ents]
        self.assertTrue(any("2024" in t for t in texts))
        self.assertTrue(any("99.9" in t for t in texts))


class TestRelations(unittest.TestCase):
    def test_employment_and_title(self):
        result = get_relation_extractor().extract(
            "张三在2024年5月加入字节跳动，担任算法工程师。")
        triples = {(r["source"]["text"], r["relation"], r["target"]["text"])
                   for r in result["relations"]}
        self.assertIn(("张三", "WORKS_AT", "字节跳动"), triples)
        self.assertIn(("张三", "SERVED_AS", "算法工程师"), triples)
        works = next(r for r in result["relations"] if r["relation"] == "WORKS_AT")
        self.assertEqual(works["qualifiers"]["time"]["text"], "2024年5月")
        self.assertEqual(works["evidence"]["sentence"][:2], "张三")

    def test_product_category_price(self):
        result = get_relation_extractor().extract(
            "华为在2023年8月发布了Mate 60 Pro手机，Mate 60 Pro属于智能手机品类，售价5999元。")
        triples = {(r["source"]["text"], r["relation"], r["target"]["text"])
                   for r in result["relations"]}
        self.assertIn(("华为", "LAUNCHES", "Mate 60 Pro"), triples)
        self.assertIn(("Mate 60 Pro", "BELONGS_TO", "智能手机品类"), triples)
        self.assertIn(("Mate 60 Pro", "PRICED_AT", "5999元"), triples)

    def test_actor_is_not_guessed_across_clause(self):
        result = get_relation_extractor().extract(
            "马云在北京的阿里巴巴公司工作，2024年10月1日发布了新产品。")
        triples = {(r["source"]["text"], r["relation"], r["target"]["text"])
                   for r in result["relations"]}
        self.assertIn(("马云", "WORKS_AT", "阿里巴巴"), triples)
        self.assertFalse(any(r["target"]["text"] == "发布了新产品"
                             for r in result["relations"]))

    def test_generic_event_has_time_and_direction(self):
        result = get_relation_extractor().extract("华为在2023年8月发布了新产品。")
        event_edge = next(r for r in result["relations"]
                          if r["relation"] == "PERFORMED")
        self.assertEqual(event_edge["source"]["type"], "ORGANIZATION")
        self.assertEqual(event_edge["target"]["type"], "EVENT")
        self.assertEqual(event_edge["qualifiers"]["time"]["text"], "2023年8月")
        self.assertTrue(any(r["relation"] == "OCCURRED_AT"
                            for r in result["relations"]))

    def test_no_relation_is_fabricated(self):
        result = get_relation_extractor().extract(
            "今天的天气和北京的交通都很普通。")
        self.assertEqual([], result["relations"])


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.kg = KnowledgeGraphStore(StoreRegistry(self.tmp, shard_size=10))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cross_document_merge_and_evidence(self):
        text_a = "张三在2024年5月加入字节跳动。"
        text_b = "张三在2025年1月加入字节跳动。"
        self.kg.ingest_text(text_a, doc_id="a", name="A")
        self.kg.ingest_text(text_b, doc_id="b", name="B")
        graph = self.kg.graph()
        self.assertEqual(graph["stats"]["documents"], 2)
        edge = next(e for e in graph["edges"] if e["relation"] == "WORKS_AT")
        self.assertEqual(edge["doc_count"], 2)
        self.assertEqual(edge["mention_count"], 2)
        self.assertEqual(len(edge["qualifiers"]["all_times"]), 2)
        detail = self.kg.node_detail(edge["source_id"])
        self.assertTrue(any("张三" in m["sentence"] for m in detail["mentions"]))

    def test_reingest_is_idempotent(self):
        text = "华为发布了Mate 60 Pro手机，Mate 60 Pro属于智能手机品类。"
        self.kg.ingest_text(text, doc_id="d")
        first = self.kg.stats()
        self.kg.ingest_text(text, doc_id="d")
        self.assertEqual(first, self.kg.stats())

    def test_alias_rebuild(self):
        self.kg.ingest_text("阿里巴巴公司发布了新产品。", doc_id="a")
        node = next(n for n in self.kg.nodes.all()
                    if n["type"] == "ORGANIZATION" and n["canonical"] == "阿里巴巴")
        # 规范名已自动去掉公司后缀，因此两篇文档应对到同一节点。
        self.kg.ingest_text("阿里巴巴发布了另一个产品。", doc_id="b")
        self.assertEqual(node["id"], next(
            n["id"] for n in self.kg.nodes.all()
            if n["type"] == "ORGANIZATION" and n["canonical"] == "阿里巴巴"))

    def test_explicit_alias_aligns_documents(self):
        self.kg.ingest_text("阿里巴巴公司发布了新产品。", doc_id="a")
        node = next(n for n in self.kg.nodes.all()
                    if n["type"] == "ORGANIZATION" and n["canonical"] == "阿里巴巴")
        self.kg.add_alias(node["id"], "阿里")
        self.kg.ingest_text("阿里发布了新产品。", doc_id="b")
        self.kg.rebuild()
        merged = self.kg.nodes.get(node["id"])
        self.assertEqual(merged["doc_count"], 2)
        self.assertIn("阿里", merged["aliases"])
        org_count = sum(1 for n in self.kg.nodes.all()
                        if not n.get("_deleted") and n["type"] == "ORGANIZATION")
        self.assertEqual(org_count, 1)


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        r = get_sentiment().analyze("这个产品非常好用，我很喜欢")
        self.assertEqual(r["polarity"], "positive")

    def test_negative(self):
        r = get_sentiment().analyze("服务态度很差，令人失望")
        self.assertEqual(r["polarity"], "negative")


class TestSummarizer(unittest.TestCase):
    def test_shorter(self):
        text = ("自然语言处理是人工智能的重要分支。它研究如何让计算机理解语言。"
                "分词是基础任务。词性标注是另一个任务。")
        r = get_summarizer().summarize(text, ratio=0.5)
        self.assertTrue(len(r["summary"]) < len(text))
        self.assertTrue(r["top_indices"])


class TestTranslator(unittest.TestCase):
    def test_zh2en(self):
        r = get_translator().translate("我喜欢机器学习", "zh2en")
        self.assertIn("machine learning", r["translation"].lower())

    def test_en2zh(self):
        r = get_translator().translate("I like China", "en2zh")
        self.assertTrue(r["translation"])


class TestKeywords(unittest.TestCase):
    def test_extract(self):
        r = get_keywords().extract("自然语言处理是人工智能的重要分支", top_k=5)
        self.assertTrue(r["keywords"])
        for k in r["keywords"]:
            self.assertIn("word", k)
            self.assertIn("score", k)


class TestEmbeddings(unittest.TestCase):
    def test_train_nearest(self):
        texts = [
            "自然语言处理是人工智能的重要分支",
            "机器学习是人工智能的核心技术",
            "深度学习推动了人工智能的发展",
            "分词是自然语言处理的基础任务",
        ] * 3
        emb = get_embeddings()
        emb.train(texts, vocab_size=60, dim=8, window=3, min_count=1)
        self.assertTrue(emb.vocab)
        self.assertTrue(emb.vectors)
        # 近邻应返回词且不包含自身
        nb = emb.nearest(emb.vocab[0], k=3)
        self.assertTrue(nb)
        self.assertNotIn(emb.vocab[0], [n["word"] for n in nb])
        # 2D 投影
        proj = emb.project_2d()
        self.assertEqual(len(proj), len(emb.vectors))


class TestHMM(unittest.TestCase):
    def test_viterbi(self):
        hmm = HMM(["A", "B"], add_k=0.1)
        hmm.train([[(1, "A"), (2, "B")], [(1, "A"), (2, "B")], [(2, "B"), (1, "A")]])
        path = hmm.viterbi([1, 2])
        self.assertEqual(len(path), 2)
        self.assertIn(path[0], ("A", "B"))


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_insert_query(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        store.insert_many([{"v": i} for i in range(25)])
        self.assertEqual(store.stats()["total"], 25)
        self.assertEqual(store.stats()["shard_count"], 3)
        self.assertEqual(len(store.query(where=[("v", "gt", 20)])), 4)
        self.assertEqual(len(store.query(where=[("v", "in", [1, 2, 3])])), 3)

    def test_delete_compact(self):
        store = ShardedStore(self.tmp, "t", shard_size=10)
        ids = store.insert_many([{"v": i} for i in range(15)])
        store.delete(ids[0])
        stats = store.compact()
        self.assertEqual(stats["records"], 14)

    def test_concurrent_insert(self):
        store = ShardedStore(self.tmp, "t", shard_size=20)
        errors = []

        def worker(offset):
            try:
                store.insert_many([{"v": offset * 1000 + i} for i in range(30)])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(store.stats()["total"], 180)

    def test_registry_tasks(self):
        reg = StoreRegistry(self.tmp)
        reg.task("a").insert({"x": 1})
        reg.task("b").insert({"x": 2})
        # 造一个非存储目录，不应被识别为任务
        os.makedirs(os.path.join(self.tmp, "models"))
        self.assertEqual(reg.tasks(), ["a", "b"])


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = PipelineEngine().register_builtin()

    def test_run_chain(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment"}, {"name": "pos"}, {"name": "sentiment"}]}
        out = self.engine.build(cfg).run({"text": "这个产品非常好用"})
        self.assertIn("words", out)
        self.assertIn("pos", out)
        self.assertIn("sentiment", out)

        cfg = {"name": "p", "stages": [
            {"name": "segment"}, {"name": "ner"}, {"name": "knowledge_graph"}]}
        out = self.engine.build(cfg).run(
            {"text": "张三在2024年5月加入字节跳动。"})
        self.assertIn("knowledge_graph", out)
        self.assertTrue(out["knowledge_graph"]["relations"])

    def test_batch(self):
        cfg = {"name": "p", "stages": [{"name": "segment"}, {"name": "keywords"}]}
        results = self.engine.run_batch(
            cfg, ["今天天气很好", "这个产品非常好用"], max_workers=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["ok"] for r in results))

    def test_cycle_detected(self):
        cfg = {"name": "p", "stages": [
            {"name": "segment", "deps": ["pos"]},
            {"name": "pos", "deps": ["segment"]},
        ]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)

    def test_missing_stage(self):
        cfg = {"name": "p", "stages": [{"name": "not_exist"}]}
        with self.assertRaises(PipelineError):
            self.engine.build(cfg)


if __name__ == "__main__":
    unittest.main()
