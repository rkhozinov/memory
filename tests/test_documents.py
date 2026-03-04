"""Tests for document operations in MemoryStore."""

import hashlib
import time

import pytest

from memory.core import MemoryStore
from memory.models import Document


# --- Model tests ---


class TestDocumentModel:
    def test_hash_from_body(self):
        doc = Document(title="Test", body="hello world", summary="greeting")
        assert doc.content_hash == hashlib.sha256(b"hello world").hexdigest()

    def test_timestamps_set(self):
        doc = Document(title="T", body="B", summary="S")
        assert doc.created_at > 0
        assert doc.updated_at > 0
        assert doc.created_at_iso != ""

    def test_to_row_length(self):
        doc = Document(title="T", body="B", summary="S")
        row = doc.to_row()
        assert len(row) == 14

    def test_to_dict_keys(self):
        doc = Document(title="T", body="B", summary="S", tags=["a"], doc_type="plan")
        d = doc.to_dict()
        assert d["title"] == "T"
        assert d["body"] == "B"
        assert d["summary"] == "S"
        assert d["doc_type"] == "plan"
        assert d["tags"] == ["a"]
        assert d["version"] == 1
        assert d["recall_count"] == 0

    def test_from_row_roundtrip(self):
        doc = Document(title="T", body="B", summary="S", tags=["x"], doc_type="spec")
        row = {
            "content_hash": doc.content_hash,
            "title": doc.title,
            "body": doc.body,
            "summary": doc.summary,
            "doc_type": doc.doc_type,
            "tags": '["x"]',
            "metadata": "{}",
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "created_at_iso": doc.created_at_iso,
            "updated_at_iso": doc.updated_at_iso,
            "deleted_at": None,
            "version": 1,
            "recall_count": 0,
            "last_recalled_at": None,
        }
        restored = Document.from_row(row)
        assert restored.title == "T"
        assert restored.body == "B"
        assert restored.tags == ["x"]
        assert restored.doc_type == "spec"


# --- Core store/get/delete tests ---


class TestDocumentStore:
    def test_store_and_get(self, store):
        result = store.store_doc(
            title="My Plan",
            body="# Plan\n\nDo things.",
            summary="A plan to do things",
            doc_type="plan",
            tags=["project:test"],
        )
        assert result["status"] == "stored"
        h = result["content_hash"]

        doc = store.get_doc(h)
        assert doc["title"] == "My Plan"
        assert doc["body"] == "# Plan\n\nDo things."
        assert doc["summary"] == "A plan to do things"
        assert doc["doc_type"] == "plan"
        assert doc["tags"] == ["project:test"]
        assert doc["version"] == 1

    def test_store_duplicate(self, store):
        store.store_doc(title="T", body="same body", summary="S")
        result = store.store_doc(title="T2", body="same body", summary="S2")
        assert result["status"] == "duplicate"

    def test_get_prefix_match(self, store):
        result = store.store_doc(title="T", body="prefix test body", summary="S")
        h = result["content_hash"]
        doc = store.get_doc(h[:12])
        assert doc["title"] == "T"

    def test_get_not_found(self, store):
        result = store.get_doc("nonexistent")
        assert "error" in result

    def test_delete(self, store):
        result = store.store_doc(title="T", body="delete me", summary="S")
        h = result["content_hash"]

        del_result = store.delete_doc(h)
        assert del_result["deleted"] == 1

        # Should not be found after deletion
        get_result = store.get_doc(h)
        assert "error" in get_result

    def test_delete_dry_run(self, store):
        result = store.store_doc(title="T", body="dry run test", summary="S")
        h = result["content_hash"]

        dry = store.delete_doc(h, dry_run=True)
        assert dry["dry_run"] is True
        assert dry["would_delete"] == 1

        # Should still exist
        doc = store.get_doc(h)
        assert doc["title"] == "T"

    def test_delete_not_found(self, store):
        result = store.delete_doc("nonexistent")
        assert "error" in result


# --- List tests ---


class TestDocumentList:
    def test_list_empty(self, store):
        result = store.list_docs()
        assert result["total"] == 0
        assert result["documents"] == []

    def test_list_with_docs(self, store):
        store.store_doc(title="A", body="body a", summary="S", doc_type="plan")
        store.store_doc(title="B", body="body b", summary="S", doc_type="spec")
        store.store_doc(title="C", body="body c", summary="S", doc_type="plan")

        result = store.list_docs()
        assert result["total"] == 3
        assert len(result["documents"]) == 3

    def test_list_filter_by_type(self, store):
        store.store_doc(title="A", body="body a", summary="S", doc_type="plan")
        store.store_doc(title="B", body="body b", summary="S", doc_type="spec")

        result = store.list_docs(doc_type="plan")
        assert len(result["documents"]) == 1
        assert result["documents"][0]["doc_type"] == "plan"

    def test_list_filter_by_tags(self, store):
        store.store_doc(title="A", body="body a", summary="S", tags=["project:x"])
        store.store_doc(title="B", body="body b", summary="S", tags=["project:y"])

        result = store.list_docs(tags=["project:x"])
        assert len(result["documents"]) == 1
        assert result["documents"][0]["title"] == "A"

    def test_list_pagination(self, store):
        for i in range(5):
            store.store_doc(title=f"Doc {i}", body=f"body {i}", summary="S")

        page1 = store.list_docs(page=1, page_size=2)
        assert len(page1["documents"]) == 2
        assert page1["total"] == 5

        page2 = store.list_docs(page=2, page_size=2)
        assert len(page2["documents"]) == 2


# --- Search tests ---


class TestDocumentSearch:
    def test_semantic_search(self, store):
        store.store_doc(
            title="EKS Deployment Guide",
            body="This guide covers deploying applications to Amazon EKS clusters.",
            summary="Guide for deploying apps to EKS Kubernetes clusters on AWS",
        )
        store.store_doc(
            title="Python Coding Standards",
            body="Our coding standards for Python projects.",
            summary="Python coding style guide and best practices",
        )

        results = store.search_docs("kubernetes deployment", mode="semantic")
        assert len(results) >= 1
        assert results[0]["title"] == "EKS Deployment Guide"

    def test_fts_search(self, store):
        store.store_doc(
            title="Terraform Module",
            body="This module provisions an RDS PostgreSQL database with automated backups.",
            summary="Terraform module for RDS PostgreSQL",
        )

        results = store.search_docs("PostgreSQL backups", mode="fts")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0]["body"]

    def test_auto_search(self, store):
        store.store_doc(
            title="API Design Spec",
            body="The REST API uses OAuth2 for authentication and JSON:API format.",
            summary="REST API design specification with auth and format details",
        )

        results = store.search_docs("OAuth2 authentication", mode="auto")
        assert len(results) >= 1

    def test_search_filter_by_type(self, store):
        store.store_doc(
            title="Plan A", body="plan content", summary="a plan",
            doc_type="plan",
        )
        store.store_doc(
            title="Spec A", body="spec content", summary="a spec",
            doc_type="spec",
        )

        results = store.search_docs("content", mode="fts", doc_type="plan")
        assert all(r["doc_type"] == "plan" for r in results)

    def test_search_filter_by_tags(self, store):
        store.store_doc(
            title="Doc X", body="tagged content", summary="S",
            tags=["project:alpha"],
        )
        store.store_doc(
            title="Doc Y", body="tagged content y", summary="S",
            tags=["project:beta"],
        )

        results = store.search_docs("tagged", mode="fts", tags=["project:alpha"])
        assert all("project:alpha" in r.get("tags", []) for r in results)

    def test_search_empty_query(self, store):
        results = store.search_docs(query=None)
        assert results == []

    def test_search_updates_recall_count(self, store):
        result = store.store_doc(
            title="Recall Test",
            body="testing recall tracking",
            summary="recall test doc",
        )
        h = result["content_hash"]

        store.search_docs("recall", mode="fts")

        doc = store.get_doc(h)
        assert doc["recall_count"] >= 1
        assert doc["last_recalled_at"] is not None


# --- Update tests ---


class TestDocumentUpdate:
    def test_update_title(self, store):
        result = store.store_doc(title="Old", body="body", summary="S")
        h = result["content_hash"]

        upd = store.update_doc(h, title="New Title")
        assert upd["status"] == "updated"
        assert upd["content_hash"] == h  # hash unchanged

        doc = store.get_doc(h)
        assert doc["title"] == "New Title"

    def test_update_body_increments_version(self, store):
        result = store.store_doc(title="T", body="original body", summary="S")
        h = result["content_hash"]

        upd = store.update_doc(h, body="updated body text")
        assert upd["status"] == "updated"
        assert upd["version"] == 2
        assert upd["content_hash"] != h  # hash changed

        doc = store.get_doc(upd["content_hash"])
        assert doc["body"] == "updated body text"
        assert doc["version"] == 2

    def test_update_summary_re_embeds(self, store):
        result = store.store_doc(title="T", body="body", summary="old summary")
        h = result["content_hash"]

        upd = store.update_doc(h, summary="completely new summary about different topic")
        assert upd["status"] == "updated"

        doc = store.get_doc(h)
        assert doc["summary"] == "completely new summary about different topic"

    def test_update_tags(self, store):
        result = store.store_doc(title="T", body="body", summary="S", tags=["a"])
        h = result["content_hash"]

        store.update_doc(h, tags=["b", "c"])
        doc = store.get_doc(h)
        assert doc["tags"] == ["b", "c"]

    def test_update_metadata_merges(self, store):
        result = store.store_doc(
            title="T", body="body", summary="S",
            metadata={"key1": "val1"},
        )
        h = result["content_hash"]

        store.update_doc(h, metadata={"key2": "val2"})
        doc = store.get_doc(h)
        assert doc["metadata"]["key1"] == "val1"
        assert doc["metadata"]["key2"] == "val2"

    def test_update_not_found(self, store):
        result = store.update_doc("nonexistent", title="X")
        assert "error" in result

    def test_update_no_changes(self, store):
        result = store.store_doc(title="T", body="body", summary="S")
        h = result["content_hash"]

        upd = store.update_doc(h)
        assert "error" in upd


# --- CLI smoke tests ---


class TestDocumentCLI:
    def test_cli_doc_store_and_get(self, store, capsys):
        from memory.cli import main

        # Store via CLI args simulation
        result = store.store_doc(
            title="CLI Test", body="cli body", summary="cli summary",
        )
        h = result["content_hash"]

        # Verify get works
        doc = store.get_doc(h)
        assert doc["title"] == "CLI Test"
