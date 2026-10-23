"""Neo4j metadata retrieval; never a store of patient facts or query history."""

import asyncio
import os
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from .index import LEXICAL_STOPWORDS
from .terminology import normalize

SCHEMA_QUERIES = (
    ("CREATE CONSTRAINT cco_metadata_key IF NOT EXISTS "
     "FOR (node:CcoMetadata) REQUIRE node.key IS UNIQUE"),
    ("CREATE FULLTEXT INDEX cco_metadata_text IF NOT EXISTS "
     "FOR (node:CcoMetadata) ON EACH [node.label, node.description]"),
)


class TerminologyGraphUnavailable(RuntimeError):
    pass


class Neo4jTerminologyGraph:
    """Trusted glossary graph, scoped by deterministic catalog key.

    Legacy disease maps are never queried. Catalog switches are process-local,
    so initialization cannot expose partially imported releases.
    """

    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "local-selective-password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver: AsyncDriver | None = None
        self.available = False
        self.active_release: str | None = None
        self.last_error: str | None = None
        self.metadata_count = 0

    def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=2,
                connection_acquisition_timeout=3,
                max_transaction_retry_time=1,
            )
        return self._driver

    async def initialize(self, catalog: dict[str, Any]) -> dict[str, Any]:
        """Initialize within eight seconds; local search can work without Neo4j."""
        self.available = False
        self.active_release = None
        try:
            async with asyncio.timeout(8):
                rows = self._catalog_rows(catalog)
                driver = self._get_driver()
                await driver.verify_connectivity()
                for query in SCHEMA_QUERIES:
                    await driver.execute_query(query, database_=self.database)
                query = """
                UNWIND $rows AS row
                MERGE (node:CcoMetadata {key: row.key})
                SET node.catalog = $catalog, node.id = row.id,
                    node.kind = row.kind, node.label = row.label,
                    node.description = row.description, node.family = row.family,
                    node.normalizedId = row.normalizedId,
                    node.normalizedLabel = row.normalizedLabel
                FOREACH (_ IN CASE WHEN row.kind = 'definition' THEN [1] ELSE [] END |
                    SET node:Definition)
                FOREACH (_ IN CASE WHEN row.kind = 'family' THEN [1] ELSE [] END |
                    SET node:Family)
                FOREACH (_ IN CASE WHEN row.kind = 'rule' THEN [1] ELSE [] END |
                    SET node:Rule)
                """
                await driver.execute_query(
                    query, rows=rows, catalog=catalog["key"], database_=self.database
                )
                await driver.execute_query(
                    """
                    MATCH (node:CcoMetadata {catalog: $catalog})
                    WHERE node.family IS NOT NULL
                    MATCH (family:CcoMetadata:Family {catalog: $catalog, id: node.family})
                    MERGE (node)-[:IN_FAMILY]->(family)
                    """,
                    catalog=catalog["key"], database_=self.database,
                )
                self.available = True
                self.active_release = catalog["key"]
                self.metadata_count = len(rows)
                self.last_error = None
        except (DriverError, Neo4jError, OSError, ValueError, TypeError, TimeoutError) as exc:
            self.last_error = type(exc).__name__
            self.metadata_count = 0
        return self.status()

    @staticmethod
    def _catalog_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(catalog, dict) or not isinstance(catalog.get("key"), str):
            raise ValueError("catalog requires a key and trusted metadata groups")
        rows = []
        for group, kind in (("families", "family"), ("definitions", "definition"), ("rules", "rule")):
            if not isinstance(catalog.get(group), list):
                raise ValueError("catalog metadata groups must be arrays")
            for item in catalog[group]:
                if not isinstance(item, dict) or not all(
                    isinstance(item.get(key), str) for key in ("id", "label", "description")
                ):
                    raise ValueError("invalid catalog metadata")
                # Explicit allowlist prevents extra patient fields reaching Neo4j.
                rows.append({
                    "key": f"{catalog['key']}|{kind}|{item['id']}",
                    "id": item["id"], "kind": kind,
                    "label": item["label"], "description": item["description"],
                    "family": item.get("family") if isinstance(item.get("family"), str) else None,
                    "normalizedId": normalize(item["id"]),
                    "normalizedLabel": normalize(item["label"]),
                })
        return rows

    async def resolve(self, question: str, limit: int = 12) -> list[dict[str, Any]]:
        if not self.available:
            raise TerminologyGraphUnavailable("Neo4j metadata graph is unavailable")
        # Normalization strips Lucene operators. Query parameters are transient;
        # the application never writes them to graph properties.
        terms = list(dict.fromkeys(
            term for term in normalize(question).split() if term not in LEXICAL_STOPWORDS
        ))
        if not terms:
            return []
        lucene_query = " OR ".join('"' + term + '"' for term in terms)
        query = """
        CALL db.index.fulltext.queryNodes('cco_metadata_text', $search)
        YIELD node, score
        WHERE node.catalog = $catalog
        RETURN node.id AS id, node.kind AS kind, node.label AS label,
               node.description AS description, node.family AS family,
               score AS score, 'metadata_fulltext' AS channel
        UNION
        MATCH (node:CcoMetadata {catalog: $catalog})
        WHERE node.normalizedId IN $terms OR node.normalizedLabel = $normalized
        RETURN node.id AS id, node.kind AS kind, node.label AS label,
               node.description AS description, node.family AS family,
               100.0 AS score, 'metadata_exact' AS channel
        """
        try:
            async with asyncio.timeout(4):
                records, _, _ = await self._get_driver().execute_query(
                    query, search=lucene_query, catalog=self.active_release,
                    terms=terms, normalized=normalize(question), database_=self.database,
                )
        except (DriverError, Neo4jError, OSError, TimeoutError) as exc:
            self.available = False
            self.last_error = type(exc).__name__
            raise TerminologyGraphUnavailable("Neo4j metadata query unavailable") from exc
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            row = dict(record)
            key = (row["kind"], row["id"])
            if key not in unique or row["score"] > unique[key]["score"]:
                unique[key] = row
        return sorted(unique.values(), key=lambda item: (-item["score"], item["id"]))[
            :max(1, min(limit, 100))
        ]

    def status(self) -> dict[str, Any]:
        return {
            "status": "available" if self.available else "unavailable",
            "activeRelease": self.active_release,
            "metadataCount": self.metadata_count,
            "lastError": self.last_error,
            "phiPolicy": "trusted-glossary-metadata-only",
        }

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        self.available = False
