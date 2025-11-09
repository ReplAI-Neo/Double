from __future__ import annotations

import argparse
import json
import hashlib
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase, Session

from .config import EmbeddingConfig, Neo4jConnectionConfig
from .embedding import SentenceTransformerEmbedder
from .tagging import TagGenerator


def chunked(iterable: Sequence[Dict[str, Any]] | Iterable[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


class Neo4jGraphBuilder:
    """
    Ingest the normalized conversation corpus into a Neo4j graph.
    """

    def __init__(
        self,
        connection: Neo4jConnectionConfig,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        self.connection = connection
        self._driver = GraphDatabase.driver(
            connection.uri,
            auth=(connection.username, connection.password),
            encrypted=connection.encrypted,
        )
        self._database = connection.database
        self.embedder = embedder
        self._batch_counter = 0

    def close(self) -> None:
        self._driver.close()

    # Constraint / index helpers -------------------------------------------------
    def ensure_constraints(self) -> None:
        print("Creating constraints...")
        statements = [
            "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
            "CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS FOR (c:Conversation) REQUIRE c.conversation_id IS UNIQUE",
            "CREATE CONSTRAINT message_id_unique IF NOT EXISTS FOR (m:Message) REQUIRE m.message_id IS UNIQUE",
            "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (t:Tag) REQUIRE t.tag_id IS UNIQUE",
        ]
        with self._session() as session:
            for stmt in statements:
                session.run(stmt)
        print("✓ Constraints created")

    def ensure_vector_index(
        self,
        index_name: str,
        dimension: int,
        property_name: str = "embedding",
        node_label: str = "Message",
        similarity_function: str = "cosine",
    ) -> None:
        print(f"Creating vector index '{index_name}' (dim={dimension})...")
        with self._session() as session:
            exists = session.run(
                """
                SHOW VECTOR INDEXES YIELD name
                WHERE name = $index_name
                RETURN name
                """,
                index_name=index_name,
            ).single()
            if exists:
                print(f"✓ Vector index '{index_name}' already exists")
                return
            session.run(
                """
                CALL db.index.vector.createNodeIndex(
                    $index_name,
                    $label,
                    $property,
                    $dimension,
                    $similarity_function
                )
                """,
                index_name=index_name,
                label=node_label,
                property=property_name,
                dimension=dimension,
                similarity_function=similarity_function,
            )
        print(f"✓ Vector index '{index_name}' created")

    # Ingestion ------------------------------------------------------------------
    def ingest(
        self,
        conversations: Sequence[Dict[str, Any]] | Iterable[Dict[str, Any]],
        batch_size: int = 50,
        tag_generator: Optional[TagGenerator] = None,
        use_zero_shot: bool = False,
        zero_shot_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ) -> None:
        if tag_generator is None:
            device = -1  # CPU
            if self.embedder and hasattr(self.embedder.model, 'device'):
                # Try to use same device as embedder
                embedder_device = str(self.embedder.model.device)
                if 'cuda' in embedder_device:
                    device = int(embedder_device.split(':')[1]) if ':' in embedder_device else 0
            tagger = TagGenerator(use_zero_shot=use_zero_shot, zero_shot_model=zero_shot_model, device=device)
        else:
            tagger = tag_generator
        embedder = self.embedder

        for batch in chunked(conversations, batch_size):
            self._batch_counter += 1
            enriched = tagger.enrich_corpus(batch)
            
            # Count messages in this batch
            message_count = sum(len(c.get("full_metadata_messages", [])) for c in enriched)
            print(f"\nProcessing batch {self._batch_counter} ({len(enriched)} conversations, {message_count} messages)...")
            
            message_embeddings = None
            conversation_embeddings = None
            if embedder:
                print(f"  Embedding {message_count} messages...")
                message_embeddings = self._embed_message_batch(enriched, embedder)
                print(f"  Embedding {len(enriched)} conversation summaries...")
                conversation_embeddings = self._embed_conversation_batch(enriched, embedder)

            with self._session() as session:
                print(f"  Writing to Neo4j...")
                self._write_conversations(session, enriched, conversation_embeddings)
                self._write_participants(session, enriched)
                self._write_tags(session, enriched)
                self._write_messages(session, enriched, message_embeddings)
                self._write_follows(session, enriched)
                self._write_sent_relationships(session, enriched)
            print(f"  ✓ Batch {self._batch_counter} complete")

    # Internal writers -----------------------------------------------------------
    def _write_conversations(
        self,
        session: Session,
        conversations: Sequence[Dict[str, Any]],
        embeddings: Optional[Dict[str, List[float]]],
    ) -> None:
        records = []
        for convo in conversations:
            derived = convo.get("derived", {})
            conversation_id = derived.get("conversation_id") or self._hash_conversation(convo)
            properties = {
                "source": convo.get("source"),
                "chat_type": convo.get("chat_type"),
                "first_timestamp": convo.get("first_message_timestamp"),
                "last_timestamp": convo.get("last_message_timestamp"),
                "num_participants": convo.get("num_participants"),
                "total_messages": len(convo.get("full_metadata_messages", [])),
                "time_of_day_buckets": derived.get("time_of_day_buckets", []),
                "dominant_time_bucket": derived.get("dominant_time_bucket"),
                "assistant_turn_ratio": derived.get("assistant_turn_ratio"),
                "user_turn_ratio": derived.get("user_turn_ratio"),
                "emoji_density": derived.get("emoji_density"),
                "question_density": derived.get("question_density"),
                "participants": derived.get("participants", []),
            }
            if embeddings and conversation_id in embeddings:
                properties["embedding"] = embeddings[conversation_id]
            records.append({"conversation_id": conversation_id, "properties": properties})
        if not records:
            return
        session.run(
            """
            UNWIND $records AS record
            MERGE (c:Conversation {conversation_id: record.conversation_id})
            SET c += record.properties
            """,
            records=records,
        )

    def _write_participants(self, session: Session, conversations: Sequence[Dict[str, Any]]) -> None:
        links = []
        for convo in conversations:
            derived = convo.get("derived", {})
            conversation_id = derived.get("conversation_id") or self._hash_conversation(convo)
            participant_counts = derived.get("participant_turn_counts", {})
            for name, count in participant_counts.items():
                person_id = hashlib.sha256(name.encode("utf-8")).hexdigest()
                links.append(
                    {
                        "person_id": person_id,
                        "display_name": name,
                        "total_messages": int(count),
                        "conversation_id": conversation_id,
                    }
                )
        if not links:
            return
        session.run(
            """
            UNWIND $rows AS row
            MERGE (p:Person {person_id: row.person_id})
            SET p.display_name = row.display_name,
                p.total_messages = coalesce(p.total_messages, 0) + row.total_messages
            WITH p, row
            MATCH (c:Conversation {conversation_id: row.conversation_id})
            MERGE (p)-[r:PARTICIPATED_IN]->(c)
            SET r.message_count = coalesce(r.message_count, 0) + row.total_messages
            """,
            rows=links,
        )

    def _write_tags(self, session: Session, conversations: Sequence[Dict[str, Any]]) -> None:
        tag_rows = []
        for convo in conversations:
            derived = convo.get("derived", {})
            conversation_id = derived.get("conversation_id") or self._hash_conversation(convo)
            
            # Original tags
            source = convo.get("source")
            if source:
                tag_rows.append(
                    {
                        "tag_id": f"source::{source}",
                        "name": source,
                        "category": "source",
                        "conversation_id": conversation_id,
                    }
                )
            chat_type = convo.get("chat_type")
            if chat_type:
                tag_rows.append(
                    {
                        "tag_id": f"chat_type::{chat_type}",
                        "name": chat_type,
                        "category": "chat_type",
                        "conversation_id": conversation_id,
                    }
                )
            for bucket in derived.get("time_of_day_buckets", []):
                tag_rows.append(
                    {
                        "tag_id": f"time_of_day::{bucket}",
                        "name": bucket,
                        "category": "time_of_day",
                        "conversation_id": conversation_id,
                    }
                )
            
            # Semantic tags
            semantic_tag_mappings = [
                ("formality", "formality"),
                ("information_content", "information_content"),
                ("tone", "tone"),
                ("sentiment", "sentiment"),
                ("friendliness", "friendliness"),
                ("purpose", "purpose"),
                ("complexity", "complexity"),
                ("resolution", "resolution"),
                ("engagement", "engagement"),
                ("domain", "domain"),
                ("urgency", "urgency"),
                ("response_depth", "response_depth"),
            ]
            
            for field_name, category in semantic_tag_mappings:
                value = derived.get(field_name)
                if value:
                    tag_rows.append(
                        {
                            "tag_id": f"{category}::{value}",
                            "name": value,
                            "category": category,
                            "conversation_id": conversation_id,
                        }
                    )
            
            # Handle content_type (list of tags)
            content_types = derived.get("content_type", [])
            for content_type in content_types:
                tag_rows.append(
                    {
                        "tag_id": f"content_type::{content_type}",
                        "name": content_type,
                        "category": "content_type",
                        "conversation_id": conversation_id,
                    }
                )
        
        if not tag_rows:
            return
        session.run(
            """
            UNWIND $rows AS row
            MERGE (t:Tag {tag_id: row.tag_id})
            SET t.name = row.name,
                t.category = row.category
            WITH t, row
            MATCH (c:Conversation {conversation_id: row.conversation_id})
            MERGE (c)-[:HAS_TAG]->(t)
            """,
            rows=tag_rows,
        )

    def _write_messages(
        self,
        session: Session,
        conversations: Sequence[Dict[str, Any]],
        embeddings: Optional[Dict[str, List[float]]],
    ) -> None:
        rows = []
        for convo in conversations:
            derived = convo.get("derived", {})
            conversation_id = derived.get("conversation_id") or self._hash_conversation(convo)
            for msg in convo.get("full_metadata_messages", []):
                derived_msg = msg.get("derived", {})
                message_id = msg.get("message_id") or derived_msg.get("synthetic_id")
                if not message_id:
                    continue
                properties = {
                    "content": msg.get("content"),
                    "timestamp": msg.get("timestamp"),
                    "author": msg.get("author"),
                    "role": derived_msg.get("role"),
                    "word_count": derived_msg.get("word_count"),
                    "char_count": derived_msg.get("char_count"),
                    "time_of_day": derived_msg.get("time_of_day"),
                    "contains_question": derived_msg.get("contains_question"),
                    "contains_emoji": derived_msg.get("contains_emoji"),
                    "gap_seconds": derived_msg.get("gap_seconds"),
                    "conversation_id": conversation_id,
                    "synthetic_id": derived_msg.get("synthetic_id"),
                }
                embedding = embeddings.get(message_id) if embeddings else None
                if embedding:
                    properties["embedding"] = embedding
                    if self.embedder:
                        properties["embedding_model"] = self.embedder.config.model_name
                rows.append(
                    {
                        "message_id": message_id,
                        "properties": properties,
                    }
                )
        if not rows:
            return
        session.run(
            """
            UNWIND $rows AS row
            MERGE (m:Message {message_id: row.message_id})
            SET m += row.properties
            WITH m
            MATCH (c:Conversation {conversation_id: m.conversation_id})
            MERGE (c)-[:HAS_MESSAGE]->(m)
            """,
            rows=rows,
        )

    def _write_follows(self, session: Session, conversations: Sequence[Dict[str, Any]]) -> None:
        links = []
        for convo in conversations:
            derived = convo.get("derived", {})
            conversation_id = derived.get("conversation_id") or self._hash_conversation(convo)
            messages = convo.get("full_metadata_messages", [])
            previous_id = None
            for msg in messages:
                derived_msg = msg.get("derived", {})
                message_id = msg.get("message_id") or derived_msg.get("synthetic_id")
                if previous_id and message_id:
                    links.append(
                        {
                            "prev_id": previous_id,
                            "next_id": message_id,
                            "conversation_id": conversation_id,
                        }
                    )
                if message_id:
                    previous_id = message_id
        if not links:
            return
        session.run(
            """
            UNWIND $links AS link
            MATCH (prev:Message {message_id: link.prev_id})
            MATCH (next:Message {message_id: link.next_id})
            MERGE (prev)-[:FOLLOWS]->(next)
            """,
            links=links,
        )

    def _write_sent_relationships(self, session: Session, conversations: Sequence[Dict[str, Any]]) -> None:
        rows = []
        for convo in conversations:
            derived = convo.get("derived", {})
            messages = convo.get("full_metadata_messages", [])
            for msg in messages:
                derived_msg = msg.get("derived", {})
                message_id = msg.get("message_id") or derived_msg.get("synthetic_id")
                author = msg.get("author")
                if not (message_id and author):
                    continue
                person_id = hashlib.sha256(author.encode("utf-8")).hexdigest()
                rows.append({"message_id": message_id, "person_id": person_id})
        if not rows:
            return
        session.run(
            """
            UNWIND $rows AS row
            MATCH (m:Message {message_id: row.message_id})
            MATCH (p:Person {person_id: row.person_id})
            MERGE (p)-[:SENT]->(m)
            """,
            rows=rows,
        )

    # Embedding helpers ---------------------------------------------------------
    def _embed_message_batch(
        self,
        conversations: Sequence[Dict[str, Any]],
        embedder: SentenceTransformerEmbedder,
    ) -> Dict[str, List[float]]:
        texts = []
        ids = []
        for convo in conversations:
            for msg in convo.get("full_metadata_messages", []):
                derived = msg.get("derived", {})
                message_id = msg.get("message_id") or derived.get("synthetic_id")
                if message_id and msg.get("content"):
                    ids.append(message_id)
                    texts.append(msg["content"])
        if not texts:
            return {}
        vectors = embedder.embed(texts)
        return {message_id: vector.tolist() for message_id, vector in zip(ids, vectors)}

    def _embed_conversation_batch(
        self,
        conversations: Sequence[Dict[str, Any]],
        embedder: SentenceTransformerEmbedder,
    ) -> Dict[str, List[float]]:
        summaries = []
        ids = []
        for convo in conversations:
            derived = convo.get("derived", {})
            conversation_id = derived.get("conversation_id") or self._hash_conversation(convo)
            summary = self._extract_conversation_summary(convo)
            if summary:
                ids.append(conversation_id)
                summaries.append(summary)
        if not summaries:
            return {}
        vectors = embedder.embed(summaries)
        return {conversation_id: vector.tolist() for conversation_id, vector in zip(ids, vectors)}

    @staticmethod
    def _extract_conversation_summary(conversation: Dict[str, Any], window: int = 8) -> str:
        openai_messages = conversation.get("openai_messages", [])
        if not openai_messages:
            return ""
        recent = openai_messages[-window:]
        parts = []
        for msg in recent:
            role = msg.get("role")
            content = msg.get("content")
            if not content:
                continue
            prefix = "You" if role == "assistant" else "Them"
            parts.append(f"{prefix}: {content}")
        summary = " ".join(parts)
        return summary[:1500]

    @staticmethod
    def _hash_conversation(conversation: Dict[str, Any]) -> str:
        raw = "|".join(
            [
                conversation.get("source") or "",
                ",".join(sorted(conversation.get("recipients", []) or [])),
                conversation.get("first_message_timestamp") or "",
                conversation.get("last_message_timestamp") or "",
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _session(self) -> Session:
        return self._driver.session(database=self._database) if self._database else self._driver.session()


__all__ = ["Neo4jGraphBuilder"]


def _load_conversations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest annotated conversations into Neo4j")
    parser.add_argument("--input", required=True, type=Path, help="Path to the annotated conversations JSON")
    parser.add_argument("--neo4j-uri", required=True, help="Neo4j bolt URI, e.g. bolt://localhost:7687")
    parser.add_argument("--neo4j-user", required=True, help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--neo4j-database", help="Neo4j database name (optional)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for ingestion")
    parser.add_argument("--ensure-constraints", action="store_true", help="Create required constraints/indexes before ingesting")
    parser.add_argument("--vector-index-name", default="message_embedding_index", help="Name of the vector index for messages")
    parser.add_argument("--embedding-model", help="SentenceTransformer model name for embeddings")
    parser.add_argument("--embedding-device", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    parser.add_argument("--use-zero-shot", action="store_true", help="Use zero-shot classification for semantic tags (slower but more accurate)")
    parser.add_argument("--zero-shot-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", help="Zero-shot model to use")
    parser.add_argument("--no-semantic-tags", action="store_true", help="Disable semantic tagging (only basic tags)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    print(f"Loading conversations from {args.input}...")
    conversations = _load_conversations(args.input)
    print(f"✓ Loaded {len(conversations)} conversations")

    connection = Neo4jConnectionConfig(
        uri=args.neo4j_uri,
        username=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )

    embedder = None
    if args.embedding_model:
        print(f"\nLoading embedding model '{args.embedding_model}'...")
        config = EmbeddingConfig(
            model_name=args.embedding_model,
            device=args.embedding_device,
            normalize_embeddings=not args.no_normalize,
        )
        embedder = SentenceTransformerEmbedder(config)
        print(f"✓ Model loaded (dimension={embedder.dimension}, device={embedder.model.device})")

    builder = Neo4jGraphBuilder(connection=connection, embedder=embedder)
    try:
        if args.ensure_constraints:
            print("\n" + "="*60)
            builder.ensure_constraints()
            if embedder:
                builder.ensure_vector_index(
                    index_name=args.vector_index_name,
                    dimension=embedder.dimension,
                    property_name="embedding",
                )
            print("="*60)
        
        print(f"\nStarting ingestion (batch_size={args.batch_size})...")
        print("="*60)
        
        # Pass zero-shot settings
        use_zero_shot = args.use_zero_shot if not args.no_semantic_tags else False
        builder.ingest(
            conversations, 
            batch_size=args.batch_size,
            use_zero_shot=use_zero_shot,
            zero_shot_model=args.zero_shot_model,
        )
        
        print("\n" + "="*60)
        print(f"✓ Ingestion complete! Processed {builder._batch_counter} batches.")
        print("="*60)
    finally:
        builder.close()


if __name__ == "__main__":
    main()

