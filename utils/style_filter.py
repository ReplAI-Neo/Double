#!/usr/bin/env python3
"""
Extract stylistic traits from conversational data and select high-alignment snippets.

Workflow:
1. Flatten assistant messages from the merged/filtered conversation dataset.
2. Chunk messages into buckets and ask an LLM to describe the dominant stylistic traits.
3. Turn each trait into a natural-language sentence, embed those trait strings, and embed
   the individual assistant messages.
4. Score each assistant message by similarity to the weighted trait vectors.
5. Keep the top-N% assistant messages and export short conversation snippets that
   preserve local context around those messages.

The output file matches the conversation schema expected by downstream scripts,
but only includes snippets that end in a high-scoring assistant turn.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

try:
    from openai import OpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert linguist who analyses text messages to discover the sender's stylistic habits. "
    "Focus on lexical quirks (token elongation, emoji use, repeated patterns), tone markers, and response cadence. "
    "Ignore content topics unless they are tied to style (e.g., always opening with 'dude'). "
    "Return a concise JSON report."
)


@dataclass
class MessageRecord:
    """Minimal representation of a single assistant message."""

    conversation_index: int
    message_index: int
    role: str
    content: str
    timestamp: Optional[str]


@dataclass
class TraitRecord:
    """Structured description of a style trait returned by the LLM and cached locally."""

    name: str
    description: str
    markers: List[str]
    examples: List[int]
    confidence: float
    polarity: str
    bucket_index: int

    def embedding_text(self) -> str:
        """Return the natural-language sentence that we feed into the embedding model."""
        markers = ", ".join(self.markers) if self.markers else ""
        markers_clause = f" Markers: {markers}." if markers else ""
        return f"{self.name}. {self.description}{markers_clause}"

    def to_cache_dict(self) -> Dict[str, Any]:
        """Convert to the JSON payload saved on disk for cache reuse."""
        return {
            "name": self.name,
            "description": self.description,
            "markers": self.markers,
            "examples": self.examples,
            "confidence": self.confidence,
            "polarity": self.polarity,
        }


class LLMStyleExtractor:
    """Call an LLM to summarise stylistic traits for a bucket of messages."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        temperature: float = 0.2,
        max_traits: int = 6,
        dry_run: bool = False,
    ):
        if OpenAI is None and not dry_run:
            raise RuntimeError(
                "openai package is required for live LLM calls. Install `openai>=1.0` or run with --dry-run."
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not dry_run:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for LLM extraction.")

        self.model_name = model_name
        self.temperature = temperature
        self.max_traits = max_traits
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.client = OpenAI(api_key=api_key) if (OpenAI is not None and not dry_run) else None

    def _bucket_cache_path(self, bucket_index: int) -> Path:
        """Return the cache file path for a given bucket index."""
        return self.cache_dir / f"bucket_{bucket_index:04d}.json"

    def analyse_bucket(self, bucket_index: int, messages: Sequence[MessageRecord]) -> List[TraitRecord]:
        """Return stylistic traits for a bucket, loading from cache or querying the LLM."""
        cache_path = self._bucket_cache_path(bucket_index)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            traits_payload = cached.get("traits", [])
            cached_records: List[TraitRecord] = []
            for trait in traits_payload:
                # Support legacy cache entries that stored a combined text field.
                if "name" not in trait and "text" in trait:
                    name, description = trait["text"].split(":", 1) if ":" in trait["text"] else ("Style trait", trait["text"])
                    cached_records.append(
                        TraitRecord(
                            name=name.strip(),
                            description=description.strip(),
                            markers=[],
                            examples=[],
                            confidence=float(trait.get("confidence", 0.5)),
                            polarity=trait.get("polarity", "positive"),
                            bucket_index=bucket_index,
                        )
                    )
                    continue
                polarity = str(trait.get("polarity", "positive")).lower()
                if polarity not in {"positive", "negative"}:
                    polarity = "positive"
                cached_records.append(
                    TraitRecord(
                        name=str(trait.get("name", "Style trait")).strip(),
                        description=str(trait.get("description", "")).strip(),
                        markers=[
                            str(marker)
                            for marker in trait.get("markers", [])
                            if isinstance(marker, (str, int, float))
                        ],
                        examples=[
                            int(idx)
                            for idx in trait.get("examples", [])
                            if isinstance(idx, (int, str)) and str(idx).isdigit()
                        ],
                        confidence=float(trait.get("confidence", 0.5)),
                        polarity=polarity,
                        bucket_index=bucket_index,
                    )
                )
            return cached_records

        if self.dry_run:
            raise RuntimeError(
                f"No cached traits found for bucket {bucket_index} and --dry-run requested."
            )

        sampled_messages = [
            f"{i+1}. {msg.content.strip()}" for i, msg in enumerate(messages)
        ]
        prompt = (
            "Analyse the following text messages that I wrote. "
            "List the most salient stylistic traits, focusing on surface signals. "
            "Return JSON with schema:\n"
            "{\n"
            '  "traits": [\n'
            '    {\n'
            '      "name": string,\n'
            '      "description": string,\n'
            '      "markers": [string],\n'
            '      "examples": [integer],\n'
            '      "confidence": float (0-1),\n'
            '      "polarity": "positive" | "negative"\n'
            "    }\n"
            "  ],\n"
            '  "summary": string\n'
            "}\n"
            f"Return no more than {self.max_traits} traits."
        )
        user_content = "Messages:\n" + "\n".join(sampled_messages)

        assert self.client is not None  # for type checking
        response = self.client.responses.create(
            model=self.model_name,
            temperature=self.temperature,
            input=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        )
        output_text = response.output_text or ""
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to parse JSON from LLM response for bucket {bucket_index}: {output_text}"
            ) from exc

        traits: List[Dict[str, Any]] = parsed.get("traits") or []
        trait_records: List[TraitRecord] = []
        for trait in traits:
            confidence = float(trait.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            polarity = str(trait.get("polarity", "positive")).lower()
            if polarity not in {"positive", "negative"}:
                polarity = "positive"
            markers_raw = trait.get("markers", [])
            markers = [str(marker) for marker in markers_raw if isinstance(marker, (str, int, float))]
            examples_raw = trait.get("examples", [])
            examples: List[int] = []
            for idx in examples_raw:
                try:
                    examples.append(int(idx))
                except (TypeError, ValueError):
                    continue
            trait_records.append(
                TraitRecord(
                    name=str(trait.get("name", "Style trait")).strip(),
                    description=str(trait.get("description", "")).strip(),
                    markers=markers,
                    examples=examples,
                    confidence=confidence,
                    polarity=polarity,
                    bucket_index=bucket_index,
                )
            )

        cache_path.write_text(
            json.dumps(
                {
                    "bucket_index": bucket_index,
                    "messages": [msg.content for msg in messages],
                    "traits": [trait.to_cache_dict() for trait in trait_records],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return trait_records


class Embedder:
    """Shared interface for embedding text with OpenAI or Sentence-Transformers."""

    def __init__(self, provider: str, model_name: str, batch_size: int = 256):
        """Initialise the embedding backend."""
        self.provider = provider
        self.model_name = model_name
        self.batch_size = batch_size

        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package is required for embedding provider 'openai'.")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI embeddings.")
            self.client = OpenAI(api_key=api_key)
        elif provider == "sentence-transformers":
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers package is required for embedding provider 'sentence-transformers'."
                )
            self.model = SentenceTransformer(model_name)
        else:  # pragma: no cover - validated by argparse
            raise ValueError(f"Unsupported embedding provider '{provider}'.")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed input texts and return a L2-normalised numpy array."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.provider == "openai":
            vectors: List[np.ndarray] = []
            for batch in batched(texts, self.batch_size):
                response = self.client.embeddings.create(model=self.model_name, input=list(batch))
                for item in response.data:
                    vectors.append(np.asarray(item.embedding, dtype=np.float32))
            array = np.vstack(vectors)
        else:  # sentence-transformers
            array = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            array = array.astype(np.float32)

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array = array / norms
        return array


def batched(iterable: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Yield successive chunks of size `batch_size` from `iterable`."""
    total = len(iterable)
    for start in range(0, total, batch_size):
        yield iterable[start : start + batch_size]


def load_conversations(path: Path) -> List[Dict[str, Any]]:
    """Load conversation JSON produced by earlier pipeline stages."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_assistant_messages(conversations: List[Dict[str, Any]]) -> List[MessageRecord]:
    """Collect assistant messages across conversations as MessageRecord instances."""
    flattened: List[MessageRecord] = []
    for conv_idx, conv in enumerate(conversations):
        openai_messages: List[Dict[str, Any]] = conv.get("openai_messages", [])
        full_metadata: List[Dict[str, Any]] = conv.get("full_metadata_messages", [])

        full_index_map: Dict[int, int] = {}
        openai_ptr = 0
        for full_idx, full_msg in enumerate(full_metadata):
            content = full_msg.get("content")
            if content:
                full_index_map[openai_ptr] = full_idx
                openai_ptr += 1
                if openai_ptr >= len(openai_messages):
                    break

        for msg_idx, message in enumerate(openai_messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "").strip()
            if not content:
                continue
            timestamp = None
            meta_idx: Optional[int] = full_index_map.get(msg_idx)
            if meta_idx is not None and meta_idx < len(full_metadata):
                timestamp = full_metadata[meta_idx].get("timestamp")
            flattened.append(
                MessageRecord(
                    conversation_index=conv_idx,
                    message_index=msg_idx,
                    role="assistant",
                    content=content,
                    timestamp=timestamp,
                )
            )
    return flattened


def select_top_messages(
    scores: List[Tuple[MessageRecord, float]],
    top_percent: float,
) -> List[Tuple[MessageRecord, float]]:
    """Return the highest-scoring messages, keeping `top_percent` of the inputs."""
    if not scores:
        return []

    sorted_scores = sorted(scores, key=lambda item: item[1], reverse=True)
    cutoff = max(1, int(math.ceil(len(sorted_scores) * (top_percent / 100.0))))
    return sorted_scores[:cutoff]


def build_snippet(
    conversation: Dict[str, Any],
    message_index: int,
    score: float,
    context_before: int,
) -> Dict[str, Any]:
    """Assemble a snippet that preserves context and attaches the style score metadata."""
    openai_messages = conversation.get("openai_messages", [])
    full_metadata = conversation.get("full_metadata_messages", [])

    start_idx = max(0, message_index - context_before)
    end_idx = message_index + 1  # exclusive
    snippet_openai = openai_messages[start_idx:end_idx]

    snippet_metadata: List[Dict[str, Any]] = []
    timestamps: List[str] = []
    if full_metadata:
        # Build map from openai index -> full metadata index
        mapping: Dict[int, int] = {}
        openai_ptr = 0
        for full_idx, full_msg in enumerate(full_metadata):
            if full_msg.get("content"):
                mapping[openai_ptr] = full_idx
                openai_ptr += 1
                if openai_ptr >= len(openai_messages):
                    break

        meta_indices = [
            mapping[idx]
            for idx in range(start_idx, end_idx)
            if idx in mapping and mapping[idx] < len(full_metadata)
        ]
        if meta_indices:
            first_meta = min(meta_indices)
            last_meta = max(meta_indices)
            snippet_metadata = full_metadata[first_meta : last_meta + 1]
            timestamps = [
                ts
                for msg in snippet_metadata
                for ts in [msg.get("timestamp")]
                if isinstance(ts, str) and ts
            ]

    snippet = {
        "openai_messages": snippet_openai,
        "full_metadata_messages": snippet_metadata,
        "source": conversation.get("source"),
        "chat_type": conversation.get("chat_type"),
        "recipients": conversation.get("recipients", []),
        "num_participants": conversation.get("num_participants"),
        "total_messages": len(snippet_openai),
        "style_score": score,
        "parent_conversation_index": conversation.get("conversation_index"),
        "parent_message_index": message_index,
    }

    parent_first = conversation.get("first_message_timestamp")
    parent_last = conversation.get("last_message_timestamp")
    snippet["first_message_timestamp"] = timestamps[0] if timestamps else parent_first
    snippet["last_message_timestamp"] = timestamps[-1] if timestamps else parent_last

    return snippet


def main() -> None:
    """CLI entry point for running the stylistic filtering stage."""
    parser = argparse.ArgumentParser(
        description="Extract stylistic traits and keep the most stylistically aligned messages."
    )
    parser.add_argument("input_file", type=str, help="Path to filtered conversations JSON.")
    parser.add_argument("output_file", type=str, help="Path to write stylistically filtered snippets.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/style_cache",
        help="Directory for caching LLM trait outputs.",
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="data/merged/style_annotations.jsonl",
        help="Optional JSONL file with per-message style scores.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "sentence-transformers"],
        default="openai",
        help="Backend to embed traits and messages.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model identifier.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to call for trait extraction.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=1000,
        help="Number of assistant messages per LLM bucket.",
    )
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=120,
        help="Number of messages sampled from each bucket for the LLM prompt.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=10.0,
        help="Percentage of assistant messages to keep based on similarity scores.",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=6,
        help="Number of preceding OpenAI-format turns to keep as context in each snippet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--max-traits",
        type=int,
        default=6,
        help="Maximum number of traits to request per bucket.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the LLM; expect trait caches to exist.",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    cache_dir = Path(args.cache_dir)
    annotations_path = Path(args.annotations_file)

    conversations = load_conversations(input_path)
    for idx, conv in enumerate(conversations):
        conv["conversation_index"] = idx  # used to trace parents

    assistant_messages = flatten_assistant_messages(conversations)
    if not assistant_messages:
        print("No assistant messages found; nothing to do.", file=sys.stderr)
        sys.exit(0)

    extractor = LLMStyleExtractor(
        model_name=args.llm_model,
        cache_dir=cache_dir,
        max_traits=args.max_traits,
        dry_run=args.dry_run,
    )

    # Build buckets and extract traits
    bucket_traits: List[TraitRecord] = []
    total_messages = len(assistant_messages)
    buckets = [
        assistant_messages[i : i + args.bucket_size]
        for i in range(0, total_messages, args.bucket_size)
    ]
    for bucket_index, bucket in enumerate(buckets):
        if not bucket:
            continue
        if len(bucket) > args.samples_per_bucket:
            sampled = random.sample(bucket, args.samples_per_bucket)
        else:
            sampled = bucket
        traits = extractor.analyse_bucket(bucket_index, sampled)
        bucket_traits.extend(traits)

    if not bucket_traits:
        raise RuntimeError("No stylistic traits were extracted; check LLM configuration.")

    embedder = Embedder(
        provider=args.embedding_provider,
        model_name=args.embedding_model,
    )

    trait_embeddings = embedder.embed([trait.embedding_text() for trait in bucket_traits])
    trait_weights = np.array(
        [
            trait.confidence if trait.polarity == "positive" else -0.5 * trait.confidence
            for trait in bucket_traits
        ],
        dtype=np.float32,
    )
    weight_norm = np.abs(trait_weights).sum()
    if weight_norm == 0:
        trait_weights = np.ones_like(trait_weights)
        weight_norm = trait_weights.sum()

    assistant_embeddings = embedder.embed([msg.content for msg in assistant_messages])

    scores: List[Tuple[MessageRecord, float]] = []
    trait_matrix = trait_embeddings  # shape: (num_traits, dim)
    for msg, embedding in tqdm(
        zip(assistant_messages, assistant_embeddings),
        total=len(assistant_messages),
        desc="Scoring messages",
    ):
        # Compute cosine similarity to each trait vector
        similarities = trait_matrix @ embedding
        weighted = similarities * trait_weights
        score = float(weighted.sum() / weight_norm)
        scores.append((msg, score))

    top_messages = select_top_messages(scores, args.top_percent)

    snippets: List[Dict[str, Any]] = []
    with annotations_path.open("w", encoding="utf-8") as ann_f:
        for record, score in top_messages:
            conversation = conversations[record.conversation_index]
            snippet = build_snippet(
                conversation=conversation,
                message_index=record.message_index,
                score=score,
                context_before=args.context_before,
            )
            snippets.append(snippet)
            annotation = {
                "conversation_index": record.conversation_index,
                "message_index": record.message_index,
                "score": score,
                "timestamp": record.timestamp,
                "content": record.content,
            }
            ann_f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(snippets, handle, ensure_ascii=False, indent=2)

    print(
        f"Selected {len(top_messages)} stylistically aligned messages "
        f"out of {len(assistant_messages)} ({args.top_percent:.1f}%)."
    )
    print(f"Snippets written to: {output_path}")
    print(f"Annotations written to: {annotations_path}")


if __name__ == "__main__":
    main()


