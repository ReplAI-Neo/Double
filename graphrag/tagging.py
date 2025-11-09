#!/usr/bin/env python3
"""
Utilities for deriving lightweight tags and signals from normalized conversations.

This module enriches the unified conversation schema with additional metadata that
can later be promoted to graph properties or used as weak labels during retrieval.

Key capabilities:

* Conversation level aggregates (turn-taking ratios, participant statistics, etc.)
* Message level annotations (time-of-day buckets, lexical statistics, gaps)
* Simple heuristics for tagging topical signals (emoji usage, question density)
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytz

from utils.filter import (  # noqa: F401 - re-exported helpers
    calculate_my_avg_message_length,
    calculate_my_turn_proportion,
    get_time_of_day_pacific,
)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


PACIFIC_TZ = pytz.timezone("America/Los_Angeles")


def _parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None

    ts = timestamp.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _seconds_between(prev: Optional[datetime], current: Optional[datetime]) -> Optional[float]:
    if not prev or not current:
        return None
    return (current - prev).total_seconds()


def _hash_message(conversation_id: str, message: Dict[str, Any]) -> str:
    message_id = message.get("message_id")
    payload = "|".join(
        [
            conversation_id,
            message_id or "",
            message.get("author") or "",
            message.get("timestamp") or "",
            (message.get("content") or "").strip(),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class MessageAnnotation:
    message_id: str
    conversation_id: str
    author: Optional[str]
    role: Optional[str]
    timestamp: Optional[str]
    time_of_day: Optional[str]
    word_count: int
    char_count: int
    contains_question: bool
    contains_emoji: bool
    gap_seconds: Optional[float]
    synthetic_id: str


@dataclass
class ConversationAnnotation:
    conversation_id: str
    participants: List[str]
    participant_turn_counts: Dict[str, int]
    assistant_turn_ratio: float
    user_turn_ratio: float
    assistant_avg_word_count: float
    time_of_day_buckets: List[str]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    duration_hours: Optional[float]
    message_count: int
    emoji_density: float
    question_density: float
    dominant_time_bucket: Optional[str]
    # Semantic tags
    formality: Optional[str]  # formal, informal
    information_content: Optional[str]  # informational, conversational
    tone: Optional[str]  # humorous, serious, neutral
    sentiment: Optional[str]  # empathetic, neutral, detached
    friendliness: Optional[str]  # friendly, professional, neutral
    purpose: Optional[str]  # technical_support, debugging, explanation, etc.
    complexity: Optional[str]  # beginner, intermediate, advanced
    resolution: Optional[str]  # resolved, unresolved, partial
    engagement: Optional[str]  # high_back_and_forth, medium_interaction, low_interaction
    content_type: List[str]  # contains_code, contains_error, contains_urls, text_only
    domain: Optional[str]  # programming, data_science, web_dev, etc.
    urgency: Optional[str]  # urgent, time_sensitive, normal
    response_depth: Optional[str]  # detailed, moderate, brief


class TagGenerator:
    """
    Generate derived tags for the unified conversation schema.
    Supports both heuristic-based and ML-based (zero-shot) classification.
    """

    def __init__(
        self, 
        timezone_name: str = "America/Los_Angeles",
        use_zero_shot: bool = True,
        zero_shot_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: int = -1,
    ) -> None:
        self.timezone_name = timezone_name
        self.local_tz = pytz.timezone(timezone_name)
        self.use_zero_shot = use_zero_shot and TRANSFORMERS_AVAILABLE
        self.classifier = None
        
        if self.use_zero_shot:
            print(f"Loading zero-shot classifier '{zero_shot_model}'...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=zero_shot_model,
                device=device,
            )
            print(f"✓ Zero-shot classifier loaded")
        else:
            if use_zero_shot and not TRANSFORMERS_AVAILABLE:
                print("⚠️  transformers not available, falling back to heuristics")
            print("Using heuristic-based classification")

    def annotate_conversation(
        self, conversation: Dict[str, Any], conversation_id: Optional[str] = None
    ) -> Tuple[ConversationAnnotation, List[MessageAnnotation]]:
        full_messages = conversation.get("full_metadata_messages", [])
        openai_messages = conversation.get("openai_messages", [])

        convo_id = conversation_id or self._build_conversation_id(conversation)
        participants = conversation.get("recipients", []) or []
        participants = list(dict.fromkeys(participants))

        role_counter = Counter(msg.get("role") for msg in openai_messages if msg.get("role"))
        assistant_turns = role_counter.get("assistant", 0)
        user_turns = role_counter.get("user", 0)

        assistant_avg_words = calculate_my_avg_message_length(openai_messages)
        assistant_turn_ratio = calculate_my_turn_proportion(openai_messages)
        user_turn_ratio = 100.0 - assistant_turn_ratio

        toc_counter: Counter[str] = Counter()
        emoji_count = 0
        question_count = 0
        message_annotations: List[MessageAnnotation] = []

        prev_timestamp: Optional[datetime] = None

        for msg in full_messages:
            timestamp = msg.get("timestamp")
            dt = _parse_timestamp(timestamp)
            localized_bucket: Optional[str] = None
            if timestamp:
                localized_bucket = get_time_of_day_pacific(timestamp)
                toc_counter[localized_bucket] += 1

            content = msg.get("content") or ""
            message_emoji_count = _count_emojis(content)
            emoji_count += message_emoji_count
            question = "?" in content
            if question:
                question_count += 1

            gap_seconds = _seconds_between(prev_timestamp, dt)
            prev_timestamp = dt

            word_count = len(content.split())
            char_count = len(content)

            synthetic_id = _hash_message(convo_id, msg)

            message_annotations.append(
                MessageAnnotation(
                    message_id=msg.get("message_id") or synthetic_id,
                    synthetic_id=synthetic_id,
                    conversation_id=convo_id,
                    author=msg.get("author"),
                    role=self._infer_role(msg, conversation),
                    timestamp=timestamp,
                    time_of_day=localized_bucket,
                    word_count=word_count,
                    char_count=char_count,
                    contains_question=question,
                    contains_emoji=bool(message_emoji_count),
                    gap_seconds=gap_seconds,
                )
            )

        first_ts = conversation.get("first_message_timestamp")
        last_ts = conversation.get("last_message_timestamp")
        duration_hours = None
        start_dt = _parse_timestamp(first_ts)
        end_dt = _parse_timestamp(last_ts)
        if start_dt and end_dt:
            duration_hours = (end_dt - start_dt).total_seconds() / 3600.0

        total_messages = len(full_messages) if full_messages else 0
        emoji_density = (emoji_count / total_messages) if total_messages else 0.0
        question_density = (question_count / total_messages) if total_messages else 0.0

        # Apply semantic classifiers
        formality = self._classify_formality(full_messages)
        information_content = self._classify_information_content(full_messages, question_density)
        tone = self._classify_tone(full_messages, emoji_density)
        sentiment = self._classify_sentiment(full_messages)
        friendliness = self._classify_friendliness(full_messages)
        purpose = self._classify_purpose(full_messages, openai_messages)
        complexity = self._classify_complexity(full_messages, assistant_avg_words)
        resolution = self._classify_resolution(full_messages, openai_messages)
        engagement = self._classify_engagement(total_messages, assistant_turn_ratio, duration_hours)
        content_type = self._classify_content_type(full_messages)
        domain = self._classify_domain(full_messages)
        urgency = self._classify_urgency(full_messages)
        response_depth = self._classify_response_depth(assistant_avg_words, total_messages)

        annotation = ConversationAnnotation(
            conversation_id=convo_id,
            participants=participants,
            participant_turn_counts=self._participant_turn_counts(full_messages),
            assistant_turn_ratio=assistant_turn_ratio,
            user_turn_ratio=user_turn_ratio,
            assistant_avg_word_count=assistant_avg_words,
            time_of_day_buckets=sorted(toc_counter.keys()),
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            duration_hours=duration_hours,
            message_count=total_messages,
            emoji_density=emoji_density,
            question_density=question_density,
            dominant_time_bucket=toc_counter.most_common(1)[0][0] if toc_counter else None,
            # Semantic tags
            formality=formality,
            information_content=information_content,
            tone=tone,
            sentiment=sentiment,
            friendliness=friendliness,
            purpose=purpose,
            complexity=complexity,
            resolution=resolution,
            engagement=engagement,
            content_type=content_type,
            domain=domain,
            urgency=urgency,
            response_depth=response_depth,
        )

        return annotation, message_annotations

    def enrich_conversation(
        self, conversation: Dict[str, Any], conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich a single conversation dictionary with derived metadata inplace.
        """
        conv_annotation, message_annotations = self.annotate_conversation(conversation, conversation_id)

        conversation["derived"] = {
            "conversation_id": conv_annotation.conversation_id,
            "participants": conv_annotation.participants,
            "participant_turn_counts": conv_annotation.participant_turn_counts,
            "assistant_turn_ratio": conv_annotation.assistant_turn_ratio,
            "user_turn_ratio": conv_annotation.user_turn_ratio,
            "assistant_avg_word_count": conv_annotation.assistant_avg_word_count,
            "time_of_day_buckets": conv_annotation.time_of_day_buckets,
            "dominant_time_bucket": conv_annotation.dominant_time_bucket,
            "duration_hours": conv_annotation.duration_hours,
            "message_count": conv_annotation.message_count,
            "emoji_density": conv_annotation.emoji_density,
            "question_density": conv_annotation.question_density,
            # Semantic tags
            "formality": conv_annotation.formality,
            "information_content": conv_annotation.information_content,
            "tone": conv_annotation.tone,
            "sentiment": conv_annotation.sentiment,
            "friendliness": conv_annotation.friendliness,
            "purpose": conv_annotation.purpose,
            "complexity": conv_annotation.complexity,
            "resolution": conv_annotation.resolution,
            "engagement": conv_annotation.engagement,
            "content_type": conv_annotation.content_type,
            "domain": conv_annotation.domain,
            "urgency": conv_annotation.urgency,
            "response_depth": conv_annotation.response_depth,
        }

        enriched_messages: List[Dict[str, Any]] = []
        for msg, annotation in zip(conversation.get("full_metadata_messages", []), message_annotations):
            enriched = dict(msg)
            enriched["derived"] = {
                "synthetic_id": annotation.synthetic_id,
                "role": annotation.role,
                "time_of_day": annotation.time_of_day,
                "word_count": annotation.word_count,
                "char_count": annotation.char_count,
                "contains_question": annotation.contains_question,
                "contains_emoji": annotation.contains_emoji,
                "gap_seconds": annotation.gap_seconds,
            }
            enriched_messages.append(enriched)

        conversation["full_metadata_messages"] = enriched_messages

        return conversation

    def enrich_corpus(
        self, conversations: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        enriched = []
        seen_ids = set()
        for convo in conversations:
            annotation, _ = self.annotate_conversation(convo)
            convo_id = annotation.conversation_id
            if convo_id in seen_ids:
                suffix = hashlib.sha256(f"{convo_id}|{len(seen_ids)}".encode("utf-8")).hexdigest()[:8]
                convo_id = f"{convo_id}-{suffix}"
            seen_ids.add(convo_id)
            enriched.append(self.enrich_conversation(convo, conversation_id=convo_id))
        return enriched

    def _build_conversation_id(self, conversation: Dict[str, Any]) -> str:
        recipients = "|".join(sorted(conversation.get("recipients") or []))
        first_ts = conversation.get("first_message_timestamp") or ""
        last_ts = conversation.get("last_message_timestamp") or ""
        source = conversation.get("source") or ""
        raw = "|".join([source, recipients, first_ts, last_ts])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _participant_turn_counts(messages: List[Dict[str, Any]]) -> Dict[str, int]:
        counter: Dict[str, int] = defaultdict(int)
        for msg in messages:
            author = msg.get("author")
            if author:
                counter[author] += 1
        return dict(counter)

    @staticmethod
    def _infer_role(message: Dict[str, Any], conversation: Dict[str, Any]) -> Optional[str]:
        author = message.get("author")
        if not author:
            return None

        recipients = conversation.get("recipients") or []
        if author in recipients:
            return "user"
        return "assistant"

    # Helper for zero-shot classification ------------------------------------
    def _zero_shot_classify(
        self, 
        text: str, 
        candidate_labels: List[str],
        max_length: int = 512,
    ) -> str:
        """
        Use zero-shot classification to classify text into one of the candidate labels.
        Returns the label with highest score.
        """
        if not self.classifier:
            return candidate_labels[0]  # Fallback to first label
        
        # Truncate text to avoid token limits
        if len(text) > max_length * 4:  # Rough char estimate
            text = text[:max_length * 4]
        
        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False,
        )
        return result['labels'][0]  # Return highest scoring label
    
    # Semantic tag classifiers -----------------------------------------------
    def _classify_formality(self, messages: List[Dict[str, Any]]) -> str:
        """Classify conversation formality."""
        if not messages:
            return "neutral"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["formal", "informal", "neutral"],
            )
        
        # Fallback to heuristics
        informal_markers = ["lol", "lmao", "omg", "btw", "idk", "nvm", "gonna", "wanna", "kinda"]
        formal_markers = ["therefore", "furthermore", "regarding", "pursuant", "hereby", "shall"]
        
        text_lower = text.lower()
        contractions = ["'ll", "'re", "'ve", "'d", "n't", "'m", "can't", "won't"]
        contraction_count = sum(text_lower.count(c) for c in contractions)
        
        informal_score = sum(text_lower.count(marker) for marker in informal_markers) + (contraction_count * 0.5)
        formal_score = sum(text_lower.count(marker) for marker in formal_markers)
        
        if informal_score > formal_score * 2:
            return "informal"
        elif formal_score > informal_score:
            return "formal"
        return "neutral"

    def _classify_information_content(self, messages: List[Dict[str, Any]], question_density: float) -> str:
        """Classify whether conversation is informational or conversational."""
        if not messages:
            return "conversational"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["informational", "conversational"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        info_keywords = ["how", "what", "why", "explain", "define", "tutorial", "learn"]
        conv_keywords = ["thanks", "appreciate", "cool", "lol", "yeah", "ok"]
        
        info_score = sum(text_lower.count(keyword) for keyword in info_keywords)
        conv_score = sum(text_lower.count(keyword) for keyword in conv_keywords)
        
        if question_density > 0.3 or info_score > conv_score * 1.5:
            return "informational"
        return "conversational"

    def _classify_tone(self, messages: List[Dict[str, Any]], emoji_density: float) -> str:
        """Classify conversational tone."""
        if not messages:
            return "neutral"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["humorous", "serious", "neutral"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        humor_markers = ["lol", "lmao", "haha", "rofl", "funny", "hilarious"]
        serious_markers = ["critical", "important", "urgent", "serious", "concern", "issue"]
        
        humor_score = sum(text_lower.count(marker) for marker in humor_markers) + (emoji_density * 5)
        serious_score = sum(text_lower.count(marker) for marker in serious_markers)
        
        if humor_score > 2:
            return "humorous"
        elif serious_score > humor_score * 2:
            return "serious"
        return "neutral"

    def _classify_sentiment(self, messages: List[Dict[str, Any]]) -> str:
        """Classify emotional sentiment of conversation."""
        if not messages:
            return "neutral"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["empathetic", "neutral", "detached"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        empathetic_markers = ["understand", "feel", "sorry", "appreciate", "help", "support"]
        detached_markers = ["incorrect", "wrong", "error", "failed", "invalid"]
        
        emp_score = sum(text_lower.count(marker) for marker in empathetic_markers)
        det_score = sum(text_lower.count(marker) for marker in detached_markers)
        
        if emp_score > 3:
            return "empathetic"
        elif det_score > emp_score * 2:
            return "detached"
        return "neutral"

    def _classify_friendliness(self, messages: List[Dict[str, Any]]) -> str:
        """Classify friendliness level."""
        if not messages:
            return "neutral"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["friendly", "professional", "neutral"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        friendly_markers = ["thanks", "appreciate", "great", "awesome", "nice"]
        professional_markers = ["regarding", "kindly", "respectively", "accordingly"]
        
        friendly_score = sum(text_lower.count(marker) for marker in friendly_markers)
        professional_score = sum(text_lower.count(marker) for marker in professional_markers)
        
        if friendly_score > 5:
            return "friendly"
        elif professional_score > friendly_score:
            return "professional"
        return "neutral"

    def _classify_purpose(self, messages: List[Dict[str, Any]], openai_messages: List[Dict[str, Any]]) -> str:
        """Classify the primary purpose of the conversation."""
        if not messages:
            return "general"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=[
                    "debugging", "learning", "explanation", "code_review",
                    "brainstorming", "technical_support", "general"
                ],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["debug", "error", "bug", "traceback"]):
            return "debugging"
        elif any(keyword in text_lower for keyword in ["how do i", "how to", "tutorial", "learn"]):
            return "learning"
        elif any(keyword in text_lower for keyword in ["explain", "what is", "clarify"]):
            return "explanation"
        elif any(keyword in text_lower for keyword in ["help", "issue", "problem", "not working"]):
            return "technical_support"
        return "general"

    def _classify_complexity(self, messages: List[Dict[str, Any]], avg_word_count: float) -> str:
        """Classify technical complexity level."""
        if not messages:
            return "beginner"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["beginner", "intermediate", "advanced"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        advanced_terms = ["algorithm", "architecture", "asynchronous", "optimization", "scalability"]
        beginner_terms = ["what is", "how do i", "beginner", "basic", "tutorial"]
        
        advanced_score = sum(text_lower.count(term) for term in advanced_terms)
        beginner_score = sum(text_lower.count(term) for term in beginner_terms)
        
        if advanced_score > 2 or avg_word_count > 100:
            return "advanced"
        elif beginner_score > 2 or avg_word_count < 30:
            return "beginner"
        return "intermediate"

    def _classify_resolution(self, messages: List[Dict[str, Any]], openai_messages: List[Dict[str, Any]]) -> str:
        """Classify whether the conversation reached resolution."""
        if not messages or not openai_messages:
            return "unresolved"
        
        # Focus on last few messages
        last_messages = messages[-3:] if len(messages) >= 3 else messages
        text = " ".join(msg.get("content", "") for msg in last_messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["resolved", "unresolved", "partial"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        resolved_markers = ["works", "fixed", "solved", "thanks", "perfect", "got it"]
        unresolved_markers = ["still", "doesn't work", "error", "confused"]
        
        resolved_score = sum(text_lower.count(marker) for marker in resolved_markers)
        unresolved_score = sum(text_lower.count(marker) for marker in unresolved_markers)
        
        if resolved_score > unresolved_score * 2:
            return "resolved"
        elif unresolved_score > resolved_score:
            return "unresolved"
        return "partial"

    @staticmethod
    def _classify_engagement(message_count: int, turn_ratio: float, duration_hours: Optional[float]) -> str:
        """Classify engagement level based on interaction patterns."""
        # High back-and-forth: many messages, balanced turns, reasonable duration
        if message_count > 20 and 30 < turn_ratio < 70:
            return "high_back_and_forth"
        elif message_count > 10 or (duration_hours and duration_hours > 1):
            return "medium_interaction"
        return "low_interaction"

    @staticmethod
    def _classify_content_type(messages: List[Dict[str, Any]]) -> List[str]:
        """Identify types of content present in conversation."""
        if not messages:
            return ["text_only"]
        
        text = " ".join(msg.get("content", "") for msg in messages)
        content_types = []
        
        # Check for code blocks (markdown)
        if "```" in text or "    " in text:  # code blocks or indented code
            content_types.append("contains_code")
        
        # Check for errors
        if any(keyword in text.lower() for keyword in ["error", "exception", "traceback", "failed", "warning"]):
            content_types.append("contains_error")
        
        # Check for URLs
        if "http://" in text or "https://" in text or "www." in text:
            content_types.append("contains_urls")
        
        # Check for file references
        if any(ext in text for ext in [".py", ".js", ".ts", ".java", ".cpp", ".md", ".json", ".yaml", ".txt"]):
            content_types.append("contains_files")
        
        if not content_types:
            content_types.append("text_only")
        
        return content_types

    def _classify_domain(self, messages: List[Dict[str, Any]]) -> str:
        """Classify knowledge domain of conversation."""
        if not messages:
            return "general"
        
        text = " ".join(msg.get("content", "") for msg in messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=[
                    "programming", "data_science", "web_dev", "machine_learning",
                    "devops", "database", "general"
                ],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        domains = {
            "exercise": ["exercise", "workout", "gym", "fitness", "yoga"],
            "diet": ["diet", "nutrition", "food", "meal", "calories"],
            "sleep": ["sleep", "rest", "nap", "bedtime", "waking up"],
            "mood": ["mood", "anxiety", "depression", "stress", "anxiety"],
            "health": ["health", "wellness", "fitness", "yoga", "meditation"],
            "general": ["general", "chat", "conversation", "discussion", "question"],
        }
        
        scores = {domain: sum(text_lower.count(keyword) for keyword in keywords) 
                  for domain, keywords in domains.items()}
        
        max_domain = max(scores, key=scores.get)
        if scores[max_domain] > 2:
            return max_domain
        return "general"

    def _classify_urgency(self, messages: List[Dict[str, Any]]) -> str:
        """Classify urgency level."""
        if not messages:
            return "normal"
        
        # Check first few messages for urgency
        first_messages = messages[:3] if len(messages) >= 3 else messages
        text = " ".join(msg.get("content", "") for msg in first_messages)
        
        if self.use_zero_shot:
            return self._zero_shot_classify(
                text,
                candidate_labels=["urgent", "time_sensitive", "normal"],
            )
        
        # Fallback to heuristics
        text_lower = text.lower()
        urgent_markers = ["urgent", "asap", "immediately", "emergency", "critical"]
        time_markers = ["today", "tonight", "deadline", "soon"]
        
        urgent_score = sum(text_lower.count(marker) for marker in urgent_markers)
        time_score = sum(text_lower.count(marker) for marker in time_markers)
        
        if urgent_score > 0:
            return "urgent"
        elif time_score > 0:
            return "time_sensitive"
        return "normal"

    @staticmethod
    def _classify_response_depth(assistant_avg_word_count: float, message_count: int) -> str:
        """Classify the depth of assistant responses."""
        if assistant_avg_word_count > 150:
            return "detailed"
        elif assistant_avg_word_count > 50:
            return "moderate"
        return "brief"


def _count_emojis(text: str) -> int:
    count = 0
    for ch in text:
        if unicodedata.category(ch) in {"So", "Sk"} and ord(ch) >= 0x1F000:
            count += 1
    return count


__all__ = [
    "TagGenerator",
    "ConversationAnnotation",
    "MessageAnnotation",
    "calculate_my_avg_message_length",
    "calculate_my_turn_proportion",
    "get_time_of_day_pacific",
]


