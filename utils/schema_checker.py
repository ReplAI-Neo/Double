#!/usr/bin/env python3
"""
Schema checker for conversation data.

Validates JSON files against the CONVERSATION_SCHEMA.md specification.
"""

import json
import sys
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass


class ConversationSchemaChecker:
    """Validates conversation data against the schema defined in CONVERSATION_SCHEMA.md."""
    
    VALID_SOURCES = {"discord", "imessage", "instagram"}
    VALID_CHAT_TYPES = {"direct", "group"}
    VALID_ROLES = {"user", "assistant"}
    
    # ISO 8601 regex pattern (with timezone)
    ISO_8601_PATTERN = re.compile(
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?([+-]\d{2}:\d{2}|Z)$'
    )
    
    def __init__(self, strict: bool = True):
        """
        Initialize the schema checker.
        
        Args:
            strict: If True, perform strict validation (check all constraints).
                   If False, perform basic validation only.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, data: Any) -> Tuple[bool, List[str], List[str]]:
        """
        Validate data against the conversation schema.
        
        Args:
            data: JSON data to validate (should be a list of conversation objects)
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            self._validate_top_level(data)
        except SchemaValidationError as e:
            self.errors.append(str(e))
            return False, self.errors, self.warnings
        
        # If top-level validation passed, validate each conversation
        if isinstance(data, list):
            for idx, conversation in enumerate(data):
                self._validate_conversation(conversation, idx)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_top_level(self, data: Any) -> None:
        """Validate that data is a list."""
        if not isinstance(data, list):
            raise SchemaValidationError(
                f"Top-level structure must be an array, got {type(data).__name__}"
            )
    
    def _validate_conversation(self, conv: Dict[str, Any], index: int) -> None:
        """Validate a single conversation object."""
        conv_errors = []
        
        if not isinstance(conv, dict):
            conv_errors.append(f"Conversation must be an object, got {type(conv).__name__}")
            # Can't continue validation if it's not a dict
            for err in conv_errors:
                self.errors.append(f"Conversation {index}: {err}")
            return
        
        # Required fields
        required_fields = [
            "openai_messages",
            "full_metadata_messages",
            "first_message_timestamp",
            "last_message_timestamp",
            "recipients",
            "num_participants",
            "total_messages",
            "source",
            "chat_type"
        ]
        
        for field in required_fields:
            if field not in conv:
                conv_errors.append(f"Missing required field: {field}")
        
        # Validate openai_messages
        if "openai_messages" in conv:
            if not isinstance(conv["openai_messages"], list):
                conv_errors.append("openai_messages must be an array")
            else:
                for msg_idx, msg in enumerate(conv["openai_messages"]):
                    try:
                        self._validate_openai_message(msg, msg_idx)
                    except SchemaValidationError as e:
                        conv_errors.append(str(e))
        
        # Validate full_metadata_messages
        if "full_metadata_messages" in conv:
            if not isinstance(conv["full_metadata_messages"], list):
                conv_errors.append("full_metadata_messages must be an array")
            else:
                for msg_idx, msg in enumerate(conv["full_metadata_messages"]):
                    try:
                        self._validate_full_metadata_message(msg, msg_idx)
                    except SchemaValidationError as e:
                        conv_errors.append(str(e))
        
        # Validate timestamps
        if "first_message_timestamp" in conv:
            try:
                self._validate_timestamp(conv["first_message_timestamp"], "first_message_timestamp")
            except SchemaValidationError as e:
                conv_errors.append(str(e))
        
        if "last_message_timestamp" in conv:
            try:
                self._validate_timestamp(conv["last_message_timestamp"], "last_message_timestamp")
            except SchemaValidationError as e:
                conv_errors.append(str(e))
        
        # Validate recipients
        if "recipients" in conv:
            if not isinstance(conv["recipients"], list):
                conv_errors.append("recipients must be an array")
            else:
                for idx, recipient in enumerate(conv["recipients"]):
                    if not isinstance(recipient, str):
                        conv_errors.append(f"recipients[{idx}] must be a string")
        
        # Validate num_participants
        if "num_participants" in conv:
            if not isinstance(conv["num_participants"], int):
                conv_errors.append("num_participants must be a number")
            elif conv["num_participants"] < 1:
                conv_errors.append("num_participants must be at least 1")
        
        # Validate total_messages
        if "total_messages" in conv:
            if not isinstance(conv["total_messages"], int):
                conv_errors.append("total_messages must be a number")
            elif conv["total_messages"] < 0:
                conv_errors.append("total_messages must be non-negative")
        
        # Validate source
        if "source" in conv:
            if not isinstance(conv["source"], str):
                conv_errors.append("source must be a string")
            elif conv["source"] not in self.VALID_SOURCES:
                conv_errors.append(
                    f"source must be one of {self.VALID_SOURCES}, got '{conv['source']}'"
                )
        
        # Validate chat_type
        if "chat_type" in conv:
            if not isinstance(conv["chat_type"], str):
                conv_errors.append("chat_type must be a string")
            elif conv["chat_type"] not in self.VALID_CHAT_TYPES:
                conv_errors.append(
                    f"chat_type must be one of {self.VALID_CHAT_TYPES}, got '{conv['chat_type']}'"
                )
        
        # Cross-field validation (only if no critical errors)
        if self.strict and len(conv_errors) == 0:
            # Check that total_messages matches full_metadata_messages length
            if "total_messages" in conv and "full_metadata_messages" in conv:
                if conv["total_messages"] != len(conv["full_metadata_messages"]):
                    self.warnings.append(
                        f"Conversation {index}: total_messages ({conv['total_messages']}) does not match "
                        f"full_metadata_messages length ({len(conv['full_metadata_messages'])})"
                    )
            
            # Check that first_message_timestamp <= last_message_timestamp
            if "first_message_timestamp" in conv and "last_message_timestamp" in conv:
                try:
                    first_ts = datetime.fromisoformat(conv["first_message_timestamp"].replace('Z', '+00:00'))
                    last_ts = datetime.fromisoformat(conv["last_message_timestamp"].replace('Z', '+00:00'))
                    if first_ts > last_ts:
                        self.warnings.append(
                            f"Conversation {index}: first_message_timestamp is after last_message_timestamp"
                        )
                except (ValueError, AttributeError):
                    pass  # Already validated timestamp format
        
        # Add all errors for this conversation
        for err in conv_errors:
            self.errors.append(f"Conversation {index}: {err}")
    
    def _validate_openai_message(self, msg: Dict[str, Any], index: int) -> None:
        """Validate an OpenAIMessage object."""
        if not isinstance(msg, dict):
            raise SchemaValidationError(f"openai_messages[{index}] must be an object")
        
        # Required fields
        if "role" not in msg:
            raise SchemaValidationError(f"openai_messages[{index}] missing required field: role")
        if "content" not in msg:
            raise SchemaValidationError(f"openai_messages[{index}] missing required field: content")
        
        # Validate role
        if not isinstance(msg["role"], str):
            raise SchemaValidationError(f"openai_messages[{index}].role must be a string")
        if msg["role"] not in self.VALID_ROLES:
            raise SchemaValidationError(
                f"openai_messages[{index}].role must be one of {self.VALID_ROLES}, "
                f"got '{msg['role']}'"
            )
        
        # Validate content
        if not isinstance(msg["content"], str):
            raise SchemaValidationError(f"openai_messages[{index}].content must be a string")
        
        # According to schema, empty messages should be filtered out
        if self.strict and not msg["content"].strip():
            self.warnings.append(
                f"openai_messages[{index}] has empty content (should be filtered out)"
            )
    
    def _validate_full_metadata_message(self, msg: Dict[str, Any], index: int) -> None:
        """Validate a FullMetadataMessage object."""
        if not isinstance(msg, dict):
            raise SchemaValidationError(f"full_metadata_messages[{index}] must be an object")
        
        # Required fields
        required_fields = ["message_id", "timestamp", "content", "author"]
        for field in required_fields:
            if field not in msg:
                raise SchemaValidationError(
                    f"full_metadata_messages[{index}] missing required field: {field}"
                )
        
        # Validate message_id
        if not isinstance(msg["message_id"], str):
            raise SchemaValidationError(f"full_metadata_messages[{index}].message_id must be a string")
        
        # Validate timestamp
        self._validate_timestamp(msg["timestamp"], f"full_metadata_messages[{index}].timestamp")
        
        # Validate content
        if not isinstance(msg["content"], str):
            raise SchemaValidationError(f"full_metadata_messages[{index}].content must be a string")
        
        # Validate author
        if not isinstance(msg["author"], str):
            raise SchemaValidationError(f"full_metadata_messages[{index}].author must be a string")
    
    def _validate_timestamp(self, timestamp: Any, field_name: str) -> None:
        """Validate that a timestamp is in ISO 8601 format."""
        if not isinstance(timestamp, str):
            raise SchemaValidationError(f"{field_name} must be a string")
        
        # Check regex pattern
        if not self.ISO_8601_PATTERN.match(timestamp):
            raise SchemaValidationError(
                f"{field_name} must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sss+00:00), "
                f"got '{timestamp}'"
            )
        
        # Try to parse it to ensure it's valid
        try:
            # Handle 'Z' timezone indicator
            timestamp_normalized = timestamp.replace('Z', '+00:00')
            datetime.fromisoformat(timestamp_normalized)
        except ValueError as e:
            raise SchemaValidationError(
                f"{field_name} is not a valid ISO 8601 timestamp: {str(e)}"
            )


def validate_file(file_path: str, strict: bool = True, verbose: bool = True) -> bool:
    """
    Validate a JSON file against the conversation schema.
    
    Args:
        file_path: Path to the JSON file to validate
        strict: If True, perform strict validation
        verbose: If True, print validation results
    
    Returns:
        True if valid, False otherwise
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        return False
    
    checker = ConversationSchemaChecker(strict=strict)
    is_valid, errors, warnings = checker.validate(data)
    
    if verbose:
        if is_valid:
            print(f"✓ {file_path} is valid")
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  - {warning}")
        else:
            print(f"✗ {file_path} is invalid")
            print("\nErrors:")
            for error in errors:
                print(f"  - {error}")
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  - {warning}")
    
    return is_valid


def main():
    """Command-line interface for the schema checker."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate JSON files against the conversation schema"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON file(s) to validate"
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict validation (skip cross-field checks)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (only return exit code)"
    )
    
    args = parser.parse_args()
    
    all_valid = True
    for file_path in args.files:
        is_valid = validate_file(
            file_path,
            strict=not args.no_strict,
            verbose=not args.quiet
        )
        if not is_valid:
            all_valid = False
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

