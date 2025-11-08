#!/usr/bin/env python3
"""
Parse Instagram message data into OpenAI-compatible chat format.

Parses Instagram message data into a standardized schema compatible with
Discord, iMessage, and Instagram formats.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import argparse
from encryption_utils import generate_encryption_key, encrypt_file


def fix_instagram_encoding(text: str) -> str:
    """
    Fix Instagram's encoding issues where Unicode characters are double-encoded.
    Instagram exports UTF-8 text but represents it as Latin-1.
    """
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def parse_message_file(
    file_path: Path,
    user_name: str,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse a single Instagram message.json file into universal conversation schema.

    Args:
        file_path: Path to message.json file
        user_name: User's display name (case-insensitive)
        start_time_ms: Optional start time filter (Unix ms)
        end_time_ms: Optional end time filter (Unix ms)

    Returns:
        Conversation object matching universal schema, or None if no valid messages
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract participants
    participants_data = data.get("participants", [])
    all_participants = [
        fix_instagram_encoding(p.get("name", "")) for p in participants_data
    ]

    # Get conversation name from folder
    conversation_name = file_path.parent.name

    # Get messages
    messages_data = data.get("messages", [])

    # Sort by timestamp (oldest first)
    sorted_messages = sorted(messages_data, key=lambda x: x.get("timestamp_ms", 0))

    openai_messages = []
    full_metadata_messages = []
    timestamps = []
    authors = set()

    for idx, msg in enumerate(sorted_messages):
        timestamp_ms = msg.get("timestamp_ms", 0)

        # Apply time filters
        if start_time_ms and timestamp_ms < start_time_ms:
            continue
        if end_time_ms and timestamp_ms > end_time_ms:
            continue

        # Get message content
        content = msg.get("content")
        if not content:
            continue

        content = fix_instagram_encoding(content)

        # Filter out system messages
        if content == "Liked a message" or not content.strip():
            continue

        # Get sender
        sender = fix_instagram_encoding(msg.get("sender_name", ""))
        authors.add(sender)

        # Determine role
        role = "assistant" if sender.lower() == user_name.lower() else "user"

        # Convert timestamp to ISO format
        timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        timestamp_iso = timestamp_dt.isoformat()
        timestamps.append(timestamp_dt)

        # Create message ID
        message_id = f"ig_{conversation_name}_{timestamp_ms}_{idx}"

        # Add to full metadata
        full_metadata_messages.append(
            {
                "message_id": message_id,
                "timestamp": timestamp_iso,
                "content": content,
                "author": sender,
            }
        )

        # Add to OpenAI messages
        openai_messages.append({"role": role, "content": content})

    if not full_metadata_messages:
        return None

    # Determine recipients (everyone except user)
    recipients = [author for author in authors if author.lower() != user_name.lower()]

    # Determine chat type
    chat_type = "direct" if len(all_participants) <= 2 else "group"

    # Get first and last timestamps
    first_timestamp = timestamps[0].isoformat() if timestamps else None
    last_timestamp = timestamps[-1].isoformat() if timestamps else None

    return {
        "openai_messages": openai_messages,
        "full_metadata_messages": full_metadata_messages,
        "first_message_timestamp": first_timestamp,
        "last_message_timestamp": last_timestamp,
        "recipients": recipients,
        "num_participants": len(all_participants),
        "total_messages": len(full_metadata_messages),
        "source": "instagram",
        "chat_type": chat_type,
    }


def parse_all_messages(
    root_dir: str,
    user_name: str,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Parse all Instagram message files in a directory.

    Args:
        root_dir: Root directory containing message folders
        user_name: User's display name
        start_time_ms: Optional start time filter
        end_time_ms: Optional end time filter

    Returns:
        List of conversation objects
    """
    root_path = Path(root_dir)
    conversations = []

    # Find all message.json files
    message_files = list(root_path.glob("**/message.json"))

    if not message_files:
        print(f"No message.json files found in {root_dir}")
        return []

    print(f"Found {len(message_files)} message file(s)")

    for msg_file in message_files:
        try:
            print(f"  Processing: {msg_file.parent.name}...", end=" ")

            conversation = parse_message_file(
                msg_file, user_name, start_time_ms, end_time_ms
            )

            if conversation:
                conversations.append(conversation)
                print(f"✓ {conversation['total_messages']} messages")
            else:
                print("✗ No valid messages")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    return conversations


def parse_date_to_ms(date_str: str) -> int:
    """Parse date string to Unix timestamp in milliseconds."""
    import calendar

    try:
        # Try as unix timestamp (seconds)
        timestamp = float(date_str)
        # If < 10 billion, assume seconds; otherwise milliseconds
        if timestamp < 10000000000:
            return int(timestamp * 1000)
        return int(timestamp)
    except ValueError:
        pass

    # Try ISO format
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Treat as UTC
            return int(calendar.timegm(dt.timetuple()) * 1000)
        else:
            return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(
            f"Invalid date format: {date_str}. "
            "Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) or unix timestamp."
        )


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parse Instagram messages into universal conversation schema"
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Root directory containing Instagram message folders",
    )

    parser.add_argument(
        "--user-name",
        help="Your name as it appears in Instagram messages (case-insensitive)",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file",
        metavar="FILE",
    )

    parser.add_argument(
        "--start-time",
        help="Only include messages from this date onward (ISO format or unix timestamp)",
    )

    parser.add_argument(
        "--end-time",
        help="Only include messages up to this date (ISO format or unix timestamp)",
    )

    parser.add_argument(
        "--pretty", action="store_true", help="Pretty print JSON output"
    )

    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt output file with AES-256",
    )

    parser.add_argument(
        "--encryption-key",
        help="Encryption key (base64-encoded). If not provided with --encrypt, generates new key.",
    )

    parser.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate a new encryption key and exit",
    )

    args = parser.parse_args()

    # Handle key generation mode
    if args.generate_key:
        key = generate_encryption_key()
        print("Generated encryption key (save this securely!):")
        print(key)
        print("\nUse this key with: --encryption-key YOUR_KEY")
        print("Keep this key safe! You'll need it to decrypt your data.")
        return

    # Validate required arguments when not generating key
    if not args.input_dir:
        parser.error("input_dir is required (unless using --generate-key)")

    if not args.user_name:
        parser.error("--user-name is required (unless using --generate-key)")

    if not args.output:
        parser.error("-o/--output is required (unless using --generate-key)")

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        return

    # Parse time filters if provided
    start_time_ms = None
    end_time_ms = None

    if args.start_time:
        try:
            start_time_ms = parse_date_to_ms(args.start_time)
        except ValueError as e:
            print(f"Error: {e}")
            return

    if args.end_time:
        try:
            end_time_ms = parse_date_to_ms(args.end_time)
        except ValueError as e:
            print(f"Error: {e}")
            return

    # Handle encryption key
    encryption_key = None
    if args.encrypt:
        if args.encryption_key:
            encryption_key = args.encryption_key
        else:
            encryption_key = generate_encryption_key()
            print("\n" + "=" * 60)
            print("🔑 GENERATED NEW ENCRYPTION KEY (SAVE THIS SECURELY!):")
            print(encryption_key)
            print("=" * 60)
            print("⚠️  You MUST save this key to decrypt your data later!")
            print("=" * 60 + "\n")

    # Parse messages
    print(f"\nParsing Instagram messages from: {args.input_dir}")
    print(f"User name: {args.user_name}")
    if start_time_ms:
        print(
            f"Start time: {datetime.fromtimestamp(start_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')}"
        )
    if end_time_ms:
        print(
            f"End time: {datetime.fromtimestamp(end_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')}"
        )
    print(f"Encryption: {'ENABLED' if encryption_key else 'disabled'}")
    print("=" * 60)

    conversations = parse_all_messages(
        args.input_dir,
        args.user_name,
        start_time_ms,
        end_time_ms,
    )

    if not conversations:
        print("\nNo conversations were parsed")
        return

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        else:
            json.dump(conversations, f, ensure_ascii=False)

    # Encrypt if requested
    if encryption_key:
        try:
            encrypt_file(output_path, encryption_key)
            print(
                f"\n✓ Successfully parsed and encrypted {len(conversations)} conversation(s)"
            )
        except Exception as e:
            print(f"\n✓ Successfully parsed {len(conversations)} conversation(s)")
            print(f"✗ Encryption failed: {e}")
    else:
        print(f"\n✓ Successfully parsed {len(conversations)} conversation(s)")

    # Print summary
    total_messages = sum(conv["total_messages"] for conv in conversations)
    total_openai_messages = sum(len(conv["openai_messages"]) for conv in conversations)

    print(f"  Total conversations: {len(conversations)}")
    print(f"  Total messages: {total_messages}")
    print(f"  OpenAI messages: {total_openai_messages}")
    print(f"  Output: {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
