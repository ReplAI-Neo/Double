#!/usr/bin/env python3
"""
Parse Instagram message data into OpenAI-compatible chat format.

This script recursively finds all message.json files in subdirectories,
parses them, and converts them to the OpenAI chat format (for training).
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def fix_instagram_encoding(text: str) -> str:
    """
    Fix Instagram's encoding issues where Unicode characters are double-encoded.
    Instagram exports UTF-8 text but represents it as Latin-1.
    """
    try:
        # Convert the incorrectly decoded string back to bytes, then decode properly
        return text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If conversion fails, return original text
        return text


def parse_message_file(
    file_path: Path,
    user_name: str,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Parse a single Instagram message.json file and convert to OpenAI format.

    Args:
        file_path: Path to the message.json file
        user_name: The name of the user (to identify which messages are from "assistant" role)
        start_time_ms: Optional start timestamp in milliseconds (inclusive)
        end_time_ms: Optional end timestamp in milliseconds (inclusive)

    Returns:
        List of message dictionaries in OpenAI format
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    [p["name"] for p in data.get("participants", [])]
    messages = data.get("messages", [])

    # Filter and convert messages
    openai_messages = []

    # Sort messages by timestamp (oldest first)
    sorted_messages = sorted(messages, key=lambda x: x.get("timestamp_ms", 0))

    for msg in sorted_messages:
        # Filter by time range if specified
        timestamp = msg.get("timestamp_ms", 0)
        if start_time_ms and timestamp < start_time_ms:
            continue
        if end_time_ms and timestamp > end_time_ms:
            continue

        # Skip messages without content or with generic content
        content = msg.get("content", "").strip()
        if not content or content == "Liked a message":
            continue

        # Fix encoding issues
        content = fix_instagram_encoding(content)
        sender = fix_instagram_encoding(msg.get("sender_name", ""))

        # Determine role (user is the assistant being trained, others are users)
        if sender.lower() == user_name.lower():
            role = "assistant"
        else:
            role = "user"

        openai_messages.append(
            {
                "role": role,
                "content": content,
                "metadata": {
                    "sender": sender,
                    "timestamp_ms": msg.get("timestamp_ms"),
                    "conversation": str(file_path.parent.name),
                },
            }
        )

    return openai_messages


def find_all_message_files(root_dir: str) -> List[Path]:
    """
    Recursively find all message.json files in the directory.

    Args:
        root_dir: Root directory to search

    Returns:
        List of Path objects to message.json files
    """
    root_path = Path(root_dir)
    return list(root_path.rglob("message*.json"))


def parse_date_to_ms(date_str: str) -> int:
    """
    Convert a date string to milliseconds timestamp.

    Supports formats:
    - ISO date: "2024-01-15" or "2024-01-15T10:30:00"
    - Unix timestamp in seconds: "1705334400"
    - Unix timestamp in milliseconds: "1705334400000"

    Args:
        date_str: Date string to parse

    Returns:
        Timestamp in milliseconds
    """
    # Try parsing as unix timestamp (seconds or milliseconds)
    if date_str.isdigit():
        timestamp = int(date_str)
        # If it looks like seconds (less than year 3000 in seconds), convert to ms
        if timestamp < 32503680000:
            return timestamp * 1000
        return timestamp

    # Try parsing as ISO date
    import calendar

    try:
        # Try parsing as ISO datetime
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

        # If timezone-naive, treat as UTC (Instagram uses UTC)
        if dt.tzinfo is None:
            return int(calendar.timegm(dt.timetuple()) * 1000)
        else:
            return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(
            f"Invalid date format: {date_str}. "
            "Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) or unix timestamp."
        )


def parse_all_messages(
    root_dir: str,
    user_name: str,
    output_format: str = "combined",
    output_dir: str = "output",
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> None:
    """
    Parse all Instagram messages and save to OpenAI format.

    Args:
        root_dir: Root directory containing message folders
        user_name: The user's name in the messages
        output_format: Either "combined" (all conversations in one file) or
                      "separate" (one file per conversation)
        output_dir: Directory to save output files
        start_time_ms: Optional start timestamp in milliseconds (inclusive)
        end_time_ms: Optional end timestamp in milliseconds (inclusive)
    """
    message_files = find_all_message_files(root_dir)

    if not message_files:
        print(f"No message.json files found in {root_dir}")
        return

    print(f"Found {len(message_files)} message file(s)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if output_format == "combined":
        # Combine all messages into a single conversation
        all_messages = []

        for msg_file in message_files:
            print(f"Processing: {msg_file}")
            try:
                messages = parse_message_file(
                    msg_file, user_name, start_time_ms, end_time_ms
                )
                all_messages.extend(messages)
            except Exception as e:
                print(f"Error processing {msg_file}: {e}")

        # Sort all messages by timestamp
        all_messages.sort(key=lambda x: x.get("metadata", {}).get("timestamp_ms", 0))

        # Create output structure
        output_data = {
            "messages": [
                {"role": msg["role"], "content": msg["content"]} for msg in all_messages
            ],
        }

        # Save to file
        output_file = output_path / "all_messages_combined.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved combined messages to: {output_file}")
        print(f"Total messages: {len(all_messages)}")

    else:
        # Save each conversation separately
        conversation_stats = []

        for msg_file in message_files:
            print(f"Processing: {msg_file}")
            try:
                messages = parse_message_file(
                    msg_file, user_name, start_time_ms, end_time_ms
                )

                if not messages:
                    print("  -> No valid messages found, skipping")
                    continue

                # Create output structure
                output_data = {
                    "messages": [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                    ],
                }

                # Create safe filename
                conversation_name = msg_file.parent.name
                safe_name = "".join(
                    c if c.isalnum() or c in ("-", "_") else "_"
                    for c in conversation_name
                )
                output_file = output_path / f"{safe_name}.json"

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                conversation_stats.append(
                    {
                        "name": conversation_name,
                        "file": output_file,
                        "count": len(messages),
                    }
                )

                print(f"  -> Saved {len(messages)} messages to: {output_file}")

            except Exception as e:
                print(f"  -> Error: {e}")

        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"Processed {len(conversation_stats)} conversation(s)")
        for stat in conversation_stats:
            print(f"  - {stat['name']}: {stat['count']} messages")


def main():
    """Main entry point with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Instagram messages into OpenAI chat format"
    )
    parser.add_argument(
        "input_dir", help="Root directory containing Instagram message folders"
    )
    parser.add_argument(
        "--user-name",
        required=True,
        help="Your name as it appears in Instagram messages (case-insensitive)",
    )
    parser.add_argument(
        "--format",
        choices=["combined", "separate"],
        default="separate",
        help="Output format: combined (all in one file) or separate (one per conversation)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output files (default: output)",
    )
    parser.add_argument(
        "--start-time",
        help="Start time filter (format: YYYY-MM-DD or unix timestamp)",
    )
    parser.add_argument(
        "--end-time",
        help="End time filter (format: YYYY-MM-DD or unix timestamp)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        return

    # Parse time filters if provided
    start_time_ms = None
    end_time_ms = None

    if args.start_time:
        try:
            start_time_ms = parse_date_to_ms(args.start_time)
            start_date = datetime.fromtimestamp(start_time_ms / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

    if args.end_time:
        try:
            end_time_ms = parse_date_to_ms(args.end_time)
            end_date = datetime.fromtimestamp(end_time_ms / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

    print(f"Parsing Instagram messages from: {args.input_dir}")
    print(f"User name: {args.user_name}")
    print(f"Output format: {args.format}")
    print(f"Output directory: {args.output_dir}")
    if start_time_ms:
        print(f"Start time: {start_date}")
    if end_time_ms:
        print(f"End time: {end_date}")
    print("=" * 60)

    parse_all_messages(
        args.input_dir,
        user_name=args.user_name,
        output_format=args.format,
        output_dir=args.output_dir,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
