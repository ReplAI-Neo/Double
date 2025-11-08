#!/usr/bin/env python3
"""
Partition conversations by time gaps.

If two consecutive messages are more than X days apart (default 7),
they are partitioned into separate conversation objects.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string to a datetime object.
    
    Args:
        timestamp_str: ISO 8601 format timestamp string
        
    Returns:
        datetime object in UTC timezone
    """
    # Handle 'Z' timezone indicator
    timestamp_str = timestamp_str.replace('Z', '+00:00')
    return datetime.fromisoformat(timestamp_str)


def days_between(timestamp1: str, timestamp2: str) -> float:
    """
    Calculate the number of days between two ISO 8601 timestamps.
    
    Args:
        timestamp1: First timestamp (ISO 8601)
        timestamp2: Second timestamp (ISO 8601)
        
    Returns:
        Number of days between the timestamps (can be negative)
    """
    dt1 = parse_iso_timestamp(timestamp1)
    dt2 = parse_iso_timestamp(timestamp2)
    delta = dt2 - dt1
    return delta.total_seconds() / (24 * 3600)


def find_split_points(full_metadata_messages: List[Dict[str, Any]], max_days: int) -> List[int]:
    """
    Find indices where conversations should be split based on time gaps.
    
    Args:
        full_metadata_messages: List of full metadata messages with timestamps
        max_days: Maximum days between consecutive messages before splitting
        
    Returns:
        List of indices where splits should occur (0-indexed, after this message)
    """
    if len(full_metadata_messages) < 2:
        return []
    
    split_points = []
    
    for i in range(len(full_metadata_messages) - 1):
        current_msg = full_metadata_messages[i]
        next_msg = full_metadata_messages[i + 1]
        
        current_timestamp = current_msg.get('timestamp')
        next_timestamp = next_msg.get('timestamp')
        
        # Skip if either timestamp is missing
        if not current_timestamp or not next_timestamp:
            continue
        
        # Calculate days between messages
        days_diff = days_between(current_timestamp, next_timestamp)
        
        # If gap is greater than max_days, split after current message
        if days_diff > max_days:
            split_points.append(i + 1)  # Split after message at index i
    
    return split_points


def partition_conversation(conversation: Dict[str, Any], max_days: int) -> List[Dict[str, Any]]]:
    """
    Partition a single conversation into multiple conversations based on time gaps.
    
    Args:
        conversation: Conversation object following CONVERSATION_SCHEMA.md
        max_days: Maximum days between consecutive messages before splitting
        
    Returns:
        List of partitioned conversation objects
    """
    full_metadata_messages = conversation.get('full_metadata_messages', [])
    openai_messages = conversation.get('openai_messages', [])
    
    if len(full_metadata_messages) < 2:
        # No need to partition if there's only one or zero messages
        return [conversation]
    
    # Find split points based on full_metadata_messages
    split_points = find_split_points(full_metadata_messages, max_days)
    
    if not split_points:
        # No splits needed
        return [conversation]
    
    # Create mapping from full_metadata_messages to openai_messages
    # openai_messages only includes messages with content
    openai_index = 0
    full_to_openai_map = []
    
    for full_msg in full_metadata_messages:
        content = full_msg.get('content', '')
        # Check if this message would be in openai_messages (has content)
        if content and content.strip():
            full_to_openai_map.append(openai_index)
            openai_index += 1
        else:
            full_to_openai_map.append(None)  # Not in openai_messages
    
    # Split both arrays
    partitions = []
    start_idx = 0
    
    # Add end index for convenience
    split_points_with_end = split_points + [len(full_metadata_messages)]
    
    for end_idx in split_points_with_end:
        # Extract partition from full_metadata_messages
        partition_full_metadata = full_metadata_messages[start_idx:end_idx]
        
        # Extract corresponding partition from openai_messages
        partition_openai = []
        for i in range(start_idx, end_idx):
            openai_idx = full_to_openai_map[i]
            if openai_idx is not None:
                partition_openai.append(openai_messages[openai_idx])
        
        # Get timestamps for this partition
        partition_timestamps = [
            msg.get('timestamp') for msg in partition_full_metadata
            if msg.get('timestamp')
        ]
        
        first_timestamp = partition_timestamps[0] if partition_timestamps else None
        last_timestamp = partition_timestamps[-1] if partition_timestamps else None
        
        # Create new conversation object
        partitioned_conv = {
            'openai_messages': partition_openai,
            'full_metadata_messages': partition_full_metadata,
            'first_message_timestamp': first_timestamp,
            'last_message_timestamp': last_timestamp,
            'recipients': conversation.get('recipients', []).copy(),
            'num_participants': conversation.get('num_participants'),
            'total_messages': len(partition_full_metadata),
            'source': conversation.get('source'),
            'chat_type': conversation.get('chat_type'),
        }
        
        partitions.append(partitioned_conv)
        start_idx = end_idx
    
    return partitions


def partition_conversations(
    input_file: str,
    output_file: str,
    max_days: int = 7,
    pretty: bool = True
) -> None:
    """
    Partition conversations in a JSON file based on time gaps.
    
    Args:
        input_file: Path to input JSON file with conversations
        output_file: Path to output JSON file
        max_days: Maximum days between consecutive messages before splitting (default: 7)
        pretty: Whether to pretty-print the output JSON
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_file}")
    
    # Load conversations
    print(f"Loading conversations from {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    if not isinstance(conversations, list):
        raise ValueError("Input file must contain a JSON array of conversations")
    
    print(f"Loaded {len(conversations)} conversation(s)")
    
    # Partition each conversation
    partitioned_conversations = []
    total_original = len(conversations)
    total_partitioned = 0
    
    for i, conversation in enumerate(conversations):
        partitions = partition_conversation(conversation, max_days)
        partitioned_conversations.extend(partitions)
        total_partitioned += len(partitions)
        
        if len(partitions) > 1:
            print(f"  Conversation {i+1}: partitioned into {len(partitions)} conversation(s)")
    
    # Save partitioned conversations
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(partitioned_conversations, f, indent=2, ensure_ascii=False)
        else:
            json.dump(partitioned_conversations, f, ensure_ascii=False)
    
    print(f"\nPartitioned {total_original} conversation(s) into {total_partitioned} conversation(s)")
    print(f"Output saved to: {output_file}")


def main():
    """Command-line interface for partitioning conversations."""
    parser = argparse.ArgumentParser(
        description="Partition conversations by time gaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Partition conversations with default 7-day gap
  python utils/partition.py data/merged/all_conversations.json -o data/merged/all_conversations_partitioned.json
  
  # Partition with 14-day gap
  python utils/partition.py data/merged/all_conversations.json -o data/merged/all_conversations_partitioned.json --max-days 14
  
  # Partition without pretty-printing
  python utils/partition.py data/merged/all_conversations.json -o data/merged/all_conversations_partitioned.json --no-pretty
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input JSON file with conversations to partition"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--max-days",
        type=int,
        default=7,
        help="Maximum days between consecutive messages before splitting (default: 7)"
    )
    
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Don't pretty-print JSON output (smaller file size)"
    )
    
    args = parser.parse_args()
    
    partition_conversations(
        input_file=args.input_file,
        output_file=args.output,
        max_days=args.max_days,
        pretty=not args.no_pretty
    )


if __name__ == "__main__":
    main()

