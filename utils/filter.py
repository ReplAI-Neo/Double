#!/usr/bin/env python3
"""
Filter conversations based on various criteria.

Supports filtering by:
- Start/end dates
- Excluded recipients
- Time of day (morning, afternoon, night) in Pacific timezone
- Number of participants
- Average length of my messages (words)
- My proportion of turns
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import pytz


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


def get_time_of_day_pacific(timestamp_str: str) -> str:
    """
    Get time of day (morning, afternoon, night) in Pacific timezone.
    
    Args:
        timestamp_str: ISO 8601 format timestamp string
        
    Returns:
        "morning" (6am-12pm), "afternoon" (12pm-6pm), or "night" (6pm-6am)
    """
    dt_utc = parse_iso_timestamp(timestamp_str)
    pacific = pytz.timezone('America/Los_Angeles')
    dt_pacific = dt_utc.astimezone(pacific)
    hour = dt_pacific.hour
    
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:  # 18 <= hour < 24 or 0 <= hour < 6
        return "night"


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def calculate_my_avg_message_length(openai_messages: List[Dict[str, Any]]) -> float:
    """
    Calculate average word count of my (assistant) messages.
    
    Args:
        openai_messages: List of OpenAI format messages
        
    Returns:
        Average word count, or 0.0 if no assistant messages
    """
    assistant_messages = [msg for msg in openai_messages if msg.get('role') == 'assistant']
    if not assistant_messages:
        return 0.0
    
    total_words = sum(count_words(msg.get('content', '')) for msg in assistant_messages)
    return total_words / len(assistant_messages)


def calculate_my_turn_proportion(openai_messages: List[Dict[str, Any]]) -> float:
    """
    Calculate proportion of turns that are mine (assistant).
    
    Args:
        openai_messages: List of OpenAI format messages
        
    Returns:
        Proportion as a percentage (0-100), or 0.0 if no messages
    """
    if not openai_messages:
        return 0.0
    
    assistant_turns = sum(1 for msg in openai_messages if msg.get('role') == 'assistant')
    total_turns = len(openai_messages)
    
    return (assistant_turns / total_turns) * 100.0


def conversation_has_time_of_day(
    conversation: Dict[str, Any],
    time_of_day_list: List[str]
) -> bool:
    """
    Check if conversation has any messages in the specified time of day periods.
    
    Args:
        conversation: Conversation object
        time_of_day_list: List of time periods ("morning", "afternoon", "night")
        
    Returns:
        True if any message timestamp falls in the specified time periods
    """
    if not time_of_day_list:
        return True  # No filter means include all
    
    # Check all message timestamps
    for msg in conversation.get('full_metadata_messages', []):
        timestamp = msg.get('timestamp')
        if timestamp:
            tod = get_time_of_day_pacific(timestamp)
            if tod in time_of_day_list:
                return True
    
    return False


def filter_conversations(
    conversations: List[Dict[str, Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exclude_recipients: Optional[List[str]] = None,
    time_of_day: Optional[List[str]] = None,
    num_participants_min: Optional[int] = None,
    num_participants_max: Optional[int] = None,
    my_avg_message_length_min: Optional[float] = None,
    my_avg_message_length_max: Optional[float] = None,
    my_turn_proportion_min: Optional[float] = None,
    my_turn_proportion_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Filter conversations based on the specified criteria.
    
    Args:
        conversations: List of conversation objects
        start_date: ISO 8601 date string - filter conversations starting after this date
        end_date: ISO 8601 date string - filter conversations ending before this date
        exclude_recipients: List of recipient names to exclude
        time_of_day: List of time periods ("morning", "afternoon", "night") in Pacific timezone
        num_participants_min: Minimum number of participants (inclusive)
        num_participants_max: Maximum number of participants (inclusive)
        my_avg_message_length_min: Minimum average word count of my messages
        my_avg_message_length_max: Maximum average word count of my messages
        my_turn_proportion_min: Minimum proportion of my turns (0-100, default 0)
        my_turn_proportion_max: Maximum proportion of my turns (0-100, default 100)
        
    Returns:
        Filtered list of conversations
    """
    filtered = []
    
    # Parse dates if provided
    start_dt = None
    if start_date:
        start_dt = parse_iso_timestamp(start_date)
    
    end_dt = None
    if end_date:
        end_dt = parse_iso_timestamp(end_date)
    
    # Set defaults for turn proportion
    if my_turn_proportion_min is None:
        my_turn_proportion_min = 0.0
    if my_turn_proportion_max is None:
        my_turn_proportion_max = 100.0
    
    for conv in conversations:
        # Filter by start date
        if start_dt:
            first_timestamp = conv.get('first_message_timestamp')
            if first_timestamp:
                first_dt = parse_iso_timestamp(first_timestamp)
                if first_dt < start_dt:
                    continue
            else:
                continue  # Skip if no first timestamp
        
        # Filter by end date
        if end_dt:
            last_timestamp = conv.get('last_message_timestamp')
            if last_timestamp:
                last_dt = parse_iso_timestamp(last_timestamp)
                if last_dt > end_dt:
                    continue
            else:
                continue  # Skip if no last timestamp
        
        # Filter by excluded recipients
        if exclude_recipients:
            recipients = conv.get('recipients', [])
            if any(recipient in exclude_recipients for recipient in recipients):
                continue
        
        # Filter by time of day
        if time_of_day:
            if not conversation_has_time_of_day(conv, time_of_day):
                continue
        
        # Filter by number of participants
        num_participants = conv.get('num_participants', 0)
        if num_participants_min is not None and num_participants < num_participants_min:
            continue
        if num_participants_max is not None and num_participants > num_participants_max:
            continue
        
        # Filter by average length of my messages
        openai_messages = conv.get('openai_messages', [])
        avg_length = calculate_my_avg_message_length(openai_messages)
        if my_avg_message_length_min is not None and avg_length < my_avg_message_length_min:
            continue
        if my_avg_message_length_max is not None and avg_length > my_avg_message_length_max:
            continue
        
        # Filter by my proportion of turns
        turn_proportion = calculate_my_turn_proportion(openai_messages)
        if turn_proportion < my_turn_proportion_min:
            continue
        if turn_proportion > my_turn_proportion_max:
            continue
        
        # All filters passed
        filtered.append(conv)
    
    return filtered


def main():
    """Command-line interface for filtering conversations."""
    parser = argparse.ArgumentParser(
        description='Filter conversations based on various criteria'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input JSON file containing conversations'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output JSON file for filtered conversations'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (ISO 8601 format) - filter conversations starting after this date'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (ISO 8601 format) - filter conversations ending before this date'
    )
    parser.add_argument(
        '--exclude-recipients',
        type=str,
        nargs='+',
        help='List of recipient names to exclude'
    )
    parser.add_argument(
        '--time-of-day',
        type=str,
        nargs='+',
        choices=['morning', 'afternoon', 'night'],
        help='Time of day periods to include (morning, afternoon, night) in Pacific timezone'
    )
    parser.add_argument(
        '--num-participants-min',
        type=int,
        help='Minimum number of participants (inclusive)'
    )
    parser.add_argument(
        '--num-participants-max',
        type=int,
        help='Maximum number of participants (inclusive)'
    )
    parser.add_argument(
        '--my-avg-message-length-min',
        type=float,
        help='Minimum average word count of my messages'
    )
    parser.add_argument(
        '--my-avg-message-length-max',
        type=float,
        help='Maximum average word count of my messages'
    )
    parser.add_argument(
        '--my-turn-proportion-min',
        type=float,
        default=0.0,
        help='Minimum proportion of my turns (0-100, default: 0)'
    )
    parser.add_argument(
        '--my-turn-proportion-max',
        type=float,
        default=100.0,
        help='Maximum proportion of my turns (0-100, default: 100)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print the output JSON'
    )
    
    args = parser.parse_args()
    
    # Load conversations
    with open(args.input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Filter conversations
    filtered = filter_conversations(
        conversations,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_recipients=args.exclude_recipients,
        time_of_day=args.time_of_day,
        num_participants_min=args.num_participants_min,
        num_participants_max=args.num_participants_max,
        my_avg_message_length_min=args.my_avg_message_length_min,
        my_avg_message_length_max=args.my_avg_message_length_max,
        my_turn_proportion_min=args.my_turn_proportion_min,
        my_turn_proportion_max=args.my_turn_proportion_max,
    )
    
    # Save filtered conversations
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2 if args.pretty else None, ensure_ascii=False)
    
    print(f"Filtered {len(conversations)} conversations to {len(filtered)} conversations")
    print(f"Output saved to: {args.output_file}")


if __name__ == '__main__':
    main()

