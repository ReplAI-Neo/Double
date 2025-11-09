#!/usr/bin/env python3
"""
Fix conversation format to have alternating user/assistant messages.

This script processes conversations with multiple consecutive user messages
and converts them into proper alternating format by:
1. Merging consecutive user messages into one
2. Ensuring user/assistant alternation
3. Filtering out conversations without assistant responses
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def join_message_content(role: Optional[str], content_parts: List[str]) -> str:
    """
    Join message chunks, using double newlines for assistant messages so they can
    later be split back into individual texts.
    """
    if not content_parts:
        return ""
    separator = "\n\n" if role == "assistant" else "\n"
    return separator.join(content_parts)


def merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages from the same role.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        List of messages with consecutive same-role messages merged
    """
    if not messages:
        return []
    
    merged = []
    current_role = None
    current_content = []
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '').strip()
        
        if not content:  # Skip empty messages
            continue
        
        if role == current_role:
            # Same role, accumulate content
            current_content.append(content)
        else:
            # Different role, save previous and start new
            if current_content:
                merged.append({
                    'role': current_role,
                    'content': join_message_content(current_role, current_content)
                })
            current_role = role
            current_content = [content]
    
    # Don't forget the last accumulated message
    if current_content:
        merged.append({
            'role': current_role,
            'content': join_message_content(current_role, current_content)
        })
    
    return merged


def ensure_alternating(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure messages alternate between user and assistant.
    
    Args:
        messages: List of merged messages
    
    Returns:
        List of alternating messages, or empty if invalid
    """
    if not messages:
        return []
    
    # Must start with user
    if messages[0]['role'] != 'user':
        messages = messages[1:]
    
    if not messages:
        return []
    
    # Must have at least one assistant message
    has_assistant = any(m['role'] == 'assistant' for m in messages)
    if not has_assistant:
        return []
    
    # Build alternating sequence
    alternating = []
    expected_role = 'user'
    
    for msg in messages:
        if msg['role'] == expected_role:
            alternating.append(msg)
            expected_role = 'assistant' if expected_role == 'user' else 'user'
        elif msg['role'] == 'user' and expected_role == 'assistant':
            # Skip this user message, we need an assistant message
            continue
        elif msg['role'] == 'assistant' and expected_role == 'user':
            # Got assistant when expecting user, skip it
            continue
    
    # Must end with assistant for complete conversation
    if alternating and alternating[-1]['role'] != 'assistant':
        alternating = alternating[:-1]
    
    return alternating


def fix_conversation(conv: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Fix a single conversation to have proper alternating format.
    
    Args:
        conv: Conversation dict with 'messages' field
    
    Returns:
        Fixed conversation or None if invalid
    """
    messages = conv.get('messages', [])
    
    # Step 1: Merge consecutive messages
    merged = merge_consecutive_messages(messages)
    
    # Step 2: Ensure alternating
    alternating = ensure_alternating(merged)
    
    if not alternating or len(alternating) < 2:
        return None
    
    return {'messages': alternating}


def main():
    parser = argparse.ArgumentParser(
        description='Fix conversation format to have alternating user/assistant messages'
    )
    parser.add_argument(
        'input_file',
        help='Input JSON file with conversations'
    )
    parser.add_argument(
        'output_file',
        help='Output JSON file with fixed conversations'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )
    
    args = parser.parse_args()
    
    # Load input
    print(f"Loading conversations from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Fix conversations
    print("Fixing conversation format...")
    fixed_conversations = []
    skipped = 0
    
    for i, conv in enumerate(conversations):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(conversations)}...")
        
        fixed = fix_conversation(conv)
        if fixed:
            fixed_conversations.append(fixed)
        else:
            skipped += 1
    
    print(f"\nResults:")
    print(f"  Original conversations: {len(conversations)}")
    print(f"  Fixed conversations: {len(fixed_conversations)}")
    print(f"  Skipped (no assistant or too short): {skipped}")
    
    # Save output
    print(f"\nSaving to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(fixed_conversations, f, indent=2, ensure_ascii=False)
        else:
            json.dump(fixed_conversations, f, ensure_ascii=False)
    
    print("✓ Done!")
    
    # Show example
    if fixed_conversations:
        print("\nExample fixed conversation:")
        example = fixed_conversations[0]
        for i, msg in enumerate(example['messages'][:4]):
            content_preview = msg['content'][:80].replace('\n', ' ')
            print(f"  {i+1}. [{msg['role']}]: {content_preview}...")


if __name__ == '__main__':
    main()
