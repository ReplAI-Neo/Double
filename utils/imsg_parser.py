"""
iMessage HTML Parser - Universal Schema

Parses iMessage HTML exports into a standardized schema compatible with
Discord, iMessage, and Instagram formats.
"""

from bs4 import BeautifulSoup
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to datetime object."""
    try:
        # Format: "Oct 27, 2024  1:42:02 PM"
        return datetime.strptime(timestamp_str.strip(), "%b %d, %Y  %I:%M:%S %p")
    except ValueError:
        try:
            # Try alternative format without seconds
            return datetime.strptime(timestamp_str.strip(), "%b %d, %Y  %I:%M %p")
        except ValueError:
            return None


def parse_imessage_html(html_content: str, filename: str = "") -> Optional[Dict[str, Any]]:
    """
    Parse iMessage HTML export into universal conversation schema.
    
    Returns a conversation object matching the universal schema.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract recipient from filename (e.g., "+16692689561.html")
    recipient = Path(filename).stem if filename else "Unknown"
    
    openai_messages = []
    full_metadata_messages = []
    authors = set()
    timestamps = []
    
    # Find all message divs
    message_divs = soup.find_all('div', class_='message')
    
    for idx, msg_div in enumerate(message_divs):
        # Find sender
        sender_span = msg_div.find('span', class_='sender')
        if not sender_span:
            continue
        
        sender = sender_span.get_text(strip=True)
        authors.add(sender)
        
        # Find bubble content
        bubble_span = msg_div.find('span', class_='bubble')
        if not bubble_span:
            continue
        
        content = bubble_span.get_text(strip=True)
        
        # Find timestamp
        timestamp = None
        timestamp_span = msg_div.find('span', class_='timestamp')
        if timestamp_span:
            timestamp_link = timestamp_span.find('a')
            if timestamp_link:
                timestamp_text = timestamp_link.get_text(strip=True)
                timestamp_dt = parse_timestamp(timestamp_text)
                if timestamp_dt:
                    timestamp = timestamp_dt.isoformat()
                    timestamps.append(timestamp_dt)
        
        # Determine role
        role = 'assistant' if sender == 'Me' else 'user'
        
        # Create message ID
        message_id = f"imsg_{recipient}_{idx}"
        
        # Add to full metadata (always include all messages)
        full_metadata_messages.append({
            'message_id': message_id,
            'timestamp': timestamp,
            'content': content,
            'author': sender
        })
        
        # Add to OpenAI messages (only if has content)
        if content.strip():
            openai_messages.append({
                'role': role,
                'content': content
            })
    
    if not full_metadata_messages:
        return None
    
    # Determine recipients (everyone except "Me")
    recipients = [author for author in authors if author != 'Me']
    
    # Determine chat type
    chat_type = 'direct' if len(authors) <= 2 else 'group'
    
    # Get first and last timestamps
    first_timestamp = timestamps[0].isoformat() if timestamps else None
    last_timestamp = timestamps[-1].isoformat() if timestamps else None
    
    return {
        'openai_messages': openai_messages,
        'full_metadata_messages': full_metadata_messages,
        'first_message_timestamp': first_timestamp,
        'last_message_timestamp': last_timestamp,
        'recipients': recipients,
        'num_participants': len(authors),
        'total_messages': len(full_metadata_messages),
        'source': 'imessage',
        'chat_type': chat_type
    }


def parse_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Parse all HTML files in a folder.
    
    Args:
        folder_path: Path to folder containing HTML files
    
    Returns:
        List of conversation objects
    """
    folder = Path(folder_path)
    conversations = []
    
    # Find all HTML files
    html_files = list(folder.glob('*.html'))
    
    if not html_files:
        print(f"No HTML files found in {folder_path}", file=sys.stderr)
        return []
    
    print(f"Found {len(html_files)} HTML file(s) in {folder_path}", file=sys.stderr)
    
    for html_file in sorted(html_files):
        print(f"  Processing: {html_file.name}...", file=sys.stderr)
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            conversation = parse_imessage_html(html_content, html_file.name)
            
            if conversation:
                conversations.append(conversation)
                print(f"    ✓ Parsed {conversation['total_messages']} messages "
                      f"from {conversation['recipients']}", file=sys.stderr)
            else:
                print("    ✗ No messages found", file=sys.stderr)
                
        except Exception as e:
            print(f"    ✗ Error: {e}", file=sys.stderr)
            continue
    
    return conversations


def main():
    """CLI interface for parsing iMessage HTML files."""
    parser = argparse.ArgumentParser(
        description='Parse iMessage HTML exports into universal conversation schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Parse single file
  python imsg_parser.py file.html -o conversations.json
  
  # Parse entire folder
  python imsg_parser.py imsg_data/ -o conversations.json
  
  # Pretty print output
  python imsg_parser.py imsg_data/ -o conversations.json --pretty
        '''
    )
    
    parser.add_argument(
        'input_path',
        help='Path to HTML file or folder containing HTML files'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file',
        metavar='FILE'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output with indentation'
    )
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input_path)
        
        # Check if input is a file or folder
        if input_path.is_file():
            print(f"Processing single file: {input_path.name}", file=sys.stderr)
            with open(input_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            conversation = parse_imessage_html(html_content, input_path.name)
            conversations = [conversation] if conversation else []
        elif input_path.is_dir():
            print(f"Processing folder: {input_path}", file=sys.stderr)
            conversations = parse_folder(str(input_path))
        else:
            print(f"Error: '{args.input_path}' is not a valid file or folder", 
                  file=sys.stderr)
            sys.exit(1)
        
        if not conversations:
            print("No conversations were parsed", file=sys.stderr)
            sys.exit(1)
        
        # Write output
        output_data = conversations
        
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(output_data, f, ensure_ascii=False)
        
        # Print summary
        total_messages = sum(conv['total_messages'] for conv in conversations)
        total_openai_messages = sum(len(conv['openai_messages']) for conv in conversations)
        
        print(f"\n✓ Successfully parsed {len(conversations)} conversation(s)", 
              file=sys.stderr)
        print(f"  Total messages: {total_messages}", file=sys.stderr)
        print(f"  OpenAI messages: {total_openai_messages}", file=sys.stderr)
        print(f"  Output: {args.output}", file=sys.stderr)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
