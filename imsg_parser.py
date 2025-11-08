from bs4 import BeautifulSoup
import json
import argparse
import sys

def parse_messages_to_openai_format(html_content):
    """
    Parse HTML messages and convert to OpenAI chat format.
    
    Messages with sender 'Me' → role 'assistant'
    All other messages → role 'user'
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    messages = []
    
    # Find all divs with class "message"
    message_divs = soup.find_all('div', class_='message')
    
    for msg_div in message_divs:
        # Find the sender
        sender_span = msg_div.find('span', class_='sender')
        if not sender_span:
            continue
            
        sender = sender_span.get_text(strip=True)
        
        # Find the bubble content
        bubble_span = msg_div.find('span', class_='bubble')
        if not bubble_span:
            continue
            
        content = bubble_span.get_text(strip=True)
        
        # Determine role based on sender
        role = 'assistant' if sender == 'Me' else 'user'
        
        messages.append({
            'role': role,
            'content': content
        })
    
    return messages

def parse_html_file(filepath):
    """Parse HTML file and return OpenAI chat format messages."""
    with open(filepath, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return parse_messages_to_openai_format(html_content)

def main():
    """CLI interface for parsing iMessage HTML files."""
    parser = argparse.ArgumentParser(
        description='Parse iMessage HTML export and convert to OpenAI chat format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Parse and print to stdout
  python imsg_parser.py messages.html
  
  # Save to JSON file
  python imsg_parser.py messages.html -o output.json
  
  # Pretty print to stdout
  python imsg_parser.py messages.html --pretty
        '''
    )
    
    parser.add_argument(
        'input_file',
        help='Path to HTML file containing iMessage export'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (default: print to stdout)',
        metavar='FILE'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output with indentation'
    )
    
    args = parser.parse_args()
    
    try:
        # Parse the HTML file
        messages = parse_html_file(args.input_file)
        
        # Format output
        if args.pretty or not args.output:
            output = json.dumps(messages, indent=2, ensure_ascii=False)
        else:
            output = json.dumps(messages, ensure_ascii=False)
        
        # Write to file or stdout
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"✓ Successfully parsed {len(messages)} messages to {args.output}")
        else:
            print(output)
            print(f"\n# Total messages: {len(messages)}", file=sys.stderr)
            
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

