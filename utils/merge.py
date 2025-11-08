#!/usr/bin/env python3
"""
Merge multiple JSON conversation files into a single JSON file.

Each input file should be a JSON array of conversation objects following
the schema defined in CONVERSATION_SCHEMA.md.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def merge_conversation_files(input_dir: str, output_file: str, pretty: bool = True) -> None:
    """
    Merge all JSON conversation files from a directory into one file.
    
    Args:
        input_dir: Directory containing JSON files to merge
        output_file: Path to output JSON file
        pretty: Whether to pretty-print the output JSON (indent=2)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Find all JSON files in the directory
    json_files = sorted(input_path.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to merge")
    
    all_conversations: List[Dict[str, Any]] = []
    total_conversations = 0
    errors = []
    
    # Read and merge all JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            # Validate that it's a list
            if not isinstance(conversations, list):
                errors.append(f"{json_file.name}: Not a JSON array (skipped)")
                continue
            
            # Add all conversations from this file
            all_conversations.extend(conversations)
            total_conversations += len(conversations)
            print(f"  Loaded {len(conversations)} conversation(s) from {json_file.name}")
            
        except json.JSONDecodeError as e:
            errors.append(f"{json_file.name}: Invalid JSON - {e}")
        except Exception as e:
            errors.append(f"{json_file.name}: Error - {e}")
    
    # Report any errors
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
    
    # Write merged output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)
        else:
            json.dump(all_conversations, f, ensure_ascii=False)
    
    print(f"\nMerged {total_conversations} conversation(s) from {len(json_files)} file(s)")
    print(f"Output saved to: {output_file}")


def main():
    """Command-line interface for merging conversation files."""
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON conversation files into a single JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all JSON files in a directory
  python utils/merge.py data/processed -o data/merged/all_conversations.json
  
  # Merge without pretty-printing (smaller file size)
  python utils/merge.py data/processed -o data/merged/all_conversations.json --no-pretty
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing JSON conversation files to merge"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Don't pretty-print JSON output (smaller file size)"
    )
    
    args = parser.parse_args()
    
    merge_conversation_files(
        input_dir=args.input_dir,
        output_file=args.output,
        pretty=not args.no_pretty
    )


if __name__ == "__main__":
    main()

