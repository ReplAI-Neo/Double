import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class DiscordParser:
    """Parser for Discord direct message exports in JSON format."""
    
    # The user's Discord ID (stephenx)
    SELF_USER_ID = "669729546840309760"
    
    def __init__(self, data_dir: str):
        """
        Initialize the parser.
        
        Args:
            data_dir: Path to directory containing Discord JSON exports
        """
        self.data_dir = Path(data_dir)
    
    def parse_all(self) -> List[Dict[str, Any]]:
        """
        Parse all Discord JSON files in the data directory.
        
        Returns:
            List of parsed conversations with all metadata (empty conversations are excluded)
        """
        conversations = []
        
        # Get all JSON files in the directory
        json_files = sorted(self.data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                # parse_conversation returns a single conversation dict
                conversation = self.parse_conversation(json_file)
                # Only include non-empty conversations
                if conversation['total_messages'] > 0:
                    conversations.append(conversation)
            except Exception as e:
                print(f"Error parsing {json_file.name}: {e}")
                continue
        
        return conversations
    
    def parse_conversation(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single Discord conversation JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing parsed conversation data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = data.get('messages', [])
        channel = data.get('channel', {})
        
        # Extract recipients from messages only (not channel name)
        recipients = self._extract_recipients(channel, messages)
        
        # Get first and last message timestamps
        first_timestamp = None
        last_timestamp = None
        if messages:
            first_timestamp = messages[0].get('timestamp')
            last_timestamp = messages[-1].get('timestamp')
        
        # Parse messages into OpenAI format and full metadata
        openai_messages = []
        full_metadata_messages = []
        
        for msg in messages:
            # Create OpenAI-compatible message
            openai_msg = self._to_openai_format(msg)
            if openai_msg:  # Only include if there's actual content
                openai_messages.append(openai_msg)
            
            # Create full metadata message
            full_metadata_msg = self._extract_full_metadata(msg, recipients)
            full_metadata_messages.append(full_metadata_msg)
        
        # Count unique participants
        participant_ids = set()
        for msg in messages:
            author = msg.get('author', {})
            author_id = author.get('id')
            if author_id:
                participant_ids.add(author_id)
        
        # Always include self in participant count
        participant_ids.add(self.SELF_USER_ID)
        
        # Determine chat type: direct (2 people) or group (3+ people)
        chat_type = "direct" if len(participant_ids) == 2 else "group"
        
        return {
            'openai_messages': openai_messages,
            'full_metadata_messages': full_metadata_messages,
            'first_message_timestamp': first_timestamp,
            'last_message_timestamp': last_timestamp,
            'recipients': recipients,
            'num_participants': len(participant_ids),
            'total_messages': len(messages),
            'source': 'discord',
            'chat_type': chat_type
        }
    
    def _extract_recipients(self, channel: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract list of recipient names from message authors only.
        
        Args:
            channel: Channel metadata (unused but kept for compatibility)
            messages: List of messages
            
        Returns:
            List of recipient names (excluding self)
        """
        recipients = set()
        
        # Extract from message authors only
        for msg in messages:
            author = msg.get('author', {})
            author_id = author.get('id')
            # Don't include self
            if author_id and author_id != self.SELF_USER_ID:
                nickname = author.get('nickname') or author.get('name', '')
                if nickname:
                    recipients.add(nickname)
        
        return sorted(list(recipients))
    
    def _to_openai_format(self, message: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Convert Discord message to OpenAI-compatible format.
        
        Args:
            message: Discord message object
            
        Returns:
            OpenAI-formatted message or None if no content
        """
        content = message.get('content', '')
        author = message.get('author', {})
        author_id = author.get('id')
        
        # Skip messages without content (e.g., system messages)
        if not content:
            return None
        
        # Determine role: self is "assistant", others are "user"
        role = "assistant" if author_id == self.SELF_USER_ID else "user"
        
        return {
            'role': role,
            'content': content
        }
    
    def _extract_full_metadata(self, message: Dict[str, Any], recipients: List[str]) -> Dict[str, Any]:
        """
        Extract full metadata for a message.
        
        Args:
            message: Discord message object
            recipients: List of conversation recipients (unused but kept for compatibility)
            
        Returns:
            Dictionary with complete message metadata
        """
        author = message.get('author', {})
        # Use nickname if available, otherwise use name
        author_name = author.get('nickname') or author.get('name', '')
        
        return {
            'message_id': message.get('id'),
            'timestamp': message.get('timestamp'),
            'content': message.get('content', ''),
            'author': author_name
        }
    
    def save_parsed_data(self, conversations: List[Dict[str, Any]], output_file: str):
        """
        Save parsed conversations to a JSON file.
        
        Args:
            conversations: List of parsed conversations
            output_file: Path to output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(conversations)} conversations to {output_file}")
    
    def get_summary_stats(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for all parsed conversations.
        
        Args:
            conversations: List of parsed conversations
            
        Returns:
            Dictionary with summary statistics
        """
        total_messages = sum(conv['total_messages'] for conv in conversations)
        total_openai_messages = sum(len(conv['openai_messages']) for conv in conversations)
        
        return {
            'total_conversations': len(conversations),
            'total_messages': total_messages,
            'total_openai_messages': total_openai_messages,
            'avg_messages_per_conversation': total_messages / len(conversations) if conversations else 0,
            'conversations_by_type': self._count_by_type(conversations)
        }
    
    def _count_by_type(self, conversations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count conversations by chat type."""
        type_counts = {}
        for conv in conversations:
            chat_type = conv['chat_type']
            type_counts[chat_type] = type_counts.get(chat_type, 0) + 1
        return type_counts


def main():
    """Example usage of the Discord parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse Discord direct message exports')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing Discord JSON exports')
    parser.add_argument('--output', type=str, default='discord_parsed.json',
                        help='Output JSON file for parsed data')
    parser.add_argument('--stats', action='store_true',
                        help='Print summary statistics')
    
    args = parser.parse_args()
    
    # Initialize parser
    discord_parser = DiscordParser(args.data_dir)
    
    # Parse all conversations
    print(f"Parsing Discord data from {args.data_dir}...")
    conversations = discord_parser.parse_all()
    
    # Save parsed data
    discord_parser.save_parsed_data(conversations, args.output)
    
    # Print statistics if requested
    if args.stats:
        stats = discord_parser.get_summary_stats(conversations)
        print("\n=== Summary Statistics ===")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Total OpenAI-compatible messages: {stats['total_openai_messages']}")
        print(f"Average messages per conversation: {stats['avg_messages_per_conversation']:.2f}")
        print("\nConversations by type:")
        for chat_type, count in stats['conversations_by_type'].items():
            print(f"  {chat_type}: {count}")


if __name__ == '__main__':
    main()

