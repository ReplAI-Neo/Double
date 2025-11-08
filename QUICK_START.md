# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Parse a Folder

```bash
python imsg_parser.py imsg_data/ -o conversations.json --pretty
```

### Parse a Single File

```bash
python imsg_parser.py conversation.html -o output.json --pretty
```

## Output Format

Your output will be a JSON array of conversation objects:

```json
[
  {
    "openai_messages": [...],           // For AI training
    "full_metadata_messages": [...],    // Full message data
    "first_message_timestamp": "...",   // ISO 8601
    "last_message_timestamp": "...",    // ISO 8601
    "recipients": ["..."],              // Who you're talking to
    "num_participants": 2,              // Total people
    "total_messages": 1315,             // Message count
    "source": "imessage",               // Platform
    "chat_type": "direct"               // direct or group
  }
]
```

## Common Tasks

### Use with OpenAI API

```python
import json
import openai

# Load your conversations
with open('conversations.json') as f:
    conversations = json.load(f)

# Use the first conversation's messages
messages = conversations[0]['openai_messages']

# Make API call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages
)
```

### Analyze Your Data

```python
import json

with open('conversations.json') as f:
    conversations = json.load(f)

for conv in conversations:
    print(f"Conversation with: {conv['recipients']}")
    print(f"  Messages: {conv['total_messages']}")
    print(f"  Type: {conv['chat_type']}")
    print(f"  Duration: {conv['first_message_timestamp']} to {conv['last_message_timestamp']}")
```

### Filter Conversations

```python
import json

with open('conversations.json') as f:
    conversations = json.load(f)

# Get only group chats
group_chats = [c for c in conversations if c['chat_type'] == 'group']

# Get conversations with specific person
person_convs = [c for c in conversations if '+16692689561' in c['recipients']]

# Get conversations with 100+ messages
long_convs = [c for c in conversations if c['total_messages'] >= 100]
```

## Schema Reference

### Message Roles

- `assistant` = Your messages
- `user` = Other people's messages

### Chat Types

- `direct` = 1-on-1 conversation
- `group` = Group chat (3+ people)

### Timestamps

All in ISO 8601 format: `2024-10-27T13:42:02`

## Troubleshooting

### No messages found

Make sure your HTML file has `<div class="message">` elements.

### Missing timestamps

Some exports may not include timestamps. The parser will handle this gracefully.

### Multiple conversations not merging

Each HTML file is treated as a separate conversation. Recipients list distinguishes them.

## Next Steps

1. ✅ Parse your data
2. ✅ Validate the output
3. ✅ Use for AI training or analysis
4. ✅ Check out README.md for advanced usage

## Help

```bash
python imsg_parser.py --help
```

