# Quick Start Guide

## Basic Usage

Once you have your Instagram data export:

### 1. Single Conversation Mode (Default)

Creates one JSON file per conversation:

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages --user-name "Your Display Name"
```

**Output:** `output/conversation_name.json` for each chat

### 2. Combined Mode

Merges all conversations into one chronological file:

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages \
  --user-name "Your Name" \
  --format combined
```

**Output:** `output/all_messages_combined.json`

### 3. Filter by Date Range

Only include messages from specific time periods:

```bash
# Only recent messages (2024 onwards)
python3 parse_instagram_messages.py /path/to/instagram/messages \
  --user-name "Your Name" \
  --start-time "2024-01-01"

# Specific year only
python3 parse_instagram_messages.py /path/to/instagram/messages \
  --user-name "Your Name" \
  --start-time "2023-01-01" \
  --end-time "2023-12-31"
```

### 4. Custom Output Directory

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages \
  --user-name "Your Name" \
  --output-dir ./my_chats
```

## Complete Example

```bash
# Parse messages from 2023 onwards, combine into one file
python3 parse_instagram_messages.py ~/Downloads/instagram-data/messages \
  --user-name "Your Name" \
  --format combined \
  --start-time "2023-01-01" \
  --output-dir ./training_data
```

## What Gets Filtered Out?

The script automatically skips:

- ❌ "Liked a message" (reactions without text)
- ❌ Empty messages
- ❌ Messages without content
- ✅ Keeps actual text messages with proper emoji support

## Output Format

Each output file looks like:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "friend's message here"
    },
    {
      "role": "assistant",
      "content": "your response here"
    },
    {
      "role": "user",
      "content": "another friend message"
    },
    {
      "role": "assistant",
      "content": "your response"
    }
  ]
}
```

## Role Mapping (For Training)

- **assistant** → **Your messages** (specified by `--user-name`) - what the model learns to imitate
- **user** → **Everyone else's messages** - the prompts/context

This format is designed for training a model to replicate your conversation style. Your messages become the assistant's responses that the model learns from.

## Time Filtering

Filter messages by date to train on specific time periods:

### Supported Date Formats

- **ISO Date**: `"2024-01-15"` or `"2024-01-15T10:30:00"`
- **Unix timestamp (seconds)**: `"1705334400"`
- **Unix timestamp (milliseconds)**: `"1705334400000"`

### Use Cases

```bash
# Only recent conversations (last year)
python3 parse_instagram_messages.py ./messages --user-name "You" --start-time "2024-01-01"

# Specific time period (college years)
python3 parse_instagram_messages.py ./messages --user-name "You" \
  --start-time "2019-09-01" --end-time "2023-05-31"

# Everything up to a certain date
python3 parse_instagram_messages.py ./messages --user-name "You" --end-time "2022-12-31"
```

**Note:** All dates are interpreted as UTC (to match Instagram's timestamp format). Filters are inclusive.

## Tips

1. **Start with separate mode** to inspect individual conversations
2. **Use combined mode** when creating a training dataset
3. **Use time filters** to focus on specific eras of your communication style
4. **Check encoding** - Emojis should appear correctly (😍, 🎉, ❤️)
5. **Verify output** before using for training/analysis

## Common Issues

### "No message.json files found"

- Make sure you're pointing to the messages directory (usually `instagram-data/messages/inbox/`)
- Check files are named `message.json` (sometimes `message_1.json`, `message_2.json` etc.)

### Missing required argument

- The `--user-name` argument is required - you must specify your Instagram display name
- Use the exact name as it appears in your Instagram messages
- Names are case-insensitive

### Wrong person as "assistant"

- Double-check you're using the correct name with `--user-name`
- Remember: YOUR messages become "assistant" (what the model learns to imitate)

### Emojis look broken

- This should be fixed automatically
- If not, ensure your terminal supports UTF-8
- Check the JSON file directly in a text editor
