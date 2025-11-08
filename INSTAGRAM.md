# Instagram Message Parser

Convert your Instagram message data into OpenAI-compatible chat format for training a model to talk like you.

## Features

- Parses Instagram's exported message.json files
- Handles Unicode encoding issues in Instagram exports
- Converts to OpenAI chat format with proper role assignments for training
- Supports both individual and group conversations
- Filters out non-content messages (like "Liked a message")
- Chronological message ordering
- Time-based filtering (train on specific date ranges)
- Two output modes: combined or separate files

## Installation

No additional dependencies required! Uses only Python standard library.

## Usage

### Basic Usage

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages --user-name "Your Display Name"
```

This will:

- Find all `message.json` files in the directory
- Parse each conversation
- Save separate JSON files for each conversation in the `output/` directory

### Command Line Options

```bash
python3 parse_instagram_messages.py <input_dir> --user-name <name> [options]
```

**Required Arguments:**

- `input_dir` - Root directory containing your Instagram message folders
- `--user-name NAME` - Your display name as it appears in Instagram messages (case-insensitive)

**Optional Arguments:**

- `--format {combined|separate}` - Output format (default: separate)
  - `combined` - All conversations in one file
  - `separate` - One file per conversation
- `--output-dir DIR` - Directory to save output files (default: "output")
- `--start-time DATE` - Only include messages from this date onward (format: YYYY-MM-DD or unix timestamp)
- `--end-time DATE` - Only include messages up to this date (format: YYYY-MM-DD or unix timestamp)

### Examples

**Basic parsing:**

```bash
python3 parse_instagram_messages.py ./instagram_data --user-name "John Doe"
```

**Combine all conversations into one file:**

```bash
python3 parse_instagram_messages.py ./instagram_data --user-name "John Doe" --format combined
```

**Filter by date range (e.g., only 2024 messages):**

```bash
python3 parse_instagram_messages.py ./instagram_data --user-name "John Doe" \
  --start-time "2024-01-01" --end-time "2024-12-31"
```

**Train on recent conversations only:**

```bash
python3 parse_instagram_messages.py ./instagram_data --user-name "John Doe" \
  --start-time "2023-01-01"
```

**Complete example with all options:**

```bash
python3 parse_instagram_messages.py ./instagram_data \
  --user-name "John Doe" \
  --format combined \
  --start-time "2023-01-01" \
  --output-dir ./training_data
```

## How to Get Your Instagram Data

To download your Instagram data from Meta Accounts Center:

1. **Open Instagram** and go to your profile by tapping your profile picture in the bottom right corner
2. **Access the menu** by tapping the three horizontal lines (hamburger menu) in the top right corner
3. **Navigate to Accounts Center** by tapping "Accounts Center"
4. **Go to your information and permissions** and select "Download your information"
5. **Initiate the export** by tapping "Download or transfer information"
6. **Select your profile** and tap "Next"
7. **Choose what to download** - You can download "All available information" or select specific types
8. **Choose a destination** - Select "Export to device" and tap "Next"
9. **Select your file options:**
   - **Format:** Choose **JSON** (for use with this script)
   - **Date range:** Select the time period you want to download
   - **Media quality:** Adjust the quality for photos and videos
10. **Submit your request** by tapping "Create files" or "Start export"
11. **Wait for the download** - You'll be notified via email when it's ready (can take a few hours to days)
12. **Extract the zip file** and look for the `messages` or `inbox` folder

## Output Format

The script converts messages to this format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hey, how's it going?"
    },
    {
      "role": "assistant",
      "content": "Good! What are you up to?"
    },
    {
      "role": "user",
      "content": "What are you working on?"
    },
    {
      "role": "assistant",
      "content": "Just coding some cool stuff!"
    }
  ]
}
```

**Role Assignment for Training:**

- `assistant` - **Your messages** (the specified --user-name) - this is what the model learns to imitate
- `user` - **Everyone else's messages** - these are the prompts/context the model responds to

This role assignment is designed for training a model to talk like you. Your messages become the "assistant" responses that the model learns to generate.

## Features Explained

### Unicode Encoding Fix

Instagram exports have encoding issues where UTF-8 characters (emojis, special characters) are double-encoded. The script automatically fixes this.

### Message Filtering

Automatically skips:

- Messages without content
- Generic messages like "Liked a message"
- Empty messages

### Metadata Preservation

While the main output uses clean OpenAI format, the script internally tracks:

- Original sender name
- Timestamp (milliseconds)
- Conversation name

You can modify the script to include this in the output if needed.

## Example Directory Structure

**Input:**

```
instagram_data/
├── aarush_g/
│   └── message.json
├── john_doe/
│   └── message.json
└── group_chat_xyz/
    └── message.json
```

**Output (separate mode):**

```
output/
├── aarush_g.json
├── john_doe.json
└── group_chat_xyz.json
```

## Tips

1. **Large datasets**: If you have many conversations, use `separate` mode first to inspect individual conversations
2. **Training data**: Use `combined` mode to create a single training dataset
3. **Privacy**: Remember to review and redact any sensitive information before sharing or using the data

## Troubleshooting

**"No message.json files found"**

- Make sure you're pointing to the correct directory
- Check that the Instagram export is in JSON format (not HTML)

**Encoding issues with emojis**

- The script should handle this automatically
- If you still see issues, check that your terminal supports UTF-8

**Missing messages**

- Messages without text content (photos only, reactions) are filtered out
- Check the original JSON file to verify the content exists
