# Message Parsers for AI Training

Convert your Instagram and iMessage data into OpenAI-compatible chat format for training a model to talk like you. **Both parsers include built-in AES-256 encryption** for secure data storage and uploading to training platforms like Hugging Face.

## Features

### Instagram Parser

- Parses Instagram's exported message.json files
- Handles Unicode encoding issues in Instagram exports
- Converts to OpenAI chat format with proper role assignments for training
- Supports both individual and group conversations
- Filters out non-content messages (like "Liked a message")
- Chronological message ordering
- Time-based filtering (train on specific date ranges)
- **AES-256 encryption for secure data protection**
- Two output modes: combined or separate files

### iMessage Parser

- Parses iMessage HTML exports (from imessage-exporter)
- Supports both individual and group conversations
- Simple, lightweight parser
- Outputs clean OpenAI chat format
- **AES-256 encryption for secure data protection**

## Installation

```bash
pip install -r requirements.txt
```

Or install the required encryption library directly:

```bash
pip install cryptography
```

## Quick Start

### Instagram Messages

#### 1. Generate an encryption key (recommended)

```bash
python3 instagram_parser.py --generate-key
```

Save the displayed key securely in a password manager!

#### 2. Parse and encrypt your messages

```bash
python3 instagram_parser.py /path/to/instagram/messages \
  --user-name "Your Display Name" \
  --encrypt \
  --encryption-key "YOUR_KEY_HERE"
```

Or let it auto-generate a key:

```bash
python3 instagram_parser.py /path/to/instagram/messages \
  --user-name "Your Display Name" \
  --encrypt
```

**⚠️ IMPORTANT: Save the encryption key! You'll need it to decrypt your data.**

---

### iMessage

#### 1. Export your iMessages

First, install the iMessage exporter:

```bash
brew install imessage-exporter
```

Then export your messages to HTML:

```bash
imessage-exporter -f html -o ~/Desktop/imessage
```

This will create HTML files for each conversation in `~/Desktop/imessage/`.

#### 2. Parse iMessage HTML files

Parse a single conversation:

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html -o output.json
```

Parse and encrypt:

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html -o output.json --encrypt
```

Or print to stdout:

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html
```

## Usage

### Instagram Parser

#### Command Line Options

```bash
python3 instagram_parser.py <input_dir> --user-name <name> [options]
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
- `--encrypt` - Encrypt output files (highly recommended for uploading to training platforms)
- `--encryption-key KEY` - Use a specific encryption key (if not provided with --encrypt, generates new key)
- `--generate-key` - Generate a new encryption key and exit

#### Examples

**Basic parsing with encryption:**

```bash
python3 instagram_parser.py ./instagram_data \
  --user-name "John Doe" \
  --encrypt
```

**Combine all conversations into one encrypted file:**

```bash
python3 instagram_parser.py ./instagram_data \
  --user-name "John Doe" \
  --format combined \
  --encrypt
```

**Filter by date range:**

```bash
python3 instagram_parser.py ./instagram_data \
  --user-name "John Doe" \
  --start-time "2024-01-01" \
  --end-time "2024-12-31" \
  --encrypt
```

**Train on recent conversations only:**

```bash
python3 instagram_parser.py ./instagram_data \
  --user-name "John Doe" \
  --start-time "2023-01-01" \
  --format combined \
  --encrypt
```

**Use a pre-generated encryption key:**

```bash
python3 instagram_parser.py ./instagram_data \
  --user-name "John Doe" \
  --format combined \
  --encrypt \
  --encryption-key "xMzE5NjQ3ODkwMTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MA=="
```

---

### iMessage Parser

#### Command Line Options

```bash
python3 imsg_parser.py <input_file> [options]
```

**Required Arguments:**

- `input_file` - Path to HTML file containing iMessage export

**Optional Arguments:**

- `-o, --output FILE` - Output JSON file (default: print to stdout)
- `--pretty` - Pretty print JSON output with indentation
- `--encrypt` - Encrypt output file (requires -o/--output)
- `--encryption-key KEY` - Use a specific encryption key (if not provided with --encrypt, generates new key)
- `--generate-key` - Generate a new encryption key and exit

#### Examples

**Parse and save to file:**

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html -o conversation.json
```

**Parse and print to stdout:**

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html
```

**Pretty print:**

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html --pretty
```

**Parse and encrypt:**

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html -o conversation.json --encrypt
```

**Generate an encryption key:**

```bash
python3 imsg_parser.py --generate-key
```

**Use a specific encryption key:**

```bash
python3 imsg_parser.py ~/Desktop/imessage/+12345678900.html \
  -o conversation.json \
  --encrypt \
  --encryption-key "your-key-here"
```

**Parse multiple conversations with encryption:**

```bash
# Generate a key once
KEY=$(python3 imsg_parser.py --generate-key | tail -1)

# Loop through all HTML files using the same key
for file in ~/Desktop/imessage/*.html; do
  output_name=$(basename "$file" .html).json
  python3 imsg_parser.py "$file" -o "output/$output_name" --encrypt --encryption-key "$KEY"
done

echo "Encryption key: $KEY"
```

#### Output Format

The iMessage parser outputs messages in the same format as the Instagram parser:

```json
[
  {
    "role": "user",
    "content": "Hey, how's it going?"
  },
  {
    "role": "assistant",
    "content": "Good! What are you up to?"
  }
]
```

**Role Assignment:**

- `assistant` - **Your messages** (sender = "Me") - what the model learns to imitate
- `user` - **Everyone else's messages** - the prompts/context

---

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

## How to Get Your iMessage Data

To export your iMessages to HTML format:

### Prerequisites

- **macOS only** - iMessage exporter requires macOS with iMessage enabled
- **Homebrew** - Package manager for macOS ([install here](https://brew.sh))

### Export Steps

1. **Install imessage-exporter:**

```bash
brew install imessage-exporter
```

2. **Export your messages:**

```bash
imessage-exporter -f html -o ~/Desktop/imessage
```

This will create HTML files in `~/Desktop/imessage/` with filenames like:

- Individual chats: `+12345678900.html`
- Group chats: `+12345678900, +10987654321, +11122233344.html`

3. **Grant permissions if needed:**

If prompted, you may need to grant Terminal or imessage-exporter access to your Messages database in System Settings → Privacy & Security → Full Disk Access.

### Export Options

```bash
# Export specific conversation by phone number or email
imessage-exporter -f html -o ~/Desktop/imessage --filter "+12345678900"

# Export within date range
imessage-exporter -f html -o ~/Desktop/imessage --after "2023-01-01" --before "2024-01-01"

# See all options
imessage-exporter --help
```

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

---

## 🔐 Encryption Guide

### Why Encrypt?

When training models on platforms like Hugging Face, your data files may be:

- Stored on remote servers
- Potentially accessible by platform administrators
- Subject to data breaches

Encryption ensures your personal conversations remain private, even if the encrypted files are leaked.

### How It Works

The script uses **Fernet** (symmetric encryption) which provides:

- AES-256 encryption
- Built-in authentication (prevents tampering)
- Simple key management

### Encryption Workflow

#### Step 1: Generate a Key (Optional but Recommended)

```bash
python3 instagram_parser.py --generate-key
```

Outputs:

```
Generated encryption key (save this securely!):
xMzE5NjQ3ODkwMTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MA==

Use this key with: --encryption-key YOUR_KEY
⚠️  Keep this key safe! You'll need it to decrypt your data.
```

**Save this key in a password manager!**

#### Step 2: Parse and Encrypt

With your own key:

```bash
python3 instagram_parser.py ~/instagram/messages \
  --user-name "krish" \
  --format combined \
  --encrypt \
  --encryption-key "YOUR_KEY_HERE"
```

Or let it auto-generate:

```bash
python3 instagram_parser.py ~/instagram/messages \
  --user-name "krish" \
  --format combined \
  --encrypt
```

#### Step 3: Upload to Hugging Face

Your encrypted files are now safe to upload! Even if someone accesses them, they can't read the content without your key.

#### Step 4: Decrypt for Training

On your training platform, decrypt the files programmatically:

```python
from cryptography.fernet import Fernet
import json

# Read and decrypt the file
with open('all_messages_combined.json', 'rb') as f:
    encrypted_data = f.read()

cipher = Fernet(b'YOUR_KEY_HERE')
decrypted_data = cipher.decrypt(encrypted_data)

# Parse the decrypted JSON
data = json.loads(decrypted_data)
messages = data['messages']
```

#### Step 5: Decrypt Locally (if needed)

To decrypt files on your local machine:

```bash
python3 decrypt_messages.py encrypted_file.json \
  --encryption-key "YOUR_KEY" \
  --output decrypted.json
```

### Security Best Practices

#### ✅ DO:

- **Save your encryption key in a password manager** (1Password, Bitwarden, LastPass)
- Store the key separately from the encrypted files
- Use environment variables when deploying to training platforms
- Generate a new key for each dataset
- Keep encrypted backups

#### ❌ DON'T:

- Commit encryption keys to Git repositories
- Store keys in the same location as encrypted files
- Share keys over insecure channels (email, Slack, etc.)
- Use the same key for multiple datasets
- Store keys in plain text files

### Key Storage Recommendations

**Option 1: Password Manager (Recommended)**

Store in 1Password, Bitwarden, or LastPass with a note like:

```
Instagram Messages Encryption Key - Generated 2025-01-08
xMzE5NjQ3ODkwMTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MA==
```

**Option 2: Environment Variable**

```bash
# In your ~/.zshrc or ~/.bashrc
export INSTAGRAM_ENCRYPTION_KEY="xMzE5NjQ3ODkwMTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MA=="
```

Then use:

```bash
python3 instagram_parser.py ./data \
  --user-name "Your Name" \
  --encrypt \
  --encryption-key "$INSTAGRAM_ENCRYPTION_KEY"
```

**Option 3: Encrypted Notes**

Store in Apple Notes (with encryption), Notion, or any encrypted note-taking app.

---

## Time Filtering

Filter messages by date to train on specific time periods:

### Supported Date Formats

- **ISO Date**: `"2024-01-15"` or `"2024-01-15T10:30:00"`
- **Unix timestamp (seconds)**: `"1705334400"`
- **Unix timestamp (milliseconds)**: `"1705334400000"`

### Use Cases

```bash
# Only recent conversations (last year)
python3 instagram_parser.py ./messages \
  --user-name "You" \
  --start-time "2024-01-01" \
  --encrypt

# Specific time period (college years)
python3 instagram_parser.py ./messages \
  --user-name "You" \
  --start-time "2019-09-01" \
  --end-time "2023-05-31" \
  --encrypt

# Everything up to a certain date
python3 instagram_parser.py ./messages \
  --user-name "You" \
  --end-time "2022-12-31" \
  --encrypt
```

**Note:** All dates are interpreted as UTC (to match Instagram's timestamp format). Filters are inclusive.

---

## Features Explained

### Unicode Encoding Fix

Instagram exports have encoding issues where UTF-8 characters (emojis, special characters) are double-encoded. The script automatically fixes this.

### Message Filtering

Automatically skips:

- Messages without content
- Generic messages like "Liked a message"
- Empty messages
- Reactions without text

### What Gets Included

- ✅ Actual text messages with proper emoji support
- ✅ Messages with links
- ✅ All chronologically ordered conversations

---

## Troubleshooting

### "No message.json files found"

- Make sure you're pointing to the correct directory (usually `instagram-data/messages/inbox/`)
- Check that the Instagram export is in JSON format (not HTML)
- Files should be named `message.json` (sometimes `message_1.json`, `message_2.json`, etc.)

### Missing required argument

- The `--user-name` argument is required - you must specify your Instagram display name
- Use the exact name as it appears in your Instagram messages
- Names are case-insensitive

### Encoding issues with emojis

- The script should handle this automatically
- If you still see issues, check that your terminal supports UTF-8
- Check the JSON file directly in a text editor

### "cryptography library is required"

Install it:

```bash
pip install cryptography
```

### "Decryption failed. Invalid key"

- Double-check you're using the exact key (no extra spaces/characters)
- Make sure you're using the key that was generated/used during encryption
- Verify the file hasn't been corrupted

### "Invalid encryption key format"

- Ensure the key is base64-encoded
- The key should be exactly 44 characters long (ending in ==)
- Don't modify the key in any way

---

## Complete End-to-End Example

```bash
# 1. Generate and save a key
python3 instagram_parser.py --generate-key
# Save the key displayed!

# 2. Parse and encrypt your Instagram messages
python3 instagram_parser.py ~/Downloads/instagram/messages \
  --user-name "krish" \
  --format combined \
  --start-time "2023-01-01" \
  --encrypt \
  --encryption-key "YOUR_KEY_HERE" \
  --output-dir ./encrypted_training_data

# 3. Upload encrypted files to Hugging Face
# Files in ./encrypted_training_data/ are now encrypted and safe to upload

# 4. In your training script on Hugging Face:
# Decrypt programmatically using the key (see Python example above)

# 5. Later: decrypt locally if needed
python3 decrypt_messages.py encrypted_training_data/all_messages_combined.json \
  --encryption-key "YOUR_KEY_HERE" \
  --output readable_data.json
```

---

## Tips

1. **Always encrypt** when uploading to training platforms
2. **Use combined mode** when creating a single training dataset
3. **Use time filters** to focus on specific eras of your communication style
4. **Start with separate mode** to inspect individual conversations first
5. **Save encryption keys securely** - use a password manager
6. **Check encoding** - Emojis should appear correctly (😍, 🎉, ❤️)
7. **Verify output** before using for training/analysis
8. **Privacy first** - Remember to review and redact any sensitive information

---

## Technical Details

- **Encryption Algorithm**: Fernet (AES-128-CBC with HMAC-SHA256)
- **Key Size**: 256 bits (32 bytes)
- **Key Format**: Base64 URL-safe encoding
- **Authentication**: Built-in (prevents tampering)
- **Standard Library**: Uses Python's built-in JSON, datetime modules
- **External Dependency**: cryptography library for encryption
