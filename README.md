# ReplAI
Never ghost anyone again.

## Message Parsers

This repository contains parsers for converting various messaging platform exports into structured formats for AI training and analysis.

### Available Parsers

1. **Discord Parser** (`discord_parser.py`)
   - Converts Discord conversation exports into structured format
   - Outputs OpenAI-compatible messages with roles
   - Includes full metadata (reactions, attachments, embeds, stickers)
   - Supports JSON, CSV, and OpenAI fine-tuning formats
   - [Full Documentation](DISCORD_PARSER.md)

2. **Instagram Parser** (`data/parse_instagram_messages.py`)
   - Parses Instagram message exports
   - Handles Instagram's encoding issues
   - Time-based filtering support
   - [Full Documentation](data/INSTAGRAM.md)

3. **iMessage Parser** (`imsg_parser.py`)
   - Converts iMessage HTML exports to OpenAI format
   - Simple command-line interface

### Quick Start

#### Discord
```bash
python discord_parser.py /path/to/discord/export --user-id YOUR_DISCORD_USER_ID
```

#### Instagram
```bash
python data/parse_instagram_messages.py /path/to/instagram/export --user-name "YourName"
```

#### iMessage
```bash
python imsg_parser.py messages.html -o output.json
```

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### Output Formats

All parsers support multiple output formats suitable for:
- OpenAI fine-tuning
- Data analysis
- Custom processing

See individual parser documentation for detailed usage and examples.
