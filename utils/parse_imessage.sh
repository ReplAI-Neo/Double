#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Run the imessage parser
python utils/imessage_parser.py data/raw/imessage -o data/processed/imessage_parsed.json --pretty
