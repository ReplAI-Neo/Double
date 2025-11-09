#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Process conversations
echo "Step 1/6: Processing conversations..."
# bash utils/parse_discord.sh
# bash utils/parse_imessage.sh
# bash utils/parse_instagram.sh
echo "✓ Processing complete"

# Merge all processed conversations
echo "Step 2/6: Merging conversations..."
python utils/merge.py data/processed -o data/merged/all_conversations.json
if [ ! -f "data/merged/all_conversations.json" ]; then
    echo "Error: merge failed - output file not found"
    exit 1
fi
echo "✓ Merge complete"

# Partition into training chunks
echo "Step 3/6: Partitioning conversations..."
python utils/partition.py data/merged/all_conversations.json -o data/merged/all_conversations_partitioned.json --max-days 2
if [ ! -f "data/merged/all_conversations_partitioned.json" ]; then
    echo "Error: partition failed - output file not found"
    exit 1
fi
echo "✓ Partition complete"

# Filter conversations
echo "Step 4/6: Filtering conversations..."
python utils/filter.py data/merged/all_conversations_partitioned.json data/merged/all_conversations_partitioned_filtered.json \
  --start-date "2023-01-01T00:00:00+00:00" \
  --end-date "2025-11-01T23:59:59+00:00" \
  --source discord imessage instagram \
  --pretty
#   --time-of-day morning afternoon night
#   --num-participants-min 2
#   --num-participants-max 10
#   --my-turn-proportion-min 10 \
#   --my-turn-proportion-max 90 \
if [ ! -f "data/merged/all_conversations_partitioned_filtered.json" ]; then
    echo "Error: filter failed - output file not found"
    exit 1
fi
echo "✓ Filter complete"

# Style-based selection
echo "Step 5/6: Selecting stylistically aligned snippets..."
python utils/style_filter.py \
  data/merged/all_conversations_partitioned_filtered.json \
  data/merged/all_conversations_style_selected.json \
  --cache-dir data/style_cache \
  --annotations-file data/merged/style_annotations.jsonl
if [ ! -f "data/merged/all_conversations_style_selected.json" ]; then
    echo "Error: style selection failed - output file not found"
    exit 1
fi
echo "✓ Style selection complete"

# Encrypt the data (auto-generates encryption key)
echo "Step 6/6: Encrypting conversations..."
python utils/encrypt_conversations.py --encrypt data/merged/all_conversations_style_selected.json -o data/merged/all_conversations_encrypted.json
if [ ! -f "data/merged/all_conversations_encrypted.json" ]; then
    echo "Error: encryption failed - output file not found"
    exit 1
fi
echo "✓ Encryption complete"

# Upload to Hugging Face as a dataset
# NOTE: Change "Stephen-Xie/chat-dataset" to your own Hugging Face repository ID
HF_TOKEN_VALUE=$(env | grep "^HF_TOKEN=" | cut -d= -f2- || echo "")
if [ -z "$HF_TOKEN_VALUE" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi
echo "Uploading dataset to Hugging Face..."
env HF_TOKEN="$HF_TOKEN_VALUE" python utils/upload_dataset.py \
  data/merged/all_conversations_encrypted.json \
  --repo-id Stephen-Xie/chat-dataset-2 \
  --token "$HF_TOKEN_VALUE"
echo "✓ Upload complete"

echo "All steps completed successfully!"