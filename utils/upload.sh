#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable

# process
echo "Step 1/5: Processing conversations..."
bash utils/parse_discord.sh
# bash parse_imessage.sh
# bash parse_instagram.sh
echo "✓ Processing complete"

# merge
echo "Step 2/5: Merging conversations..."
python utils/merge.py data/processed -o data/merged/all_conversations.json
if [ ! -f "data/merged/all_conversations.json" ]; then
    echo "Error: merge failed - output file not found"
    exit 1
fi
echo "✓ Merge complete"

# partition
echo "Step 3/5: Partitioning conversations..."
python utils/partition.py data/merged/all_conversations.json -o data/merged/all_conversations_partitioned.json --max-days 7
if [ ! -f "data/merged/all_conversations_partitioned.json" ]; then
    echo "Error: partition failed - output file not found"
    exit 1
fi
echo "✓ Partition complete"

# filter
echo "Step 4/5: Filtering conversations..."
python utils/filter.py data/merged/all_conversations_partitioned.json data/merged/all_conversations_partitioned_filtered.json \
  --start-date "2022-01-01T00:00:00+00:00" \
  --end-date "2025-11-01T23:59:59+00:00" \
#   --time-of-day morning afternoon night \
#   --num-participants-min 2 \
#   --num-participants-max 10 \
  --my-avg-message-length-min 10 \
  --my-avg-message-length-max 100 \
  --my-turn-proportion-min 10 \
  --my-turn-proportion-max 90 \
  --pretty
if [ ! -f "data/merged/all_conversations_partitioned_filtered.json" ]; then
    echo "Error: filter failed - output file not found"
    exit 1
fi
echo "✓ Filter complete"

# encrypt
echo "Step 5/5: Encrypting conversations..."
# TODO: Add encryption step

echo "All steps completed successfully!"

# upload
# TODO: Add upload step
# hf 