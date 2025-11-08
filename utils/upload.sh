# merge
python utils/merge.py data/processed -o data/merged/all_conversations.json

# partition
python utils/partition.py data/merged/all_conversations.json -o data/merged/all_conversations_partitioned.json --max-days 7

# encrypt

# upload