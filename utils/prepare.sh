python utils/download_and_decrypt_dataset.py Stephen-Xie/chat-dataset-gap-2 $(pwd)/train/LLaMA-Factory/data/replai.json --encryption-key "$ENCRYPTION_KEY_1"

python utils/fix_conversation_format.py  train/LLaMA-Factory/data/replai.json data/training-data/replai.json