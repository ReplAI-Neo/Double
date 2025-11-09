#!/usr/bin/env python3
"""
Download an encrypted conversation dataset from the Hugging Face Hub,
decrypt the sensitive fields, and save the result as a JSON file in the
project's data directory.

The dataset is expected to have been produced by `upload_dataset.py` after
its conversations were encrypted via `encrypt_conversations.py`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an encrypted dataset from the Hugging Face Hub, decrypt it, "
            "and save the decrypted conversations as JSON."
        )
    )
    parser.add_argument(
        "repo_id",
        help='Hugging Face dataset repository ID (e.g., "username/dataset-name").',
    )
    parser.add_argument(
        "output_path",
        help=(
            "Destination path for the decrypted JSON file. "
            "Can be absolute or relative to current directory."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help='Dataset split to download (default: "train").',
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision (branch, tag, or commit SHA).",
    )
    parser.add_argument(
        "--token",
        help=(
            "Hugging Face access token. "
            "Defaults to the HF_TOKEN environment variable if not provided."
        ),
    )
    parser.add_argument(
        "--encryption-key",
        dest="encryption_key",
        help=(
            "Encryption key used to decrypt the dataset. "
            "Defaults to ENCRYPTION_KEY environment variable or .env file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the decrypted JSON output.",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip calling huggingface_hub.login (useful if already authenticated).",
    )
    return parser.parse_args()

def decrypt_message_data(encrypted_data: str, encryption_key: str) -> str:
    """
    Decrypt a string of encrypted data using Fernet.

    Args:
        encrypted_data: Base64-encoded encrypted string
        encryption_key: Base64-encoded encryption key

    Returns:
        Decrypted string data
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise ImportError(
            "cryptography library is required for decryption. "
            "Install it with: pip install cryptography"
        )

    key_bytes = encryption_key.encode("utf-8")
    cipher = Fernet(key_bytes)

    # Decrypt the data
    encrypted_bytes = encrypted_data.encode("ascii")
    decrypted_bytes = cipher.decrypt(encrypted_bytes)

    return decrypted_bytes.decode("utf-8")


def decrypt_conversations(
    encrypted_conversations: List[Dict[str, Any]], encryption_key: str
) -> List[Dict[str, Any]]:
    """
    Decrypt sensitive fields in conversation data and extract only messages.
    
    Each conversation dict should have encrypted string fields like:
    - openai_messages_encrypted (encrypted JSON string)
    
    Returns only the messages field (renamed from openai_messages).

    Args:
        encrypted_conversations: List of encrypted conversation objects
        encryption_key: Encryption key

    Returns:
        List of dicts with only 'messages' field containing decrypted messages
    """
    decrypted_conversations = []

    for conv in encrypted_conversations:
        # Check if this conversation is encrypted
        if not conv.get("_encrypted"):
            print(
                "Warning: Conversation doesn't appear to be encrypted",
                file=sys.stderr,
            )
            continue

        # Decrypt openai_messages if present (it's an encrypted string)
        if "openai_messages_encrypted" in conv:
            encrypted_string = conv["openai_messages_encrypted"]
            # The encrypted field should be a string containing encrypted JSON
            if isinstance(encrypted_string, str):
                openai_json = decrypt_message_data(encrypted_string, encryption_key)
                messages = json.loads(openai_json)
                # Only output the messages field, renamed
                decrypted_conversations.append({"messages": messages})
            else:
                print(
                    f"Warning: openai_messages_encrypted is not a string: {type(encrypted_string)}",
                    file=sys.stderr,
                )

    return decrypted_conversations


def resolve_output_path(output_arg: str) -> Path:
    """
    Resolve the output path, allowing absolute or relative paths.
    
    Args:
        output_arg: Path string from user
        
    Returns:
        Resolved Path object
    """
    candidate = Path(output_arg).expanduser().resolve()
    return candidate


def read_encryption_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key

    env_key = os.environ.get("ENCRYPTION_KEY")
    if env_key:
        return env_key

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("ENCRYPTION_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value

    raise RuntimeError(
        "Encryption key is required. Provide it via --encryption-key, "
        "set ENCRYPTION_KEY in the environment, or add it to .env."
    )


def resolve_token(explicit_token: str | None) -> str | None:
    return explicit_token or os.environ.get("HF_TOKEN")


def dataset_to_records(dataset) -> List[Dict[str, Any]]:
    """Convert a Hugging Face Dataset object into a list of dictionaries."""
    return [dataset[i] for i in range(len(dataset))]


def main() -> None:
    args = parse_args()

    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = datasets_module.load_dataset
    except ImportError as exc:
        print(
            "Error: The 'datasets' library is required. Install it with "
            "'pip install datasets'.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        hf_module = importlib.import_module("huggingface_hub")
        login = hf_module.login
    except ImportError as exc:
        print(
            "Error: The 'huggingface_hub' library is required. Install it with "
            "'pip install huggingface_hub'.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(
            f"Error: Output file already exists at {output_path}. "
            "Use --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    token = resolve_token(args.token)
    encryption_key = read_encryption_key(args.encryption_key)

    if token and not args.no_login:
        # Avoid modifying git credentials unless explicitly desired.
        login(token=token, add_to_git_credential=False)

    try:
        dataset = load_dataset(
            path=args.repo_id,
            split=args.split,
            revision=args.revision,
            use_auth_token=token,
        )
    except Exception as err:
        print(f"Error: Failed to load dataset {args.repo_id}: {err}", file=sys.stderr)
        sys.exit(1)

    records = dataset_to_records(dataset)
    print(f"Downloaded {len(records)} record(s) from {args.repo_id}:{args.split}")

    # Debug: Check first record structure
    if records:
        print(f"First record keys: {list(records[0].keys())}", file=sys.stderr)
        if "_encrypted" in records[0]:
            print(f"First record _encrypted: {records[0]['_encrypted']}", file=sys.stderr)
        if "openai_messages_encrypted" in records[0]:
            encrypted_val = records[0]["openai_messages_encrypted"]
            print(f"openai_messages_encrypted type: {type(encrypted_val)}", file=sys.stderr)
            if isinstance(encrypted_val, str):
                print(f"openai_messages_encrypted length: {len(encrypted_val)}", file=sys.stderr)
                print(f"openai_messages_encrypted preview: {encrypted_val[:100]}...", file=sys.stderr)

    try:
        decrypted_records = decrypt_conversations(records, encryption_key)
    except Exception as err:
        print(f"Error decrypting conversations: {err}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    with output_path.open("w", encoding="utf-8") as fh:
        if args.pretty:
            json.dump(decrypted_records, fh, indent=2, ensure_ascii=False)
        else:
            json.dump(decrypted_records, fh, ensure_ascii=False)

    total_messages = sum(conv.get("total_messages", 0) for conv in decrypted_records)
    print(f"✓ Decrypted dataset written to {output_path}")
    print(f"  Total conversations: {len(decrypted_records)}")
    print(f"  Total messages: {total_messages}")


if __name__ == "__main__":
    main()


