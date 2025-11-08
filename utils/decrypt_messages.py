#!/usr/bin/env python3
"""
Decrypt message files that were encrypted with instagram_parser.py or imsg_parser.py.

Usage:
    python3 decrypt_messages.py <encrypted_file> --encryption-key YOUR_KEY
    python3 decrypt_messages.py <encrypted_file> --encryption-key YOUR_KEY --output decrypted.json
"""

import argparse
import sys
from pathlib import Path

from encryption_utils import decrypt_file


def main():
    parser = argparse.ArgumentParser(
        description="Decrypt encrypted Instagram or iMessage files"
    )
    parser.add_argument("input_file", help="Path to the encrypted file")
    parser.add_argument(
        "--encryption-key",
        required=True,
        help="Encryption key (base64-encoded) used to encrypt the file",
    )
    parser.add_argument(
        "--output",
        help="Output path for decrypted file (default: adds .decrypted to filename)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else None

    try:
        result_path = decrypt_file(input_path, args.encryption_key, output_path)
        print(f"✓ Successfully decrypted to: {result_path}")
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
