#!/usr/bin/env python3
import secrets, string, argparse

def generate_api_key(prefix="rag", length=32):
    alphabet = string.ascii_letters + string.digits
    return f"{prefix}_{''.join(secrets.choice(alphabet) for _ in range(length))}"

def generate_secret_key(bytes_=32):
    return secrets.token_hex(bytes_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["api-key", "secret-key"], default="api-key")
    parser.add_argument("--prefix", default="rag")
    args = parser.parse_args()
    print(generate_api_key(args.prefix) if args.type == "api-key" else generate_secret_key())
