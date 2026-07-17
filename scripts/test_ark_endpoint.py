import argparse

import httpx
from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--api-key', required=True)
    ap.add_argument('--endpoint', required=True)
    args = ap.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        http_client=httpx.Client(timeout=60.0, trust_env=False),
        max_retries=0,
    )

    resp = client.chat.completions.create(
        model=args.endpoint,
        messages=[{'role': 'user', 'content': 'ping'}],
        temperature=0,
        max_tokens=32,
    )
    print(resp.choices[0].message.content)


if __name__ == '__main__':
    main()
