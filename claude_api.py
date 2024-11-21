#!/usr/bin/env python3

import anthropic
import os
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
client = anthropic.Anthropic(
    api_key=os.environ.get('CLAUDE_API_KEY')
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)
