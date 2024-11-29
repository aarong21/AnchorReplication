import anthropic
import os
# from dotenv import load_dotenv, find_dotenv


# load_dotenv(find_dotenv())
client = anthropic.Anthropic(
    api_key='YOUR_KEY'
)

def run_claude(prompt):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user", "content": prompt
            },
            {
                "role": "user",
                "content": "Return only a single character 'A', 'B', 'C', or 'D' corresponding to the answer you found."
            }
        ]
    )
    return message.content[0].text
