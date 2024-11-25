import openai
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Define model parameters
model = "gpt-3.5-turbo"
temperature = 0.3
MAX_TOKENS = 500

def get_response_from_message_updated(message_content):
    try:
        full_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Return the answer as a single character correlating to the answer.",
            },
            {"role": "user", "content": message_content},
        ]

        response_content = ""

        completion = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            messages=full_messages,
            max_tokens=MAX_TOKENS,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.get('content'):
                content = chunk.choices[0].delta['content']
                print(content, end='', flush=True)
                response_content += content

        print(f"\nResponse: {response_content}")

        return response_content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
message = "Two quantum states with energies E1 and E2 have lifetimes of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\nA) 10^-4 eV\nB) 10^-11 eV\nC) 10^-8 eV\nD) 10^-9 eV"

get_response_from_message_updated(message)
