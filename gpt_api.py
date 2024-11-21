import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json
import prompts
client = OpenAI()

# pip freeze > requirements.txt
# add new dependencies to requirements.txt


# load env file
_ = load_dotenv(find_dotenv())
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)


# create a model
model = "gpt-4o-mini"
temperature = 0.3 # makes model more deterministic and less random
MAX_TOKENS = 500
topic = ""

# Load messages from a JSON file
messages_file = "gradLevelReasoning.json"  # Path to your JSON file
output_file = "responses.txt"  # Path to the output text file

def get_responses_from_messages(file_path, output_path):
    try:
        with open(file_path, "r") as f:
            # Load the list of messages
            messages_data = json.load(f)

        with open(output_path, "w") as out_file:
            for i, message in enumerate(messages_data):
                print(f"\nProcessing message {i+1}/{len(messages_data)}: {message['content']}\n")

                # Construct the full message list
                full_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    message
                ]

                # Get response from the model
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=full_messages,
                    max_tokens=MAX_TOKENS
                )

                # Extract the assistant's response
                response_content = completion.choices[0].message.content
                print(f"Response {i+1}: {response_content}")

                # Write the response to the output file
                out_file.write(f"Message {i+1}: {message['content']}\n")
                out_file.write(f"Response {i+1}: {response_content}\n")
                out_file.write("____________________________________________________________________________________________")
                out_file.write("")

        print(f"\nAll responses have been written to {output_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
get_responses_from_messages(messages_file, output_file)

# completion = client.chat.completions.create(
#     model=model,
#     temperature=temperature,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Why are dogs blue?"
#         }
#     ],
#     max_tokens=MAX_TOKENS
# )

# print(completion.choices[0].message.content)







# sentence1 = "What is the coloro fo the sky?"
# sentence2 = "Why are dogs green?"

# # prompts
# # system_message = prompts.system_message
# # prompt = prompts.generate_prompt(sentence1)

# messages = [sentence1,
#             sentence2]

# # helper function
# def get_response():
#     completion = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=MAX_TOKENS
#     )
#     return completion.choices[0].message


# print(get_response())