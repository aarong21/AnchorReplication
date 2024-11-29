import os
import openai
from dotenv import load_dotenv, find_dotenv
import json

# Load environment variables
load_dotenv(find_dotenv())

# Set OpenAI API key
# openai.api_key = os.environ.get('OPENAI_API_KEY')

client = openai.OpenAI(api_key=os.environ.get('OPEN_AI_API_KEY'))

# Create a model
model = "gpt-4o"
temperature = 0.3  # Makes the model more deterministic and less random
MAX_TOKENS = 500
topic = ""

def run_model(prompt):
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "You are a helpful assistant. Return only a single character 'A', 'B', 'C', or 'D' corresponding to the answer you find."},
          {
              "role": "user",
              "content": prompt
          }
      ]
  )

  return completion.choices[0].message.content


# this is for multiple prompts  NOT NEEDED
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # Load messages from a JSON file
# messages_file = "gradLevelReasoning.json"  # Path to your JSON file
# output_file = "responses.txt"  # Path to the output text file

# def get_responses_from_messages(file_path, output_path):
#     try:
#         with open(file_path, "r") as f:
#             # Load the list of messages
#             messages_data = json.load(f)

#         with open(output_path, "w") as out_file:
#             for i, message in enumerate(messages_data):
#                 print(f"\nProcessing message {i+1}/{len(messages_data)}: {message['content']}\n")

#                 # Construct the full message list
#                 full_messages = [
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant. Return the answer as a single char correlating to the answer.",
#                     },
#                     {"role": "user", "content": message['content']},
#                 ]

#                 # Get response from the model
#                 completion = openai.ChatCompletion.create(
#                     model=model,
#                     temperature=temperature,
#                     messages=full_messages,
#                     max_tokens=MAX_TOKENS,
#                 )

#                 # Extract the assistant's response
#                 response_content = completion.choices[0].message.content.strip()
#                 print(f"Response {i+1}: {response_content}")

#                 # Write the response to the output file
#                 out_file.write(f"Message {i+1}: {message['content']}\n")
#                 out_file.write(f"Response {i+1}: {response_content}\n")
#                 out_file.write("____________________________________________________________________________________________\n")

#         print(f"\nAll responses have been written to {output_path}.")

#     except Exception as e:
#         print(f"An error occurred: {e}")

st = {"role": "user", "content": "trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1. 1 was treated with pyridinium chlorochromate, forming product 2. 3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3. how many carbon atoms are there in product 3? \nA) 10, \nB) 11, \nC) 12, \nD) 14"}

def get_response_from_message_updated(message_content):
    try:
        # Construct the full message list
        full_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Return the answer as a single character correlating to the answer.",
            },
            {"role": "user", "content": message_content},
        ]

        # Get response from the model
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            messages=full_messages,
            max_tokens=MAX_TOKENS,
        )

        # Extract the assistant's response
        response_content = completion.choices[0].message.content.strip()
        print(f"Response: {response_content}")

        return response_content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


get_response_from_message_updated(st['content'])
