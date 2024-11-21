# with open('dataset.txt', 'r') as file:
#     text = file.read()

system_message = """
You are a helpful assistant.
"""

def generate_prompt(sentence):
    prompt = f"""
    
        {sentence}
    
    """
    return prompt