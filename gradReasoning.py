import requests
headers = {"Authorization": "Bearer $HF_TOKEN"}
API_URL = "https://datasets-server.huggingface.co/is-valid?dataset=Idavidrein/gpqa"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
print(data)
