import requests

url = 'http://localhost:5000/question'

question = "Is the lat pulldown considered a stregth training activity, and if so, why?"
data = {"question": question}

response = requests.post(url, json=data)

print(response.content)
print(response.json())