import requests
base="http://127.0.0.1:5000/paraphrase"
res=requests.get(base,{"model":"GPT2","text":"tets"})
print(res.json())