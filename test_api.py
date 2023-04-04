import requests
import json

post_data = json.dumps({'prompt': 'solve 5x=13'})
print(json.loads(requests.post("http://0.0.0.0:8000", post_data).text)['response'])