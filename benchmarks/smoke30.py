import requests
r = requests.post('http://127.0.0.1:30000/generate', json={
    'text': 'Sebutkan ibukota Prancis.',
    'sampling_params': {'max_new_tokens': 30, 'temperature': 0.0}}, timeout=120)
print(r.json()['text'][:200])
