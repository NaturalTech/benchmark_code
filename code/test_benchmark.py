import requests

SENTILECTO_URL = "http://dev.natural.do/api/v0.9/deep-structure/%s/%d"

def generate_url(text, apikey):
    return SENTILECTO_URL % (text, apikey)
url_request = generate_url("El perro come comida", 123456)
r = requests.get(url_request)
print(r.json())

