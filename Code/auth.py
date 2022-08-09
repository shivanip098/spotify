import requests
import base64
import json
from secrets import *

# Step 1 - Authorization 
url = "https://accounts.spotify.com/api/token"
headers = {}
data = {}
client_id = ''#request your client ID
client_secret = ''#also request this
user_id = ''#your spotify username

# Encode as Base64
message = f"{client_id}:{client_secret}"
messageBytes = message.encode('ascii')
base64Bytes = base64.b64encode(messageBytes)
base64Message = base64Bytes.decode('ascii')


headers['Authorization'] = f"Basic {base64Message}"
data['grant_type'] = "client_credentials"

r = requests.post(url, headers=headers, data=data)

token = r.json()['access_token']

playlist_id = ''#your specific playlist ID

'''
# Step 2 - Use Access Token to call playlist endpoint

playlistId = "myPlaylistId"
playlistUrl = f"https://api.spotify.com/v1/playlists/{playlistId}"
headers = {
    "Authorization": "Bearer " + token
}

res = requests.get(url=playlistUrl, headers=headers)

print(json.dumps(res.json(), indent=2))




'''