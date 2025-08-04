import requests 
url = 'http://127.0.0.1:5000/api/chat'

payload = {
    "message": "what can you do?"
}

# Optional: headers, e.g., if you want to specify content type as JSON
headers = {
    "Content-Type": "application/json"
}

# Make the POST request with JSON payload
response = requests.post(url, json=payload, headers=headers)

# Print status code and response JSON content
print("Status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except ValueError:
    print("Response content is not in JSON format")