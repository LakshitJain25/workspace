import requests 
url = 'http://127.0.0.1:5000/api/chat'

payload = {
    "message": "what if NCT05057494 has enrollment of 3000 people ,  uses quadruple blind masking, and uses randomized allocation, with a parallel intervention model?"
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