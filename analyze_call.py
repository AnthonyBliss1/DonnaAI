import requests
import os
import json

BLAND_AI_API_KEY = os.getenv('BLAND_AI_API_KEY')

call_id = "944cc7cc-a9cf-465c-8e84-9e9367d61b34"

url = f"https://api.bland.ai/v1/calls/{call_id}/analyze"

payload = {
    "goal": "Make a restaurant reservation",
    "questions": [
        ["Was the reservation confirmed?", "boolean"],
        ["What date and time was the reservation confirmed for?", "string"],
        ["How many people is the reservation for?", "integer"]
    ]
}

headers = {
    "authorization": BLAND_AI_API_KEY,
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
    print(json.dumps(response.json(), indent=2))  # Pretty print the JSON response
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    if response.text:
        print(f"Response content: {response.text}")