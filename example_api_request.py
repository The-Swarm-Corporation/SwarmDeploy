import requests
import json

# Define the API endpoint
callable_name = (
    "SequentialWorkflow"  # Replace with the actual callable name
)
swarm_id = "f082e01f-1957-45a4-98a9-7be9bf7d83a6"  # Replace with the actual swarm ID
url = f"http://localhost:8000/v1/swarms/completions/{callable_name}/{swarm_id}"

# Define the payload
payload = {
    "task": "What are the terms that are most favorable for a startup founder when raising a preseed"
}

# Make the POST request
response = requests.post(url, json=payload)

print(json.dumps(response.json()))
