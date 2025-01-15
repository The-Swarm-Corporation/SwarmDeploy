import requests
import json

# Define the API endpoint
callable_name = (
    "SequentialWorkflow"  # Replace with the actual callable name
)
swarm_id = "f0d58e55-9d96-4268-bf67-51a5074be5d9"  # Replace with the actual swarm ID
url = f"http://localhost:8000/v1/swarms/completions/{callable_name}/{swarm_id}"

# Define the payload
payload = {
    "task": "What are the terms that are most favorable for a startup founder when raising a preseed"
}

# Make the POST request
response = requests.post(url, json=payload)

print(json.dumps(response.json()))
