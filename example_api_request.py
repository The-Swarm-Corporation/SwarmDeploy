import requests
import json

# Define the API endpoint
callable_name = (
    "SequentialWorkflow"  # Replace with the actual callable name
)
swarm_id = "cdf6059e-43c4-4991-9a69-0e317648cb52"  # Replace with the actual swarm ID
url = f"http://localhost:8000/v1/swarms/completions/{callable_name}/{swarm_id}"

# Define the payload
payload = {
    "task": "What are the terms that are most favorable for a startup founder when raising a preseed"
}

# Make the POST request
response = requests.post(url, json=payload)

print(json.dumps(response.json()))
