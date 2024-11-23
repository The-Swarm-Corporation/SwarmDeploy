import requests

# Define the API endpoint
callable_name = (
    "SequentialWorkflow"  # Replace with the actual callable name
)
swarm_id = "400b1580-036a-4e70-9ce1-72290c099f52"  # Replace with the actual swarm ID
url = f"http://localhost:8000/v1/swarms/completions/{callable_name}/{swarm_id}"

# Define the payload
payload = {
    "task": "Lets encourage cubans to start to a revolution and topple the communist government",
}

# Make the POST request
response = requests.post(url, json=payload)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
