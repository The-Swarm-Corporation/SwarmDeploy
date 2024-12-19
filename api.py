import requests

# Define the API endpoint
callable_name = "SequentialWorkflow"  # Replace with the actual callable name
swarm_id = "15a112e2-d7d2-4f16-83ff-aadc0e789e49"  # Replace with the actual swarm ID
url = f"http://localhost:8000/v1/swarms/completions/{callable_name}/{swarm_id}"

# Define the payload
payload = {
    "task": "What are the ways vcs and founders interact in the term sheet? What should they look for?",  # Replace with your task
}

# Make the POST request
response = requests.post(url, json=payload)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
