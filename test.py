import requests

def send_query(user_input: str, user_id: str = "test_user"):
    response = requests.post(
        "http://localhost:8000/chat",
        json={"user_input": user_input, "user_id": user_id}  # Send as JSON body
    )
    return response.json()

# Example usage
print(send_query("What are Concordia's CS admission requirements?"))