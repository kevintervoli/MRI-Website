import requests
import base64

# Open the image file and read its contents as binary data
with open("./dataset/images/64.png", "rb") as image_file:
    # Encode the binary data as base64
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Define the payload containing the base64-encoded image
payload = {"image": encoded_image}

# Send a POST request to the Flask API endpoint with appropriate content type
headers = {'Content-Type': 'application/json'}
response = requests.post("http://127.0.0.1:5000/MRITest", json=payload, headers=headers)

# Print the response
print(response.text)
