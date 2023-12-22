from agent import generate_response
from dotenv import load_dotenv

# load env variables using dotenv
load_dotenv()

# Utilizes Agent to generate response to user queries
while True:
    q = input(">")
    print(generate_response(q))