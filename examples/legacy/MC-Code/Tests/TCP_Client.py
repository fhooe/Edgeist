import socket

# Define the server IP address and port
SERVER_IP = '10.42.1.30'  # Server IP (change it to your W5500 server IP)
SERVER_PORT = 1000  # Port the server is listening on


# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:

    # Connect to the server
    client_socket.connect((SERVER_IP, SERVER_PORT))
    print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")

    response = client_socket.recv(1024)  # Receive up to 1024 bytes from the server
    print(f"Server response: {response.decode()}")  # Decode the response to string

    # Send the "Hello, World!" message to the server
    message = "Hello, World!\0"
    x = message.encode()
    client_socket.send(message.encode())  # Encoding message to bytes

    # Receive the server's response
    response = client_socket.recv(1024)  # Receive up to 1024 bytes from the server
    print(f"Server response: {response.decode()}")  # Decode the response to string

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the connection
    client_socket.close()
    print("Connection closed.")
