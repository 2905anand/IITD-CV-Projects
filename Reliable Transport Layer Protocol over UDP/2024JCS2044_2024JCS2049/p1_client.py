import socket
import argparse
import json

# Constants
MSS = 1400  # Maximum Segment Size



def receive_file(server_ip, server_port):
    """
    Receive the file from the server with reliability, handling packet loss
    and reordering.
    """
    first = True
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(2)  # Set timeout for server response

    server_address = (server_ip, server_port)
    expected_seq_num = 0
    output_file_path = "received_file.txt"  # Default file name

    with open(output_file_path, 'wb') as file:
        while True:
            try:
                # Send initial connection request to server
                if first==True:
                    client_socket.sendto(b"START", server_address)

                    data, server_address = client_socket.recvfrom(1024)
                    # client_socket.sendto(b"AckForRTT", server_address)
                    packet = {
                        "Acknowledgement": "RTT",
                    }
                    jsonPacket = json.dumps(packet).encode("utf-8")
                    client_socket.sendto(jsonPacket, server_address)

                    first = False

                # Receive the packet
                packet, _ = client_socket.recvfrom(MSS + 100)  # Allow room for headers

                # Check if the packet contains an end signal
                if b"END" in packet:
                    print("Received END signal from server, file transfer complete")
                    break
                
                # Parse the sequence number and data from the packet
                
                seq_num, data = parse_packet(packet)

                # If the packet is in order, write it to the file
                if seq_num == expected_seq_num:
                    file.write(data)
                    print(f"Received packet {seq_num}, writing to file")
                    
                    # Update expected sequence number and send cumulative ACK
                    expected_seq_num += 1
                    send_ack(client_socket, server_address, seq_num)
                elif seq_num < expected_seq_num:
                    # Duplicate or old packet, resend ACK
                    send_ack(client_socket, server_address, seq_num)
                else:
                    # Out-of-order packet, log and ignore
                    print(f"Out-of-order packet {seq_num} received, expected {expected_seq_num}")

            except socket.timeout:
                print("Timeout waiting for data")

def parse_packet(packet):
    """
    Parse the packet to extract the sequence number and data.
    """
    jsonPacket = json.loads(packet.decode("utf-8"))
    seq_num = jsonPacket["sequence_number"]
    data = jsonPacket["data"].encode("latin1")
    return int(seq_num), data

def send_ack(client_socket, server_address, seq_num):
    """
    Send a cumulative acknowledgment for the received packet.
    """
    data = "ACK".encode("utf-8")
    ack_packet = {
        "sequence_number": seq_num,
        "data_length": len(data),
        "data": data.decode('latin1')
    }
    ack_json_packet = json.dumps(ack_packet).encode("utf-8")
    client_socket.sendto(ack_json_packet, server_address)
    print(f"Sent cumulative ACK for packet {seq_num}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Reliable file receiver over UDP.')
parser.add_argument('server_ip', help='IP address of the server')
parser.add_argument('server_port', type=int, help='Port number of the server')

args = parser.parse_args()

# Run the client
receive_file(args.server_ip, args.server_port)
