import socket
import time
import argparse
import json

# Constants
MSS = 1400  # Maximum Segment Size for each packet
WINDOW_SIZE = 5  # Number of packets in flight
DUP_ACK_THRESHOLD = 3  # Threshold for duplicate ACKs to trigger fast retransmission
FILE_PATH = "file.txt"  # Path to the file to be sent
TIMEOUT = 1.0  # Timeout duration in seconds

s_ip = ""
s_port = 8080

alpha = 0.125
beta = 0.25
sampleRTT = 0.01
devRTT = 0.005
EstimatedRTT = 0.01

def send_file(server_ip, server_port, enable_fast_recovery):
    """
    Send a predefined file to the client, ensuring reliability over UDP.
    """
    s_ip = server_ip
    s_port = server_port
    global EstimatedRTT, devRTT, TIMEOUT

    # Initialize UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((server_ip, server_port))

    print(f"Server listening on {server_ip}:{server_port}")

    # Wait for client to initiate connection
    client_address = None

    with open(FILE_PATH, 'rb') as file:
        seq_num = 0
        window_base = 0
        unacked_packets = {}
        duplicate_ack_count = 0
        last_ack_received = -1

        # Connection initiation loop
        while True:
            data, client_address = server_socket.recvfrom(1024)
            if data == b"START":
                print(f"Connection established with client {client_address}")
                startTime = time.time()

                server_socket.sendto(b"AckForRTT", client_address)
                data, client_address = server_socket.recvfrom(1024)

                sampleRTT = time.time() - startTime
                print("Sample RTT = ", sampleRTT)

                break

        EstimatedRTT = (1 - alpha) * EstimatedRTT + sampleRTT
        devRTT = (1-beta) * devRTT + beta * abs(sampleRTT - EstimatedRTT)

        TIMEOUT = EstimatedRTT + 4*devRTT
        print("TIMEOUT = ", TIMEOUT)
        # Start sending packets from the file
        while True:
            # Send packets within the window
            while seq_num < window_base + WINDOW_SIZE:
                chunk = file.read(MSS)
                if not chunk:
                    # End of file, send end signal
                    packet = {
                        "sequence_number": 1,
                        "data_length": 1,
                        "data": b"END".decode("latin1")
                    }
                    jsonPacket = json.dumps(packet).encode("utf-8")
                    server_socket.sendto(jsonPacket, client_address)
                    break

                # Create and send the packet
                packet = create_packet(seq_num, chunk)
                server_socket.sendto(packet, client_address)
                unacked_packets[seq_num] = (packet, time.time())  # Track sent packets
                print(f"Sent packet {seq_num}")
                seq_num += 1

            # Wait for ACKs or timeout
            try:
                server_socket.settimeout(TIMEOUT)
                ack_packet, _ = server_socket.recvfrom(1024)

                print("-----------------", ack_packet)

                ack_json_packet = json.loads(ack_packet.decode("utf-8"))
                try: 
                    ack_seq_num = ack_json_packet["sequence_number"]
                except Exception as e:
                    continue

                # Process ACK
                if ack_seq_num > last_ack_received:
                    print(f"Received cumulative ACK for packet {ack_seq_num}")
                    last_ack_received = ack_seq_num

                    sampleRTT = time.time() - unacked_packets[ack_seq_num][1]
                    print(sampleRTT)                        

                    if sampleRTT < 2*EstimatedRTT:
                        EstimatedRTT = (1 - alpha) * EstimatedRTT + alpha * sampleRTT
                        devRTT = (1-beta) * devRTT + beta * abs(sampleRTT - EstimatedRTT)
                        print("EstimatedRTT = ", EstimatedRTT,"devRTT = ",devRTT)
                        TIMEOUT = EstimatedRTT + 4*devRTT

                        print("Updated TIMEOUT=", TIMEOUT)
                        
                    else : 
                        print("Discarded Sample RTT = ", sampleRTT)
                    # Slide window forward and remove acknowledged packets
                    for seq in list(unacked_packets):
                        if seq <= ack_seq_num:
                            del unacked_packets[seq]
                    window_base = ack_seq_num + 1
                    duplicate_ack_count = 0  # Reset duplicate counter
                else:
                    # Handle duplicate ACK
                    duplicate_ack_count += 1
                    print(f"Received duplicate ACK for packet {ack_seq_num}, count={duplicate_ack_count}")

                    if enable_fast_recovery and duplicate_ack_count >= DUP_ACK_THRESHOLD:
                        print("Entering fast recovery mode")
                        fast_retransmit(server_socket, client_address, unacked_packets)

            except socket.timeout:
                # Timeout handling: retransmit all unacknowledged packets
                print("Timeout occurred, retransmitting unacknowledged packets")
                retransmit_unacked_packets(server_socket, client_address, unacked_packets)

            # End the loop if file is sent and acknowledged
            if not chunk and len(unacked_packets) == 0:
                print("File transfer complete")
                print(enable_fast_recovery)
                break

def create_packet(seq_num, data):
    """
    Create a packet with the sequence number and data.
    """
    packet = {
        "sequence_number": seq_num,
        "data_length": len(data),
        "data": data.decode('latin1')
    }
    jsonPacket = json.dumps(packet).encode("utf-8")
    return jsonPacket


def retransmit_unacked_packets(server_socket, client_address, unacked_packets):
    """
    Retransmit all unacknowledged packets.
    """
    for seq, (packet, _) in unacked_packets.items():
        retries = 0
        while retries < 3:
            try:
                server_socket.sendto(packet, client_address)
                print(f"Retransmitted packet {seq}")
                break 
            except OSError as e:
                if e.errno == 101:
                    print("Network is unreachable. Waiting to retry...")
                    retries += 1
                    time.sleep(1)
                    if retries == 3:
                        print("Waiting for the network to become reachable...")
                        while True:
                            try:
                                server_socket.sendto(b"PING", client_address)
                                print("Network is now reachable!")
                                break 
                            except OSError as inner_e:
                                if inner_e.errno == 101: 
                                    print("Still unreachable, waiting...")
                                    time.sleep(1)
                else:
                    raise
    return server_socket


def fast_retransmit(server_socket, client_address, unacked_packets):
    """
    Retransmit the earliest unacknowledged packet (fast retransmission).
    """
    earliest_seq = min(unacked_packets)
    server_socket.sendto(unacked_packets[earliest_seq][0], client_address)
    print(f"Fast retransmitted packet {earliest_seq}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Reliable file transfer server over UDP.')
parser.add_argument('server_ip', help='IP address of the server')
parser.add_argument('server_port', type=int, help='Port number of the server')
parser.add_argument('--fast_recovery', action='store_true', help='Enable fast recovery for duplicate ACKs')

args = parser.parse_args()

# Run the server
send_file(args.server_ip, args.server_port, args.fast_recovery)
