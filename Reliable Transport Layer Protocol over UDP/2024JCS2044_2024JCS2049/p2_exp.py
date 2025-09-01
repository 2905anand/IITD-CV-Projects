from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import time, re, os,math
import sys
import hashlib

class CustomTopo(Topo):
    def build(self, loss, delay):
        # Add two hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')

        # Add a single switch
        s1 = self.addSwitch('s1')

        # Add links
        # Link between h1 and s1 with the specified packet loss
        self.addLink(h1, s1, loss=loss, delay=f'{delay}ms')

        # Link between h2 and s1 with no packet loss
        self.addLink(h2, s1, loss=0)


def compute_md5(file_path):
    """Compute the MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            # Read the file in chunks to avoid using too much memory for large files
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def run(expname):
    # Set the log level to info to see detailed output
    setLogLevel('info')
    
    # IP and port of the remote controller
    controller_ip = '127.0.0.1'  # Change to the controller's IP address if not local
    controller_port = 6653       # Default OpenFlow controller port
    
    # Output file 
    output_file = f'reliability_{expname}_p2.csv'
    f_out = open(output_file, 'w')
    f_out.write("loss,delay,fast_recovery,md5_hash,ttc\n")


    SERVER_IP = "10.0.0.1"
    SERVER_PORT = 6555 
            
    NUM_ITERATIONS = 5
    OUTFILE = 'received_file.txt'
    delay_list, loss_list = [], []
    if expname == "loss":
        loss_list = [x*0.5 for x in range (0, 11)]
        delay_list = [20]
    elif expname == "delay":
        delay_list = [x for x in range(0, 201, 20)]
        loss_list = [1]
    print(loss_list, delay_list)

    port = 0
    
    # TOTAL_FILE_SIZE = os.path.getsize("file.txt")

    # Loop to create the topology 10 times with varying loss (1% to 10%)
    for LOSS in loss_list:
        for DELAY in delay_list:
            for FAST_RECOVERY in [True]:
                for i in range(0, NUM_ITERATIONS):
                    print(f"\n--- Running topology with {LOSS}% packet loss, {DELAY}ms delay and fast recovery {FAST_RECOVERY}")
                    # port=port+1
                    
                    # Create the custom topology
                    topo = CustomTopo(loss=LOSS, delay=DELAY)

                    net = Mininet(topo=topo, link=TCLink, controller=None)
                    remote_controller = RemoteController('c0', ip=controller_ip, port=controller_port, encoding = 'latin1')
                    net.addController(remote_controller)

                    try:
                        # Start the network
                        net.start()

                        # Get references to h1 and h2
                        h1 = net.get('h1')
                        h2 = net.get('h2')

                        
                        # fast_recovery_arg = "--fast_recovery" if FAST_RECOVERY else ""
                        h1.cmd(f"python3 -u p2_server.py {SERVER_IP} {SERVER_PORT} > server.log 2>&1 & ")
                        start_time = time.time()
                        result = h2.cmd(f"python3 -u p2_client.py {SERVER_IP} {SERVER_PORT} > client.log 2>&1")
                        end_time = time.time()
                        print("Done for the command")

                        # Record results
                        ttc = end_time - start_time
                        throughput = 1/(ttc * math.sqrt(LOSS))
                        md5_hash = compute_md5('received_file_p2.txt')
                        f_out.write(f"{LOSS},{DELAY},{FAST_RECOVERY},{md5_hash},{throughput}\n")
                        f_out.flush()
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        # Ensure network is stopped
                        net.stop()
                        print("Network stopped.")
                        time.sleep(1)  # Pause before next iteration

    print("\n--- Completed all tests ---")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python experiment.py <expname>")
    else:
        expname = sys.argv[1].lower()
        run(expname)