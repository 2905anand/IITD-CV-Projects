import matplotlib.pyplot as plt
import subprocess
import time
import json

def calcAvg(pTime):
    sum=0
    print(len(pTime))
    for i in range(len(pTime)):
        sum+=pTime[i]
    return sum/10

server_binary = './server'  
client_binary = './client'  
timeList = []

for k in range(1,11):
    with open('config.json','r') as file:
        data = json.load(file)

    data["p"]=k

    with open('config.json','w') as file:
        json.dump(data, file, indent=3)

    for i in range(10):
        print("Run:",i+1," p:",k)

        server_process = subprocess.Popen([server_binary])
        time.sleep(0.02)

        client_process = subprocess.run([client_binary], capture_output=True)

        server_process.terminate()
        # server_process.wait()

        print("Server terminated.")
        print("-" * 30)

    pTime = []
    with open('time.txt', 'r') as file:  
        inputs = file.readlines()  

    for input in inputs:
        pTime.append(float(input))

    timeList.append(calcAvg(pTime))
    

    with open('time.txt', 'w') as file:
        pass

x_axis=[1,2,3,4,5,6,7,8,9,10]

for i in range(10):
    print(timeList[i])

plt.plot(x_axis, timeList,label= "Time Taken vs P")
plt.xlabel("p--->")
plt.ylabel("Time taken")
plt.title("Graph")
plt.legend()
plt.savefig("plot.png")


# import matplotlib.pyplot as plt
# import subprocess
# import time
# import json
# import numpy as np
# from scipy import stats

# def calcAvg(pTime):
#     return np.mean(pTime)

# def calcConfidenceInterval(pTime):
#     confidence_level = 0.95  # 95% confidence level
#     degrees_freedom = len(pTime) - 1
#     sample_mean = np.mean(pTime)
#     sample_standard_error = stats.sem(pTime)  # Standard error
#     confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
#     return sample_mean, confidence_interval

# server_binary = './server'  # Replace with your actual server binary path
# client_binary = './client'  # Replace with your actual client binary path

# timeList = []
# confidence_intervals = []

# for k in range(1, 11):
#     with open('config.json', 'r') as file:
#         data = json.load(file)

#     data["p"] = k

#     with open('config.json', 'w') as file:
#         json.dump(data, file, indent=3)

#     # Path to the precompiled server and client binaries

#     # Run the server and client 10 times
#     pTime = []
#     for i in range(10):
#         print(f"Run {i + 1} with p = {k}")

#         # Start the server in the background
#         server_process = subprocess.Popen([server_binary])

#         # Wait a bit to ensure the server is running (adjust the sleep time if needed)
#         # time.sleep(0.5)

#         # Run the client and capture the output
#         client_process = subprocess.run([client_binary], capture_output=True)

#         # Terminate the server after the client finishes
#         server_process.terminate()

#         # Wait for the server process to end gracefully
#         # server_process.wait()

#         print("Server terminated.")
#         print("-" * 30)

#         # Read and store the completion times
#         with open('time.txt', 'r') as file:
#             for line in file:
#                 line = line.strip()
#                 if line:  # Check if line is not empty
#                     pTime.append(float(line))  # Store time

#     # Calculate average time and confidence interval
#     avg_time, confidence_interval = calcConfidenceInterval(pTime)

#     # Append results
#     timeList.append(avg_time)
#     confidence_intervals.append(confidence_interval)

#     # Clear time.txt for the next run
#     with open('time.txt', 'w') as file:
#         pass

# # Prepare data for plotting
# x_axis = list(range(1, 11))  # Values of p from 1 to 10
# lower_bound = [avg - conf[0] for avg, conf in zip(timeList, confidence_intervals)]
# upper_bound = [conf[1] - avg for avg, conf in zip(timeList, confidence_intervals)]
# error = [upper_bound[i] - lower_bound[i] for i in range(len(x_axis))]

# # Plot the graph with error bars (confidence intervals)
# plt.errorbar(x_axis, timeList, yerr=error, fmt='-o', label="Time Taken vs P")
# plt.xlabel("p--->")
# plt.ylabel("Time taken")
# plt.title("Completion Time vs Packet Size (p)")
# plt.legend()
# plt.savefig("plot_with_confidence_intervals.png")
# # plt.show()
