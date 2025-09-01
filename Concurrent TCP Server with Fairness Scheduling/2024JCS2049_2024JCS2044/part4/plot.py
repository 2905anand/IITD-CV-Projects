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
x_axis=[]

for k in range(0,9):
    with open('config.json','r') as file:
        data = json.load(file)
    if(k==0):
        data["num_clients"] = 1
    elif(k==1):
        data["num_clients"] = 4
    else:
        data["num_clients"] = data["num_clients"]+4
    x_axis.append(data["num_clients"])

    with open('config.json','w') as file:
        json.dump(data, file, indent=5)

    server_process = subprocess.Popen([server_binary])
    time.sleep(0.05)
    client_process = subprocess.run([client_binary], capture_output=True)
    server_process.terminate()
    print("Server terminated.")
    print("-" * 30)
    clientTime=[]
    with open('time.txt','r') as file:
        inputs = file.readlines()
    for input in inputs:
        a = float(input)
        clientTime.append(a)
    timeList.append(calcAvg(clientTime))

    with open('time.txt', 'w') as file:
        pass



for i in range(len(timeList)):
    print(timeList[i])

plt.plot(x_axis, timeList,label= "Time Taken vs P")
plt.xlabel("p--->")
plt.ylabel("Time taken")
plt.title("Graph")
plt.legend()
plt.savefig("plot.png")