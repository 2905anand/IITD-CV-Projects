#include <cstring>
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <queue>
#include <vector>
#include <condition_variable>

using namespace std;
using json = nlohmann::json;

struct configData {};

queue<int> fifo_queue;
vector<int> rr_queue;
int rr_index = 0;

mutex queue_mutex;
condition_variable queue_cv;
bool fifo_mode = true; // FIFO by default, can switch to Round-Robin

void sending(int clientSocket, int offset, int k, int p)
{
    ifstream file("file.txt");
    vector<string> words;
    string word;

    // json config
    json config;
    ifstream config_file("config.json");

    if (config_file.is_open() == -1)
    {
        cerr << "Error in opening config.json file" << endl;
    }

    config_file >> config;
    if (!file.is_open())
    {
        cerr << "Error in file opening!!" << endl;
        return;
    }
    else
    {
        int i = 1;
        while ((getline(file, word, ',')))
        {
            words.push_back(word);
            i++;
        }
        file.close();
    }

    int intervals = ceil(double(words.size()) / k);
    p = config["p"];
    cout << p << endl;
    for (int i = 0; i < intervals; i++)
    {
        if (i == intervals - 1)
        {
            words.push_back("EOF");
            int noOfPackets = ceil(double(words.size() - offset) / p);
            cout << "Packets: " << noOfPackets << "Words in each packet: " << p << endl;

            int n = offset;
            int count = 0;
            for (int j = 0; j < noOfPackets; j++)
            {
                string wordPackets = "";
                if (j == noOfPackets - 1)
                {
                    int a = 1;
                    while (a <= (words.size() - offset - count))
                    {
                        wordPackets = wordPackets + words[n] + ",";
                        n++;
                        a++;
                    }
                }
                else
                {
                    int a = 1;
                    while (a <= p)
                    {
                        wordPackets = wordPackets + words[n] + ",";
                        n++;
                        a++;
                        count++;
                    }
                }

                cout << wordPackets << endl;
                send(clientSocket, wordPackets.c_str(), wordPackets.size(), 0);
            }
            offset += k;
            cout << endl;
        }
        else
        {

            int noOfPackets = ceil(double(k) / p);
            cout << "Packets: " << noOfPackets << "Words in each packet: " << p << endl;
            int n = offset;
            int count = 0;
            for (int j = 0; j < noOfPackets; j++)
            {
                string wordPackets = "";
                if (j == noOfPackets - 1)
                {
                    int a = 1;
                    while (a <= k - count)
                    {
                        wordPackets = wordPackets + words[n] + ",";
                        n++;
                        a++;
                    }
                }
                else
                {
                    int a = 1;
                    while (a <= p)
                    {
                        wordPackets = wordPackets + words[n] + ",";
                        n++;
                        a++;
                        count++;
                    }
                }
                cout << wordPackets << endl;
                send(clientSocket, wordPackets.c_str(), wordPackets.size(), 0);
            }
            offset += k;
        }
        cout << endl;
    }
}

void* transmission(void* clientSock){
    int clientSocket = *(int*)clientSock;
    delete (int*)clientSock;
    char buffer[1024] = {0};
    recv(clientSocket, buffer, sizeof(buffer), 0);
    string s(buffer);
    int offset = stoi(s);
    sending(clientSocket, offset, 10, 10);
    return NULL;
}

// FIFO scheduling
void fifo_scheduling() {
    while (true) {
        unique_lock<mutex> lock(queue_mutex);
        queue_cv.wait(lock, [] { return !fifo_queue.empty(); });
        int clientSocket = fifo_queue.front();
        fifo_queue.pop();
        lock.unlock();
        transmission(&clientSocket);  // Serve the client
    }
}

// Round-Robin scheduling
void round_robin_scheduling() {
    while (true) {
        unique_lock<mutex> lock(queue_mutex);
        queue_cv.wait(lock, [] { return !rr_queue.empty(); });
        int clientSocket = rr_queue[rr_index];
        rr_index = (rr_index + 1) % rr_queue.size();
        lock.unlock();
        transmission(&clientSocket);  // Serve the client
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        cerr << "Usage: ./server --policy <fifo|rr>" << endl;
        return 1;
    }

    // Determine which scheduling policy is selected
    string policy = argv[1];
    if (policy == "--fifo") {
        fifo_mode = true;
    } else if (policy == "--rr") {
        fifo_mode = false;
    } else {
        cerr << "Invalid policy! Use --policy fifo or --policy rr" << endl;
        return 1;
    }

    json config;
    ifstream config_file("config.json");
    if (config_file.is_open() == -1) {
        cerr << "Error in opening config.json file" << endl;
    }

    config_file >> config;

    string server_ip = config["server_ip"];
    int server_port = config["server_port"];
    cout << server_ip << endl;

    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip.c_str(), &serverAddress.sin_addr);

    bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
    listen(serverSocket, 15);
    cout << "Waiting for connection....." << endl;

    // Create thread for FIFO or Round-Robin scheduling
    pthread_t scheduling_thread;
    if (fifo_mode) {
        pthread_create(&scheduling_thread, nullptr, reinterpret_cast<void* (*)(void*)>(fifo_scheduling), nullptr);
    } else {
        pthread_create(&scheduling_thread, nullptr, reinterpret_cast<void* (*)(void*)>(round_robin_scheduling), nullptr);
    }

    while (true) {
        int clientSocket = accept(serverSocket, nullptr, nullptr);
        if (clientSocket == -1) {
            cout << "Unable to connect" << endl;
            continue;
        }

        unique_lock<mutex> lock(queue_mutex);
        if (fifo_mode) {
            fifo_queue.push(clientSocket);
        } else {
            rr_queue.push_back(clientSocket);
        }
        lock.unlock();
        queue_cv.notify_one();  // Notify the scheduling thread
    }

    close(serverSocket);
    return 0;
}
