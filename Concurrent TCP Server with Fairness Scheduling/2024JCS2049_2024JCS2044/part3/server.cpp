#include <cstring>
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <nlohmann/json.hpp>
using namespace std;

using json = nlohmann::json;

struct configData{

};

queue<int> queueForClientId;


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
    // cout << words.size() << " " << k << " " << ceil(double(words.size()) / k) << endl;
    p = config["p"];
    cout<<p<<endl;
    for (int i = 0; i < intervals; i++)
    {
        if (i == intervals - 1)
        {
            words.push_back("EOF");
            int noOfPackets = ceil(double(words.size() - offset) / p);
            cout << "Packets: " << noOfPackets << "Words in each packet: "<<p<<endl;

            int n = offset;
            int count = 0;
            for (int j = 0; j < noOfPackets; j++)
            {
                string wordPackets = "";
                if (j == noOfPackets - 1)
                {
                    int a = 1;
                    // cout << count << endl;
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
            cout<<endl;
        }
        else
        {

            int noOfPackets = ceil(double(k) / p);
            cout << "Packets: " << noOfPackets << "Words in each packet: "<<p<<endl;
            int n = offset;
            int count = 0;
            for (int j = 0; j < noOfPackets; j++)
            {
                string wordPackets = "";
                if (j == noOfPackets - 1)
                {
                    int a = 1;
                    // cout << count << endl;
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
        cout<<endl;
    }
}

void check(){}

void *transmission(void* clientSock){
    int clientSocket = *(int*)clientSock;
    // delete clientSock
    char buffer[1024] = {0};
    recv(clientSocket, buffer, sizeof(buffer), 0);
    string s(buffer);
    int offset = stoi(s);

    sending(clientSocket, offset, 10, 10);
    return NULL;
}


int main()
{

    json config;
    ifstream config_file("config.json");

    if (config_file.is_open() == -1)
    {
        cerr << "Error in opening config.json file" << endl;
    }

    config_file >> config;

    string server_ip = config["server_ip"];
    int server_port = config["server_port"];
    int k = config["k"];
    int p = config["p"];
    string input_file = config["input_file"];
    int num_clients = config["num_clients"];
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

    // int clientSocket = accept(serverSocket, nullptr, nullptr);

    while(true){
        int clientSocket = accept(serverSocket, nullptr, nullptr);
        if(clientSocket == -1){
            cout << "Unable to connect" << endl;
            return 1;
        }
        int *clientSocketPtr = new int(clientSocket);

        pthread_t thread; 
        int response = pthread_create(&thread, nullptr, transmission,(void*)clientSocketPtr);
        if(response == -1)
        {
            cout << "Unable to create Thread"<<endl;
            close(clientSocket);
        }
        
    }

    // if (clientSocket == -1)
    // {
    //     cout << "Unable to connect" << endl;
    //     return 1;
    // }
    // else
    // {
    //     cout << "Connected to Client" << endl;
    // }

    
    // cout << "Offset sent from client: " << offset << endl;

    

    // close(clientSocket);
    close(serverSocket);
    return 0;
}