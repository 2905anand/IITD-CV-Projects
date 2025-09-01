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

void sending(int clientSocket)
{
    ifstream file("file.txt");
    vector<string> words;
    string word;

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

    json config;
    ifstream config_file("config.json");

    if (config_file.is_open() == -1)
    {
        cerr << "Error in opening config.json file" << endl;
    }

    config_file >> config;

    char buffer[1024] = {0};
    int offset;
    
    int p = config["p"];
    int k = config["k"];
    int flag=false;

    while (true)
    {
        recv(clientSocket, buffer, sizeof(buffer), 0);
        string s(buffer);
        int offset = stoi(s);

        if (offset + k < words.size())
        {
            int noOfPackets = ceil(double(k) / p);
            cout << "Packets: " << noOfPackets << " Words in each packet: "<<p<<endl;
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
        }
        else if (offset < words.size())
        {
            words.push_back("EOF");
            int noOfPackets = ceil(double(words.size() - offset) / p);
            cout << "Packets: " << noOfPackets << " Words in each packet: "<<p<<endl;

            int n = offset;
            int count = 0;
            for (int j = 0; j < noOfPackets; j++)
            {
                string wordPackets = "";
                if (j == noOfPackets - 1)
                {
                    int a = 1;
                    count = words.size() - offset - count;
                    while (a <= count)
                    {
                        wordPackets = wordPackets + words[n] + ",";
                        n++;
                        a++;
                    }
                    flag=true;
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
            // cout << endl;
        }
        if (flag == true)       
        {
            break;
        }
        
    }
    return;
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
    // cout << server_ip << endl;

    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip.c_str(), &serverAddress.sin_addr);

    bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));

    listen(serverSocket, 10);
    cout << "Waiting for connection....." << endl;

    int clientSocket = accept(serverSocket, nullptr, nullptr);

    if (clientSocket == -1)
    {
        cout << "Unable to connect" << endl;
        return 1;
    }
    else
    {
        cout << "Connected to Client" << endl;
    }

    // cout << "Offset sent from client: " << offset << endl;

    sending(clientSocket);

    close(clientSocket);
    close(serverSocket);
    return 0;
}