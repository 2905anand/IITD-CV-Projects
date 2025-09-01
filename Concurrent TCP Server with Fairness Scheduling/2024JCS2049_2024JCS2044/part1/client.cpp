#include <cstring>
#include <iostream>
#include <sstream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <map>
#include <arpa/inet.h>
#include <bits/stdc++.h>
#include <nlohmann/json.hpp>
#include <chrono>
using namespace std;
using namespace std::chrono;

using json = nlohmann::json;

void transmission(map<string, int> wordsFrequency, int clientSocket)
{
    json config;
    ifstream config_file("config.json");

    if (config_file.is_open() == -1)
    {
        cerr << "Error in opening config.json file" << endl;
    }

    config_file >> config;
    int k = config["k"];


    char buffer[1024];
    int set = 0;bool flag = false;
    while (true)
    {
        string offset = to_string(set);
        send(clientSocket, offset.c_str(), offset.size(), 0);
        int count = 0;
        while (count < k)
        {
            memset(buffer, 0, sizeof(buffer));
            recv(clientSocket, buffer, sizeof(buffer), 0);

            string words(buffer);
            stringstream ss(words);
            string word;
            while (getline(ss, word, ','))
            {
                if (word == "EOF")
                {
                    // cout<<word<<endl;
                    flag=true;
                    cout << "Analysis Completed" << endl;
                    break;
                }
                wordsFrequency[word]++;
                count++;
            }
            if(flag == true){
                break;
            }
        }
        set+=count;
        if(flag == true){
            break;
        }
        // cout<<set<<" "<<endl;
        
    }
    // wordsFrequency.erase("EOF");

    ofstream file;
    file.open("output.txt");
    for (auto i : wordsFrequency)
    {
        file << i.first << ", " << i.second << endl;
    }
    // cout << endl;
    file.close();

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

    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip.c_str(), &serverAddress.sin_addr);

    int c = connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
    if (c == -1)
    {
        cout << "Unable to Connect" << endl;
        return 1;
    }
    else
    {
        cout << "Connected to Server" << endl;
    }

    char buffer[1024] = {0};

    map<string, int> wordsFrequency;
    int c1 = 0;

    auto startTime = high_resolution_clock::now();

    transmission(wordsFrequency, clientSocket);

    auto endTime = high_resolution_clock::now();

    duration<double, milli> t = endTime - startTime;
    // cout << t.count() << endl;

    // cout << "Time Taken = " << t.count() << endl;
    // }
    ofstream file;
    file.open("time.txt", ios::app);
    file << t.count() << endl;
    file.close();

    close(clientSocket);
    return 0;
}