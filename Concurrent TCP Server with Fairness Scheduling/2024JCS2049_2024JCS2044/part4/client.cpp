#include <cstring>
#include <iostream>
#include <sstream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <arpa/inet.h>
#include <bits/stdc++.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <pthread.h>
using namespace std;
using namespace std::chrono;

using json = nlohmann::json;


void transmission(unordered_map<string, int> wordsFrequency, int clientSocket, int clientId)
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
                    cout<<word<<endl;
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
        cout<<set<<" "<<flag<<endl;
        // if (words.find("$$"))
        // {
        //     cout<<"Offset is Wrong"<<endl;
        //     return 1;
        // }
        cout << count << " " << endl;
    }

    string filename = "output"+to_string(++clientId)+".txt";
    ofstream file;
    file.open(filename);
    for (auto i : wordsFrequency)
    {
        file << i.first << ", " << i.second << endl;
    }
    cout << endl;
    file.close();

    return;
}

void* receiving(void* cId){
    int clientId = *(int*)cId;
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
    
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip.c_str(), &serverAddress.sin_addr);

    int c = connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
    if (c == -1)
    {
        cout << "Unable to Connect" << endl;
        return 0;
    }
    else
    {
        cout << "Connected to Server" << endl;
    }
    char buffer[1024] = {0};

    // string offset="0";
    // cout << "Enter Offset value: ";
    // cin >> offset;
    // send(clientSocket, offset.c_str(), offset.size(), 0);

    unordered_map<string, int> wordsFrequency;
    int c1 = 0;

    auto startTime = high_resolution_clock::now();

    transmission(wordsFrequency, clientSocket, clientId);

    auto endTime = high_resolution_clock::now();

    duration<double, milli> t = endTime - startTime;
    // cout << t.count() << endl;

    cout << "Time Taken = " << t.count() << endl;
    // }
    ofstream file;
    file.open("time.txt",ios::app);
    file << t.count() << endl;
    file.close();


    close(clientSocket);

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
    // int server_port = config["server_port"];
    // int k = config["k"];
    // int p = config["p"];
    // string input_file = config["input_file"];
    int num_clients = config["num_clients"];
    cout << server_ip << endl;

    pthread_t threads[num_clients];
    
    for (int i = 0; i < num_clients; i++)
    {
        cout<<"Client "<<i<<endl;
        int* clientId = new int(i);
        
        int response = pthread_create(&threads[i], NULL, receiving, (void *)clientId);
        if(response == -1)
        {
            cout << "At Client Side, Unable to create Thread :"<<i<<endl;
        }
    }
    
    for(int i=0;i<num_clients;i++){
        pthread_join(threads[i],nullptr);
    }
    
    return 0;
}