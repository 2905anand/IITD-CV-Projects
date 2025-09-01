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

struct clientTimeTaken{
    int timeTaken[5];
}clientTimeTaken;


double calcAvg(double timeTaken[])
{
    double sum = 0;
    for (int i = 1; i <= 10; i++)
    {
        sum += timeTaken[i];
    }
    return sum / 10;
}
void transmission(unordered_map<string, int> wordsFrequency, int clientSocket)
{
    json config;
    ifstream config_file("config.json");

    if (config_file.is_open() == -1)
    {
        cerr << "Error in opening config.json file" << endl;
    }

    config_file >> config;

    char buffer[1024];
    while (true)
    {
        memset(buffer, 0, sizeof(buffer));
        recv(clientSocket, buffer, sizeof(buffer), 0);

        string words(buffer);
        stringstream ss(words);
        cout<<words<<endl;
        string word;
        while (getline(ss, word, ','))
        {
            wordsFrequency[word]++;
        }
        if (words.find("EOF") != string::npos)
        {
            cout << "Analysis Completed" << endl;
            break;
        }
        // if (words.find("$$"))
        // {
        //     cout<<"Offset is Wrong"<<endl;
        //     return 1;
        // }
    }
    wordsFrequency.erase("EOF");

    ofstream file;
    file.open("output.txt");
    for (auto i : wordsFrequency)
    {
        file << i.first << ", " << i.second << endl;
    }
    cout << endl;
    file.close();

    return;
}

void* receiving(void* arg){
    
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

    string offset="0";
    // cout << "Enter Offset value: ";
    // cin >> offset;
    send(clientSocket, offset.c_str(), offset.size(), 0);

    unordered_map<string, int> wordsFrequency;
    int c1 = 0;

    auto startTime = high_resolution_clock::now();

    transmission(wordsFrequency, clientSocket);

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
    int th;
    for (int i = 0; i < num_clients; i++)
    {
        cout<<"Client "<<i<<endl;
        int response = pthread_create(&threads[i], NULL, receiving, NULL);
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