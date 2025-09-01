#include <cstring>
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <nlohmann/json.hpp>
#include <mutex>
using namespace std;

using json = nlohmann::json;

queue<int> clientSocketQueue;
std::mutex mutexLock;

void *transmission(void* clientSock){

    int clientSocket = *(int*)clientSock;

    ifstream file("file.txt");
    vector<string> words;
    string word;

    if (!file.is_open())
    {
        cerr << "Error in file opening!!" << endl;
        return nullptr;
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

    clientSocketQueue.push(clientSocket);

    while(clientSocketQueue.front()!= clientSocket);
    mutexLock.lock();

    cout <<clientSocket<<endl;
    
    int index = 0;
    while (index < words.size())
    {
        // recv(clientSocket, buffer, sizeof(buffer), 0);
        // string s(buffer);
        // int offset = stoi(s);
        send(clientSocket, words[index].c_str(), words[index].size(), 0);
        index++;
    }
    clientSocketQueue.pop();  
    mutexLock.unlock();

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

    int opt = 1,response;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip.c_str(), &serverAddress.sin_addr);

    response = bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
    if(response == -1){
        cout<<"Server unable to bind"<<endl;
        return 0;
    }

    response = listen(serverSocket, num_clients);
    if(response == -1){
        cout<<"Server unable to listen"<<endl;
        return 0;
    }
    cout << "Waiting for connection....." << endl;

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

    close(serverSocket);
    return 0;
}