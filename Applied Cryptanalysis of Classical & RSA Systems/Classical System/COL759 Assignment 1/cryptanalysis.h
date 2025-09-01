#include <bits/stdc++.h>
#include <iostream>
#include "util.h"
using namespace std;

class CryptAnalysis
{
public:
    string cipherText = "";
    string originalCipherText = "";
    CryptAnalysis(string ct, string ct1)
    {
        cipherText = ct;
        originalCipherText = ct1;
    };
    // unordered_set<int> possibleKeySizeByKasaski;

    // Return the possible key size
    unordered_set<int> kasaskiMethod();

    // Return the exact key size that fullfill both Kasaski and IC method
    int indexOfCoincidence(unordered_set<int> &possibleKeySizeByKasaski, int leftRange, int rightRange);

    // Return the key for the decryption
    vector<string> mutualIndexOfCoincidence(int keySize);

    string decryptVigenereCipher(string cipherText, int keySize, string testKey);
};

string CryptAnalysis::decryptVigenereCipher(string cipherText, int keySize, string testKey){
    string plainText = "";
    for (int j = 0; j < cipherText.size();)
    {
        for (int k = 0; k < keySize; k++, j++)
        {
            char plain = ((cipherText[j] - 'A') - (testKey[k] - 'A') + 26) % 26 + 'A';
            plainText += plain;
        }
    }
    return plainText;
}
vector<string> CryptAnalysis::mutualIndexOfCoincidence(int keySize)
{
    Util util;
    vector<string> substrings = util.generateSubstring(cipherText, keySize);
    cout << endl;
    vector<vector<int>> equations;
    int key[keySize];
    key[0] = 0;
    for (int i = 1; i < keySize; i++)
    {
        key[i] = -1;
    }

    for (int i = 0; i < substrings.size(); i++)
    {
        for (int j = i + 1; j < substrings.size(); j++)
        {
            pair<int, double> micValue = util.calculateMIC2strings(substrings[i], substrings[j]);
            vector<int> equation = {i, j, micValue.first};
            equations.push_back(equation);
        }
    }

    int count = 1;
    for (int i = 0; i < equations.size(); i++)
    {
        if (equations[i][0] == 0)
        {
            key[equations[i][1]] = equations[i][2];
            count++;
        }
    }
    for (int i = 0; i < equations.size() && count <= keySize; i++)
    {
        if ((key[equations[i][0]] != 0) && (key[equations[i][0]] != -1) && (key[equations[i][1]] == -1))
        {
            count++;
            key[equations[i][1]] == key[equations[i][0]] + equations[i][2];
        }
        else if ((key[equations[i][1]] != 0) && (key[equations[i][0]] == -1) && (key[equations[i][1]] != -1))
        {
            count++;
            key[equations[i][0]] == abs(key[equations[i][1]] - equations[i][2]);
        }
        else
        {
            continue;
        }
    }
    for (int i = 1; i < keySize; i++)
    {
        key[i] = 26 - key[i];
    }
    string finalKey;
    vector<string> keysList;
    vector<int> frequency;
    int maxFrequency= -1;int index=-1;
    string originalPlainText;
    unordered_set<string> dictionaryWords = {"the","The","be","to","of","and","a","A","in","that","have", "I","It","it","for","not","on","with","he","He","as","you","do","at","this","This","but","his","by","from"};

    for (int i = 0; i < 26; i++)
    {
        int testKey[keySize];
        finalKey = "";
        for (int j = 0; j < keySize; j++)
        {
            testKey[j] = (key[j] + i) % 26;
        }
        for (int j = 0; j < keySize; j++)
        {
            finalKey += char(testKey[j] + 65);
        }
        keysList.push_back(finalKey);
    }

    // For the ith string, we find the MIC with other susbtring
    cout << "-----------------------------------------------" << "\n";
    cout << "26 Possible KEYS: " << "\n"
         << "\n";
    for (int i = 0; i < 26; i++)
    {
        string testKey = keysList[i];
        cout << testKey << " ";
        string plainText = decryptVigenereCipher(cipherText,keySize,testKey);
        string a = util.addSpecialCharacters(originalCipherText, plainText);
        stringstream ss(a);
        int freq=0;
        string word;
        vector<string> words;
        while (ss >> word){
            words.push_back(word);
        }
        
        for(int i=0;i<words.size();i++){
            if(dictionaryWords.find(words[i]) != dictionaryWords.end()){
                freq++;
            }
        }
        if(freq > maxFrequency){
            maxFrequency = freq;
            index=i;
            originalPlainText = a;
        }
        frequency.push_back(freq);
        cout << frequency[i] << endl;
    }
    cout << endl;
    cout << "Key: "<< keysList[index] << endl;
    cout << "Plaintext: " << originalPlainText << endl;
    return keysList;
}

int CryptAnalysis::indexOfCoincidence(unordered_set<int> &possibleKeySizeByKasaski, int leftRange, int rightRange)
{
    Util util;
    vector<int> filteredKeys = util.filterSize(possibleKeySizeByKasaski, leftRange, rightRange);

    cout << "-----------------------------------------------" << "\n";
    cout << "Possible Key Length from " << leftRange << " to " << rightRange << " is :"
         << "\n";
    for (int i : filteredKeys)
        cout << i << " ";
    cout << "\n " << "\n"
         << "\n";

    vector<double> ICValues(rightRange + 1);
    for (int i = 0; i < filteredKeys.size(); i++)
    {
        ICValues[filteredKeys[i]] = util.getICValue(cipherText, filteredKeys[i]);
    }

    int maxIndex = 0;
    for (int i = 0; i <= rightRange; i++)
    {
        if (ICValues[i] != 0)
            cout << "For key of size" << i << "  IC VALUE IS " << ICValues[i] << endl;
        if (ICValues[i] > ICValues[maxIndex])
            maxIndex = i;
    }
    cout << "\n";

    cout << "-----------------------------------------------" << "\n";

    cout << "\n";
    cout << "Finalised Key as per Index of Coincidence Method is --  " << maxIndex << endl;
    cout << "\n"
         << "\n";
    return maxIndex;
}

unordered_set<int> CryptAnalysis::kasaskiMethod()
{
    Util util;
    // Remove the characters and space between letters and make it uppercase
    // string cipher = util.cleanCipherText(cipherText);
    int patternSize = 3;
    // Return the substring with its repetition index in the cipher text
    unordered_map<string, vector<int>> x = util.getAllSubstringFrequency(cipherText, patternSize);

    unordered_set<int> plength = util.possibleKeyLength(x);

    cout << "\n";
    cout << "-----------------------------------------------" << "\n";
    cout << "Possible Key Size as per Kasiski Method: " << "\n"
         << "\n";
    for (int i : plength)
        cout << i << " ";

    cout << "\n"
         << "\n"
         << "\n";

    return plength;
}


// vector<int> frequencyOfChars = util.frequencyOf26Chars(plainText);
        // double icValue; int n= cipherText.size();
        // for(int i=0;i<26;i++){
        //     icValue += (frequencyOfChars[i])*(frequencyOfChars[i]-1);
        // }
        // icValue = icValue/(n*(n-1));