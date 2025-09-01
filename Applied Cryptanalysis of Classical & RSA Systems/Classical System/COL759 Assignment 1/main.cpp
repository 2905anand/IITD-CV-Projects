#include <bits/stdc++.h>
#include <iostream>
#include "cryptanalysis.h"
#include "util.h"
using namespace std;

int main()
{
	cout << endl;
	freopen("output.txt", "w", stdout);
	freopen("input.txt", "r", stdin);

	string cipher;
	getline(cin, cipher);

	cout << "-----------------------------------------------" << "\n";
	cout << "INPUT CIPHER TEXT IS: " << "\n"
		 << "\n";
	cout << cipher << "\n";
	cout << "\n";

	Util util;
	string cipherText = util.cleanCipherText(cipher);

	CryptAnalysis c(cipherText, cipher);
	unordered_set<int> val = c.kasaskiMethod();
	int keySize = c.indexOfCoincidence(val, 2, 10);

	vector<string> keysList = c.mutualIndexOfCoincidence(keySize);
	// for (const string &i : keysList)
	// {
	// 	cout << i << " ";
	// }
	return 0;
};
