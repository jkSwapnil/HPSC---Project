#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include<iterator>
#include<algorithm>
#include<iostream>

using namespace std; 
int main() 
{
	int m, n;
    m = 1000, n = 6;
    float *data, *X, *Y;
    data = new float[m*n];
	ifstream fin("a.csv");
    string line, word;
    vector<string> row;
	int i = 0, flag=0;

	while(!fin.eof())
	{   

        line.clear();
        getline(fin,line);
        row.clear();
		stringstream s(line);
		while (getline(s, word, ',')) { 
            row.push_back(word); 
        }
        if(i==0 && flag ==0){
            flag = 1;
            continue;
        }
        for(int j=0; j<n-1; ++j){
        	data[i*n + j] = stof(row[j]);
        }
        data[i*n + 5] = 1; 
        i+=1;
	}

    X = new float[m*1];
    Y = new float[m*(n-1)];
	for(int i=0; i<m; i++){
    	printf("\n");
        for(int j=0; j<n; j++){
            printf("\t%f",data[i*n + j]);
        }
    }	

    delete [] data;
    fin.close();
	return 0;
}