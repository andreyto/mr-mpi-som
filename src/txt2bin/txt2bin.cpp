//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
//#
//#   See COPYING file distributed along with the MGTAXA package for the
//#   copyright and license terms.
//#
//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;


int main(int argc, char* argv[])
{
    if (argc != 5) {
        cout << "Usage: txt2bin input output numcols numrows\n";
        exit(0);
    }
    int len = atoi(argv[4]);
    int D = atoi(argv[3]);
    
    FILE *fp;
    fp = fopen(argv[1], "r");
    
    ofstream out(argv[2], ios::out | ios::binary);
    if (!out) {
        cerr << "Error: Cannot open file.";
        return 1;
    }

    for (uint64_t row = 0; row < len; row++) {
        for (uint64_t col = 0; col < D; col++) {
            float tmp = 0.0f;
            fscanf(fp, "%f", &tmp);          
            out.write((char *) &tmp, sizeof(float));  
        }
    }
    
    out.close();
    fclose(fp); 
    
    return 0;
}
