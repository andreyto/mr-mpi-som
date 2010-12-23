#include <stdio.h>

#include <iostream>
#include <fstream>
using namespace std;



int main(int argc, char* argv[])
{
   
    //float vvFeature[D];
    //cout << "sizeof vvFeature = " << sizeof vvFeature << endl;
    
    if (argc != 5) {
        cout << "Usage: txt2bin input output dimen numFeature\n";
        exit(0);
    }
    int len = atoi(argv[4]);
    int D = atoi(argv[3]);
    
    FILE *fp;
    fp = fopen(argv[1], "r");
    
    ofstream out(argv[2], ios::out | ios::binary);
    if (!out) {
        cout << "Cannot open file.";
        return 1;
    }

    for (uint64_t row = 0; row < len; row++) {
        for (uint64_t col = 0; col < D; col++) {
            float tmp = 0.0f;
            fscanf(fp, "%f", &tmp);
            //vvFeature[col] = tmp;
            
            //out.write((char *) &vvFeature, sizeof(float));           
            out.write((char *) &tmp, sizeof(float));  
        }
    }
    
    out.close();
    fclose(fp); 
    
    
    
    
    //float fnum[D];
    //ifstream in(argv[2], ios::in | ios::binary);
    //uint64_t numByte = 0;
    
    //while ( !in.eof() ) {
        //in.read((char *) &fnum, sizeof fnum);
        ////cout << in.gcount() << " bytes read\n";
        //numByte += in.gcount();
        //for (int i = 0; i < D; i++) // show values read from file
            //cout << fnum[i] << " ";
        //cout << endl;
    //}
    //in.close();
    //cout << "numByte = " << numByte << endl;
    
    /*

    double fnum[4] = {9.5, -3.4, 1.0, 2.1};
    int i;
 

    out.write((char *) &fnum, sizeof fnum);

    out.close();

    for (i = 0; i < 4; i++) // clear array
        fnum[i] = 0.0;

    ifstream in("numbers.dat", ios::in | ios::binary);
    in.read((char *) &fnum, sizeof fnum);

    // see how many bytes have been read
    cout << in.gcount() << " bytes read\n";

    for (i = 0; i < 4; i++) // show values read from file
        cout << fnum[i] << " ";
    cout << endl;

    in.close();
    */

    return 0;
}
