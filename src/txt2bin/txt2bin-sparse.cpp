//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
//#
//#   See COPYING file distributed along with the MGTAXA package for the
//#   copyright and license terms.
//#
//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
/// For tokenizer
#include <boost/algorithm/string.hpp>

/// For str -> int, int -> str
#include <boost/lexical_cast.hpp>

/// For Boost memory mapped file
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/filesystem/operations.hpp>         /// for real file size
boost::iostreams::mapped_file_source MMAPFILE;     /// Read-only Boost mmap file

using namespace std;

/// Sparse structures and routines
typedef struct sparsetype {
    uint32_t index;
    float value;
} SPARSE_STRUCT_T;

typedef struct indextype {
    uint32_t position;
    uint32_t num_values;
    uint32_t num_values_accum;
} INDEX_STRUCT_T;


int main(int argc, char* argv[])
{
    if (argc != 5) {
        cout << "Usage: txt2bin-sparse input output numcols numrows\n";
        exit(0);
    }
    
    string inputFileName(argv[1]); 
    string outFileName(argv[2]); 
    uint32_t nDimen = atoi(argv[3]);
    uint32_t nVecs = atoi(argv[4]);
    
    FILE *fp;
    fp = fopen(argv[1], "r");
    
    ofstream outputBinFile((outFileName+"-sparse.bin").c_str(), ios::binary);
    ofstream outputIndexFile((outFileName+"-sparse.idx").c_str(), ios::binary);
    ofstream numValuesFile((outFileName+"-sparse.num").c_str());
    
    uint32_t totalNData = 0;
    for (uint32_t row = 0; row < nVecs; row++) {
        uint32_t nColsWritten = 0;
        INDEX_STRUCT_T irec;
        irec.position = outputBinFile.tellp();
        
        for (uint32_t col = 0; col < nDimen; col++) {
            float tmp = 0.0f;
            fscanf(fp, "%f", &tmp);          
            
            if (tmp != 0) {
                SPARSE_STRUCT_T item;
                item.index = col;
                item.value = tmp;
                outputBinFile.write((char*)&item, sizeof(item));
                nColsWritten++;
                totalNData++;
            }
        }
        irec.num_values = nColsWritten;
        irec.num_values_accum = totalNData;
        outputIndexFile.write((char *)&irec, sizeof(irec));
    }
    cout << "Total number of items = " << totalNData << endl;
    
    numValuesFile << totalNData << endl;
    outputBinFile.close();
    outputIndexFile.close();
    numValuesFile.close();
    fclose(fp); 
    
    cout << "INFO: Files generated\n";
    cout << "\tbin file: \t" << outFileName+"-sparse.bin" << endl;
    cout << "\tindex file: \t" << outFileName+"-sparse.idx" << endl;
    cout << "\tnum file: \t" << outFileName+"-sparse.num" << endl;
 
    
    return 0;
}


/// EOF
