#include "memory_map.hpp"

#include <string>
#include <sstream>
#include <iostream>

 

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    int DIMEN = atoi(argv[2]);
    //enum {buf_size = 4096, pages_used = 10};
    int buf_size = 4096;
    cout << "buf_size = " << buf_size << endl;
    int pages_used=2;

    char buffer[buf_size];

    MMAP_AG::Len l;

    MMAP_AG::memory_map m_file(argv[1], pages_used);
    
    float f;
    while (m_file.residual() > 0) {
        l = m_file.read_bytes(buffer, buf_size);
        cout << "l = " << l << endl;
        //buffer[l] = '\0';
        //std::cout << buffer;
        
        unsigned char *c1 = (unsigned char *)malloc(sizeof(float));
    
        for (int i = 0; i < sizeof(float); i++) {
            c1[i] = buffer[i];
        }
        f = *(float *)c1;
        cout << "f = " << f << endl;
        delete c1;
    }

    m_file.close();




    //boost::iostreams::mapped_file_source m_file2;
    //const std::string path(argv[1]);
    //unsigned long int fLen = 0;
    //fLen = boost::filesystem::file_size(path);
    ////cout << "path, fLen = " << path << " " << fLen << endl;
    //m_file2(path, fLen, 0);

    //m_file2.close();
    
    
    
    
    

    const std::string path(argv[1]);
    MMAP_AG::memory_map m_file2(path);
    //l = m_file.read_bytes(buffer, boost::filesystem::file_size(path)); 
    //cout << "l = " << l << endl;   
    if (m_file2.is_open()) {
        
        //cout << m_file2.data() << endl;
        unsigned long int fLen = 0;
        fLen = boost::filesystem::file_size(path);
        cout << "path, fLen = " << path << " " << fLen << endl;
        
        char *c1 = (char *)malloc(sizeof(float));
        for (int j = 0; j < fLen; j += 4) {
            for (int i = 0; i < sizeof(float); i++) {
                c1[i] = *(m_file2.data()+i+j);
            }
            f = *(float *)c1;
            cout << "f1 = " << f << endl;
        }
         
        
        delete c1;
        
        m_file2.close();
    }
    
    
    
    
    
    //MMAP_AG::memory_map m_file2(argv[2], pages_used);
    //int buf_size2 = 4096;
    //cout << "buf_size2 = " << buf_size2 << endl;
    //char buffer2[buf_size2];
    //float f;

    //l = m_file2.read_bytes(buffer2, buf_size2);
    //cout << "l = " << l << endl;
    
    //unsigned char *c1 = (unsigned char *)malloc(sizeof(float));
    
    //for (int i = 0; i < sizeof(float); i++) {
        //c1[i] = buffer2[i];
    //}
    //f = *(float *)c1;
    //cout << "f = " << f << endl;
    
    //m_file2.close();
 

}

