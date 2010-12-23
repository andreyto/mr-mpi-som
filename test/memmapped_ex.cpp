    #include <iostream>
    #include <string>
    #include <cstring>
    #include <boost/iostreams/device/mapped_file.hpp>
    using std::cout;
    using std::endl;

    int main(int argc, char** argv) {
     const int blockSize = 64;
     bool writer = false;

     if(argc > 1) {
      if(!strcmp(argv[1], "w"))
       writer = true;
     }

     boost::iostreams::mapped_file_params  params;
     params.path = "map.dat";
    // params.length = 1024;     // default: all the file
     params.new_file_size = blockSize;

     if(writer) {
      cout << "Writer" << endl;
      params.mode = std::ios_base::out;
     }
     else {
      cout << "Reader" << endl;
      params.mode = std::ios_base::in;
     }

        boost::iostreams::mapped_file  mf;
        mf.open(params);

     if(writer)
     {
      char *block = mf.data();
      strcpy(block, "Test data block...\0");
      cout << "Written: " << block << endl;
     }
     else
     {
      cout << "Reading: " << mf.const_data() << endl;
     }

     mf.close();

        return 0;
    }
