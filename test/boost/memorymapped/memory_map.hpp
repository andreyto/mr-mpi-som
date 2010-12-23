#ifndef MEMORY_MAP_H
#define MEMORY_MAP_H 1

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/filesystem/operations.hpp>         // for real file size
#include <string>
#include <iostream>

namespace MMAP_AG {

typedef unsigned long int Len;

class memory_map {

private:
    std::string path_;     // path to the current file
    Len page_start_;       // the starting page pointer
    Len page_off_set_;     // current file pointer
    Len a_page_size_;      // a single page size
    Len mmap_size_;        // memory map size
    Len real_file_size_;   // real file size
    Len still_left_;       // still left in the file

    boost::iostreams::mapped_file_source m_file_; // current memory map source

public:

    memory_map(const std::string& path, unsigned int num_pages):
        path_(path),                                       // file path
        page_start_(0),                                    // starting page
        page_off_set_(0),                                  // starting page offset
        a_page_size_(m_file_.alignment()),                 // a single page size
        mmap_size_(a_page_size_*num_pages),                 // size mmap
        real_file_size_(boost::filesystem::file_size(path)),// real size file
        still_left_(real_file_size_),                      // still left in file
        m_file_(path,                                      // path
                mmap_size_ > real_file_size_ ?
                real_file_size_ : mmap_size_,              // map_size
                0) {               
                                            // initial offset
        std::cout << "Paged Size: " << m_file_.size() << std::endl;
        std::cout << "File Size: " << real_file_size_ << std::endl;
        std::cout << "a_page_size_: " << a_page_size_ << std::endl;
        std::cout << "mmap_size_: " << mmap_size_ << std::endl;
    };
    
    memory_map(const std::string& path):
        path_(path),                                       // file path
        page_start_(0),                                    // starting page
        page_off_set_(0),                                  // starting page offset
        //a_page_size_(m_file_.alignment()),                 // a single page size
        //mmap_size_(a_page_size_*num_pages),                 // size mmap
        real_file_size_(boost::filesystem::file_size(path)),// real size file
        still_left_(real_file_size_),                      // still left in file
        m_file_(path,                                      // path
                real_file_size_,              // map_size
                0) {               
                                            // initial offset
        std::cout << "Paged Size: " << m_file_.size() << std::endl;
        std::cout << "File Size: " << real_file_size_ << std::endl;
        //std::cout << "a_page_size_: " << a_page_size_ << std::endl;
        std::cout << "mmap_size_: " << mmap_size_ << std::endl;
    };
    
    bool is_open() {
        return m_file_.is_open();
    }
    
    Len size() {
        return m_file_.size();
    }
    
    const char* data() {
        return m_file_.data();
    }
    
    const Len residual() const {
        return still_left_;
    }
    
    void close() {
        m_file_.close();
    }

    //
    // reads into buffer, the specified num_bytes
    //
    Len read_bytes(void * buffer, Len num_bytes);
};



};

#endif
