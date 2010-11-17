#include "memory_map.hpp"


namespace MMAP_AG {

//
// reads into buffer, the specified num_bytes
//   num_bytes must be a multiple of page size
//
Len memory_map::read_bytes(void * buffer, Len num_bytes)
{

    assert(num_bytes < mmap_size_); // dont read more than a page
    assert((num_bytes % a_page_size_) == 0); // multiple of the page

    if (num_bytes > still_left_)    // don't read more than what
        num_bytes = still_left_;      // we have available

    const Len end_pointer = page_off_set_ + num_bytes;
    if (end_pointer >= mmap_size_) { // repage

        page_start_ += page_off_set_;// record next page start
        m_file_.close();             // close current page
        m_file_.open(path_, mmap_size_, page_start_);  // open next page
        page_off_set_ = 0;           // zero offset
    }
    memcpy(buffer, m_file_.data() + page_off_set_, num_bytes);
    page_off_set_ += num_bytes;    // increment this page off_set
    still_left_ -= num_bytes;      // still left in file

    return num_bytes;              // bytes we read
};


}


