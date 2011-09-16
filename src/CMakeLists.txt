### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the MGTAXA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

include_directories(${MRSOM_SOURCE_DIR}/mrmpi)
link_directories(${MRSOM_BINARY_DIR}/mrmpi)
LINK_DIRECTORIES(${LINK_DIRECTORIES} ${MRSOM_BINARY_DIR}/mrmpi)

add_executable(mrsom mrsom.cpp mrsom.hpp)
target_link_libraries(mrsom mpi)  
target_link_libraries(mrsom mrmpi)

target_link_libraries(mrsom boost_iostreams)
target_link_libraries(mrsom boost_filesystem)
target_link_libraries(mrsom boost_program_options)

# For Ranger at TACC
#target_link_libraries(mrsom boost_iostreams-gcc44-mt)
#target_link_libraries(mrsom boost_filesystem-gcc44-mt)
#target_link_libraries(mrsom boost_program_options-gcc44-mt)

#set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE distribtion)
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#SET(CMAKE_C_FLAGS_DEBUG -g "-O0")
