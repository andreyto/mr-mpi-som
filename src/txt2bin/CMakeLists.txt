### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the MGTAXA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

add_executable(txt2bin txt2bin.cpp)
add_executable(txt2bin-sparse txt2bin-sparse.cpp)
target_link_libraries(txt2bin-sparse boost_iostreams)
target_link_libraries(txt2bin-sparse boost_filesystem)

#set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE distribtion)
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
