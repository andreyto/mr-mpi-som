include_directories(${MRSOM_SOURCE_DIR}/mrmpi)
link_directories(${MRSOM_BINARY_DIR}/mrmpi)

#include_directories(/home/ssul/work/distros2/boost/)

#include_directories(${MRSOM_SOURCE_DIR}/src)
#include_directories(${MRSOM_SOURCE_DIR}/signal)

#SET(CCFLAGS "-g -Wall -O -I../mrmpi")
#SET(CMAKE_CXX_FLAGS "-g -Wall ")
#SET(LINK "/export/openmpi/bin/mpicxx /export/mrmpi/lib")
#SET(LINKFLAGS "-g -O -L../mrmpi")
#SET(USRLIB "-lmrmpi")
#SET(SYSLIB "-lpthread")
#SET(LIB "/export/mrmpi/libmrmpi.a")
#SET(DEPFLAGS "-M")
#SET(ARCHIVE "ar")
#SET(ARFLAGS "-rc")
LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${MRSOM_BINARY_DIR}/mrmpi )

#add_executable(som som.cpp)

#add_executable(som2 som2.cpp)

#if(CMAKE_COMPILER_IS_GNUCXX)
    #set(CMAKE_CXX_FLAGS "-O2")        ## Optimize
    ##set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
#endif()

#add_executable(mrsom mrsom.cpp)
#target_link_libraries(mrsom mpi)  
#target_link_libraries(mrsom mrmpi)
  
# Promoted to v4
#add_executable(mrsom2 mrsom2.cpp)
#target_link_libraries(mrsom2 mpi)  
#target_link_libraries(mrsom2 mrmpi)

#add_executable(mrsom0713 mrsom0713.cpp)
#target_link_libraries(mrsom0713 mpi)  
#target_link_libraries(mrsom0713 mrmpi)

#add_executable(mrsom3 mrsom3.cpp)
#target_link_libraries(mrsom3 mpi)  
#target_link_libraries(mrsom3 mrmpi)

# v4
#add_executable(mrsom4 mrsom4.cpp)
#target_link_libraries(mrsom4 mpi)  
#target_link_libraries(mrsom4 mrmpi)
 
# test compaction
#add_executable(mrsom4.1.0 mrsom4.1.0.cpp)
#target_link_libraries(mrsom4.1.0 mpi)  
#target_link_libraries(mrsom4.1.0 mrmpi)

# test compaction hybrid - only map use compaction
# Promoted to v5
#add_executable(mrsom4.1.1 mrsom4.1.1.cpp)
#target_link_libraries(mrsom4.1.1 mpi)  
#target_link_libraries(mrsom4.1.1 mrmpi)
 
# test d=0 && n=0
#add_executable(mrsom4.2.0 mrsom4.2.0.cpp)
#target_link_libraries(mrsom4.2.0 mpi)  
#target_link_libraries(mrsom4.2.0 mrmpi)
 
# test map all d weights in a list ==> result not good
#add_executable(mrsom4.2.1 mrsom4.2.1.cpp)
#target_link_libraries(mrsom4.2.1 mpi)  
#target_link_libraries(mrsom4.2.1 mrmpi)

# d=0 && n=0 check, hybrid compaction
#add_executable(mrsom5 mrsom5.cpp)
#target_link_libraries(mrsom5 mpi)  
#target_link_libraries(mrsom5 mrmpi)

# d=0 && n=0 check, hybrid compaction, BOOST
#add_executable(mrsom6 mrsom6.cpp memory_map.cpp)
#target_link_libraries(mrsom6 mpi)  
#target_link_libraries(mrsom6 mrmpi)
#target_link_libraries(mrsom6 boost_iostreams)
#target_link_libraries(mrsom6 boost_filesystem)

 #key -> integer
#add_executable(mrsom7 mrsom7.cpp memory_map.cpp)
#target_link_libraries(mrsom7 mpi)  
#target_link_libraries(mrsom7 mrmpi)
#target_link_libraries(mrsom7 boost_iostreams)
#target_link_libraries(mrsom7 boost_filesystem)

# New by AT
#add_executable(mrsom8 mrsom8.cpp)
#target_link_libraries(mrsom8 mpi)  
#target_link_libraries(mrsom8 mrmpi)
#target_link_libraries(mrsom8 boost_iostreams)
#target_link_libraries(mrsom8 boost_filesystem)
# For Ranger
#target_link_libraries(mrsom8 boost_iostreams-gcc44-mt)
#target_link_libraries(mrsom8 boost_filesystem-gcc44-mt)

# 12.06.2010 MPI_reduce
#add_executable(mrsom9 mrsom9.cpp)
#target_link_libraries(mrsom9 mpi)  
#target_link_libraries(mrsom9 mrmpi)
#target_link_libraries(mrsom9 boost_iostreams)
#target_link_libraries(mrsom9 boost_filesystem)
# For Ranger
#target_link_libraries(mrsom9 boost_iostreams-gcc44-mt)
#target_link_libraries(mrsom9 boost_filesystem-gcc44-mt)

add_executable(mrsom mrsom.cpp)
target_link_libraries(mrsom mpi)  
target_link_libraries(mrsom mrmpi)
target_link_libraries(mrsom boost_iostreams)
target_link_libraries(mrsom boost_filesystem)

#set(CMAKE_BUILD_TYPE Release)

SET(CMAKE_BUILD_TYPE distribtion)
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#SET(CMAKE_C_FLAGS_DEBUG -g "-O0")
