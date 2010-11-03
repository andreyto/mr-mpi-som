include_directories(${MRSOM_SOURCE_DIR}/mrmpi)
link_directories(${MRSOM_BINARY_DIR}/mrmpi)

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

#add_executable(mrsom mrsom.cpp)
#target_link_libraries(mrsom mpi)  
#target_link_libraries(mrsom mrmpi)
  
#add_executable(mrsom2 mrsom2.cpp)
#target_link_libraries(mrsom2 mpi)  
#target_link_libraries(mrsom2 mrmpi)

#add_executable(mrsom0713 mrsom0713.cpp)
#target_link_libraries(mrsom0713 mpi)  
#target_link_libraries(mrsom0713 mrmpi)

add_executable(mrsom3 mrsom3.cpp)
target_link_libraries(mrsom3 mpi)  
target_link_libraries(mrsom3 mrmpi)

add_executable(mrsom4 mrsom4.cpp)
target_link_libraries(mrsom4 mpi)  
target_link_libraries(mrsom4 mrmpi)

