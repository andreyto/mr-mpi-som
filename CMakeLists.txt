PROJECT(MRSOM)

SET( ${PROJECT_NAME}_MAJOR_VERSION 2 )
SET( ${PROJECT_NAME}_MINOR_VERSION 1 )
SET( ${PROJECT_NAME}_PATCH_LEVEL 0 )

cmake_minimum_required(VERSION 2.6)

#set(MPI_ROOT /export/openmpi)
#set(MPI_COMPILER ${MPI_ROOT}/bin/mpicxx)
#set(MPI_LIBRARY MPI_LIBRARY-NOTFOUND) 
find_package(MPI REQUIRED)
#find_package(Pthreads)

if (MPI_FOUND)   
    find_program(MPI_C_COMPILER NAMES mpicc DOC "MPI C compiler.")
    mark_as_advanced(MPI_C_COMPILER)
    if (MPI_C_COMPILER)
        set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    endif (MPI_C_COMPILER)

    find_program(MPI_CXX_COMPILER NAMES mpic++ mpicxx mpiCC DOC "MPI C++ compiler.")
    mark_as_advanced(MPI_CXX_COMPILER)
    if (MPI_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    endif (MPI_CXX_COMPILER)
  
    #SET(CMAKE_C_FLAGS "-g -Wall -pg -O -I./mrmpi")
    #SET(CMAKE_CXX_FLAGS "-g -Wall -pg -O -I./mrmpi")
    #SET(CMAKE_C_FLAGS "-Wall -pg -O -I./mrmpi")
    #SET(CMAKE_CXX_FLAGS "-Wall -pg -O -I./mrmpi")
    #SET(CMAKE_C_COMPILER /export/openmpi/bin/mpicc)
    #SET(CMAKE_CXX_COMPILER /export/openmpi/bin/mpicxx)
    #SET(CMAKE_C_COMPILER MPI_C_COMPILER)
    #SET(CMAKE_CXX_COMPILER MPI_CXX_COMPILER)

    #CMAKE_EXE_LINKER_FLAGS
    #CMAKE_SHARED_LINKER_FLAGS
    #CMAKE_MODULE_LINKER_FLAGS
    #SET(CMAKE_EXE_LINKER_FLAGS "-g -O -L./mrmpi")

    add_subdirectory(src)
    add_subdirectory(src/mrmpi)
else (MPI_FOUND) 
    message(FATAL_ERROR "OpenMPI is not found on this system")    
endif (MPI_FOUND)   

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


##SET(SRC mrsom)
##ADD_EXECUTABLE(mrsom ${SRC})
##SET(CMAKE_C_COMPILER /export/openmpi/bin/mpicc)
##SET(CMAKE_CXX_COMPILER /export/openmpi/bin/mpicxx)
##target_link_libraries(mrsom mpi)
 
###add_definitions(-Wall -W -ggdb -I./mrmpi)
###INSTALL(FILES mrsom DESTINATION ./bin)
 
#### Top-Level CmakeLists.txt
###PROJECT( mrsom )
###ADD_SUBDIRECTORY( node )
###ADD_SUBDIRECTORY( som )
##ADD_SUBDIRECTORY( mrmpi )

### specific to me
##include_directories( /export/openmpi/include )
##include_directories(${MRSOM_SOURCE_DIR}/mrmpi)
##link_directories(${MRSOM_BINARY_DIR}/mrmpi)
##LINK_LIBRARIES(${MRSOM_SOURCE_DIR}/mrmpi/libmrmpi.a)

##find_package(openmpi)
##find_package(GLUT)
#find_package(OpenGL)

#ADD_EXECUTABLE(mrsom mrsom.cpp)
#SET(CMAKE_C_COMPILER /export/openmpi/bin/mpicc)
#SET(CMAKE_CXX_COMPILER /export/openmpi/bin/mpicxx)
##SET(CCFLAGS "-g -Wall -O -I./mrmpi")
##SET(CMAKE_CXX_FLAGS "-g -Wall")
##SET(LINK /export/openmpi/bin/mpicxx)
##SET(LINKFLAGS "-g -O -L./mrmpi")
##SET(USRLIB "-lmrmpi")
##SET(SYSLIB "-lpthread")
##SET(LIB "./mrmpi/libmrmpi.a")

#target_link_libraries(mrsom mpi)
#target_link_libraries(mrsom pthread)
##target_link_libraries(mrsom ./mrmpi/libmrmpi.a)
#target_link_libraries(mrsom ./mrmpi)
##add_definitions(-Wall -W -ggdb -I./mrmpi -L/export/openmpi/lib -L./mrmpi)
