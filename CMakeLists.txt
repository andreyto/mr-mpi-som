PROJECT(MRSOM)

SET( ${PROJECT_NAME}_MAJOR_VERSION 4 )
SET( ${PROJECT_NAME}_MINOR_VERSION 0 )
SET( ${PROJECT_NAME}_PATCH_LEVEL 0 )

cmake_minimum_required(VERSION 2.6)
 
#find_package(Pthreads)

# Search for MPI ###########################
find_package( MPI REQUIRED )
if (MPI_FOUND)  
    MESSAGE("MPI information:")
    MESSAGE("   MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
    MESSAGE("   MPI_LIBRARY: ${MPI_LIBRARY}")
    MESSAGE("   MPI_LIBRARIES: ${MPI_LIBRARIES}")
    MESSAGE("   MPIEXEC: ${MPIEXEC}")
    MESSAGE("   MPI_C_COMPILER: ${MPI_C_COMPILER}")
    MESSAGE("   MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
 
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
else (MPI_FOUND) 
    message(FATAL_ERROR "Any MPI is not found on this system")    
endif (MPI_FOUND)   

# Search for Boost ###########################
#set(BOOST_ROOT "/usr")
find_package( Boost REQUIRED )

if (Boost_FOUND)
    MESSAGE("Boost information:")
    MESSAGE("   Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    MESSAGE("   Boost_LIBRARIES: ${Boost_LIBRARIES}")
    MESSAGE("   Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

    link_directories ( ${Boost_LIBRARY_DIRS} )
    include_directories ( ${Boost_INCLUDE_DIRS} )
else (Boost_FOUND)
    message(FATAL_ERROR "Any Boost lib is not found on this system")    
endif (Boost_FOUND)
###############################################


add_subdirectory(src)
add_subdirectory(src/mrmpi)
add_subdirectory(src/txt2bin)

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
