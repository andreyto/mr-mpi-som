PROJECT(MRSOM)

SET( ${PROJECT_NAME}_MAJOR_VERSION 4 )
SET( ${PROJECT_NAME}_MINOR_VERSION 0 )
SET( ${PROJECT_NAME}_PATCH_LEVEL 0 )

cmake_minimum_required(VERSION 2.6)
 
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

# To specify user-defined Boost lib location,
set(BOOST_ROOT "/home/ssul/work/distros2/boost/boost_1_45_0")

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

# EOF
