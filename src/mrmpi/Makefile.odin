# MPI-based Makefile using mpic++ and mpicc on odin.

CC =		mpic++ -m64 
CCFLAGS =	-O2 -DMRMPI_FPATH=/localdisk1/scratch 
DEPFLAGS =	-M
ARCHIVE =	ar
ARFLAGS =	-rc

include Makefile.common

