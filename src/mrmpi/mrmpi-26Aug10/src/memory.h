/* ----------------------------------------------------------------------
   MR-MPI = MapReduce-MPI library
   http://www.cs.sandia.gov/~sjplimp/mapreduce.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the modified Berkeley Software Distribution (BSD) License.

   See the README file in the top-level MapReduce directory.
------------------------------------------------------------------------- */

#ifndef MEMORY_H
#define MEMORY_H

#include "mpi.h"

namespace MAPREDUCE_NS {

class Memory {
 public:
  Memory(MPI_Comm);
  ~Memory();

  void *smalloc(size_t, const char *);
  void *smalloc_align(size_t, int, const char *);
  void sfree(void *);
  void *srealloc(void *, size_t, const char *);

 private:
  class Error *error;
};

}

#endif
