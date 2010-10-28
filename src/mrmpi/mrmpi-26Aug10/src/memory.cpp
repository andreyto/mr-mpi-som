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

#define _XOPEN_SOURCE 600        // needed for posix_memalign() in stdlib.h

#include "mpi.h"
#include "stdlib.h"
#include "stdio.h"
#include "memory.h"
#include "error.h"

using namespace MAPREDUCE_NS;

/* ---------------------------------------------------------------------- */

Memory::Memory(MPI_Comm comm)
{
  error = new Error(comm);
}

/* ---------------------------------------------------------------------- */

Memory::~Memory()
{
  delete error;
}

/* ----------------------------------------------------------------------
   safe malloc 
------------------------------------------------------------------------- */

void *Memory::smalloc(size_t n, const char *name)
{
  if (n == 0) return NULL;
  void *ptr = malloc(n);
  if (ptr == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate %lu bytes for array %s",n,name);
    error->one(str);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe malloc with alignment
------------------------------------------------------------------------- */

void *Memory::smalloc_align(size_t n, int nalign, const char *name)
{
  if (n == 0) return NULL;
  //void *ptr;
  //int ierror = posix_memalign(&ptr,nalign,n);
  //if (ierror) {
  void *ptr = malloc(n);
  if (ptr == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate %lu bytes for array %s",n,name);
    error->one(str);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe free 
------------------------------------------------------------------------- */

void Memory::sfree(void *ptr)
{
  if (ptr == NULL) return;
  free(ptr);
}

/* ----------------------------------------------------------------------
   safe realloc 
------------------------------------------------------------------------- */

void *Memory::srealloc(void *ptr, size_t n, const char *name)
{
  if (n == 0) {
    sfree(ptr);
    return NULL;
  }

  ptr = realloc(ptr,n);
  if (ptr == NULL) {
    char str[128];
    sprintf(str,"Failed to reallocate %lu bytes for array %s",n,name);
    error->one(str);
  }
  return ptr;
}
