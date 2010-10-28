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

#ifndef IRREGULAR_H
#define IRREGULAR_H

#include "mpi.h"
#include "stdint.h"

namespace MAPREDUCE_NS {

class Irregular {
 public:
  Irregular(int, class Memory *, class Error *, MPI_Comm);
  ~Irregular();

  uint64_t cssize,crsize;    // total send/recv bytes for one exchange

  int setup(int, int *, int *, int *, uint64_t, double &);
  void exchange(int, int *, char **, int *, int *, char *, char *);

 private:
  int me,nprocs;
  int all2all;
  class Memory *memory;
  class Error *error;
  MPI_Comm comm;             // MPI communicator for all communication

  // all2all and custom settings

  int *sendbytes;            // bytes to send to each proc, including self
  int *sdispls;              // proc offset into clumped send buffer
  int *recvbytes;            // bytes to recv from each proc, including self
  int *rdispls;              // proc offset into recv buffer
  int *senddatums;           // # of datums to send each proc, including self
  int *one;                  // 1 for each proc, for MPI call
  int ndatum;                // # of total datums I recv, including self

  // custom settings

  int self;                  // 0 = no data to copy to self, 1 = yes
  int nsend;                 // # of messages to send w/out self
  int nrecv;                 // # of messages to recv w/out self
  int *sendprocs;            // list of procs to send to w/out self
  int *recvprocs;            // list of procs to recv from w/out self
  MPI_Request *request;      // MPI requests for posted recvs
  MPI_Status *status;        // MPI statuses for Waitall

  void exchange_all2all(int, int *, char **, int *, char *, char *);
  void exchange_custom(int, int *, char **, int *, char *, char *);

};

}

#endif
