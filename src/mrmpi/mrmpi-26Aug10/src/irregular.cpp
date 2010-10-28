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

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "irregular.h"
#include "memory.h"
#include "error.h"

using namespace MAPREDUCE_NS;

#define MIN(A,B) ((A) < (B)) ? (A) : (B)
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

#define INTMAX 0x7FFFFFFF

/* ---------------------------------------------------------------------- */

Irregular::Irregular(int all2all_caller, Memory *memory_caller,
		     Error *error_caller, MPI_Comm comm_caller)
{
  all2all = all2all_caller;

  memory = memory_caller;
  error = error_caller;
  comm = comm_caller;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  sendbytes = new int[nprocs];
  sdispls = new int[nprocs];
  recvbytes = new int[nprocs];
  rdispls = new int[nprocs];
  senddatums = new int[nprocs];
  one = new int[nprocs];
  for (int i = 0; i < nprocs; i++) one[i] = 1;

  sendprocs = new int[nprocs];
  recvprocs = new int[nprocs];
  request = new MPI_Request[nprocs];
  status = new MPI_Status[nprocs];
}

/* ---------------------------------------------------------------------- */

Irregular::~Irregular()
{
  delete [] sendbytes;
  delete [] sdispls;
  delete [] recvbytes;
  delete [] rdispls;
  delete [] senddatums;
  delete [] one;

  delete [] sendprocs;
  delete [] recvprocs;
  delete [] request;
  delete [] status;
}

/* ----------------------------------------------------------------------
   setup irregular communication for all2all or custom
   n = # of datums contributed by this proc
   proclist = which proc each datum is to be sent to
   sizes = byte count of each datum
   recvlimit = max allowed size of received data
   return # of datums I recv
   set fraction = 1.0 if can recv all datums without exceeding two limits
   else set it to estimated fraction I can recv
   limit #1 = total volume of send data exceeds INTMAX
   limit #2 = total volume of recv data exceeds min(recvlimit,INTMAX)
   2nd limit also insures # of received datums cannot exceed INTMAX
   extra data is setup for custom communication:
     sendprocs = list of nsend procs to send to
     recvprocs = list of nrecv procs to recv from
     reorder = contiguous send indices for each send, self copy is last
------------------------------------------------------------------------- */

int Irregular::setup(int n, int *proclist, int *sizes, int *reorder,
		     uint64_t recvlimit, double &fraction)
{
  recvlimit = MIN(recvlimit,INTMAX);

  // compute sendbytes and sdispls

  for (int i = 0; i < nprocs; i++) sendbytes[i] = 0;
  for (int i = 0; i < n; i++) sendbytes[proclist[i]] += sizes[i];

  sdispls[0] = 0;
  uint64_t sendtotal = sendbytes[0];
  for (int i = 1; i < nprocs; i++) {
    sdispls[i] = sdispls[i-1] + sendbytes[i-1];
    sendtotal += sendbytes[i];
  }

  // error return if any proc's send total > INTMAX

  uint64_t sendtotalmax;
  MPI_Allreduce(&sendtotal,&sendtotalmax,1,MPI_UNSIGNED_LONG,MPI_MAX,comm);
  if (sendtotalmax > INTMAX) {
    fraction = ((double) INTMAX) / sendtotal;
    return 0;
  }

  // compute recvbytes and rdispls

  MPI_Alltoall(sendbytes,1,MPI_INT,recvbytes,1,MPI_INT,comm);

  rdispls[0] = 0;
  uint64_t recvtotal = recvbytes[0];
  for (int i = 1; i < nprocs; i++) {
    rdispls[i] = rdispls[i-1] + recvbytes[i-1];
    recvtotal += recvbytes[i];
  }

  // error return if any proc's recv total > min(recvlimit,INTMAX)
  
  uint64_t recvtotalmax;
  MPI_Allreduce(&recvtotal,&recvtotalmax,1,MPI_UNSIGNED_LONG,MPI_MAX,comm);
  if (recvtotalmax > recvlimit) {
    fraction = ((double) recvlimit) / recvtotal;
    return 0;
  }

  // successful setup
  // compute senddatums
  // nrecv = total # of datums I receive, guaranteed to be < INTMAX

  cssize = sendtotal - sendbytes[me];
  crsize = recvtotal - recvbytes[me];

  for (int i = 0; i < nprocs; i++) senddatums[i] = 0;
  for (int i = 0; i < n; i++) senddatums[proclist[i]]++;
  MPI_Reduce_scatter(senddatums,&ndatum,one,MPI_INT,MPI_SUM,comm);

  // if all2all, done

  if (all2all) {
    fraction = 1.0;
    return ndatum;
  }

  // if custom, setup additional data strucs
  // sendprocs,recvprocs = lists of procs to send to and recv from
  // begin lists with iproc > me and wrap around
  // reorder = contiguous send indices for each proc I send to
  // let s0 = senddatums[sendprocs[0]], s1 = senddatums[sendprocs[1]], etc
  // reorder[0:s0-1] = indices of datums in 1st message
  // reorder[s0:s0+s1-1] = indices of datums in 2nd message, etc
  // proc2send[i] = which send (0 to nsend-1) goes to proc I
  // offset[i] = running offset into reorder for each send (0 to nsend-1)

  int *proc2send = new int[nprocs];

  nsend = nrecv = 0;
  int iproc = me;
  for (int i = 1; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    if (sendbytes[iproc]) {
      proc2send[iproc] = nsend;
      sendprocs[nsend++] = iproc;
    }
    if (recvbytes[iproc]) recvprocs[nrecv++] = iproc;
  }

  if (sendbytes[me]) {
    self = 1;
    proc2send[me] = nsend;
  } else self = 0;
  
  int *offset = new int[nprocs];
  offset[0] = 0;
  for (int i = 1; i <= nsend; i++)
    offset[i] = offset[i-1] + senddatums[sendprocs[i-1]];

  int j;
  for (int i = 0; i < n; i++) {
    j = proclist[i];
    reorder[offset[proc2send[j]]++] = i;
  }

  delete [] proc2send;
  delete [] offset;

  fraction = 1.0;
  return ndatum;
}

/* ----------------------------------------------------------------------
   perform irregular communication via all2all or custom
   n = # of datums contributed by this proc
   proclist (for all2all) = which proc each datum is to be sent to
   sizes = byte count of each datum
   reorder (for custom) = contiguous send indices for each send
   copy = buffer to pack send datums into
   recv = buffer to recv all datums into
------------------------------------------------------------------------- */

void Irregular::exchange(int n, int *proclist, char **ptrs, int *sizes, 
			 int *reorder, char *copy, char *recv)
{
  if (all2all) exchange_all2all(n,proclist,ptrs,sizes,copy,recv);
  else exchange_custom(n,reorder,ptrs,sizes,copy,recv);
}

/* ----------------------------------------------------------------------
   wrapper on MPI_Alltoallv()
   first copy datums from ptrs into copy buf in correct order via proclist
------------------------------------------------------------------------- */

void Irregular::exchange_all2all(int n, int *proclist, char **ptrs,
				 int *sizes, char *copy, char *recv)
{
  int i,iproc;

  char **cptrs = new char*[nprocs];
  for (i = 0; i < nprocs; i++)
    cptrs[i] = &copy[sdispls[i]];

  for (int i = 0; i < n; i++) {
    iproc = proclist[i];
    memcpy(cptrs[iproc],ptrs[i],sizes[i]);
    cptrs[iproc] += sizes[i];
  }

  delete [] cptrs;

  MPI_Alltoallv(copy,sendbytes,sdispls,MPI_BYTE,
		recv,recvbytes,rdispls,MPI_BYTE,comm);
}

/* ----------------------------------------------------------------------
   custom all2all communication
   post all receives
   copying datums for one send into copy buf in correct order via indices
   copy self data while waiting for receives
   indices are 0 to N-1, contiguous for each proc to send to, self copy is last
------------------------------------------------------------------------- */

void Irregular::exchange_custom(int n, int *indices, char **ptrs, int *sizes,
				char *copy, char *recv)
{
  int i,j,iproc;
  char *ptr;

  // post all receives

  for (int irecv = 0; irecv < nrecv; irecv++) {
    iproc = recvprocs[irecv];
    MPI_Irecv(&recv[rdispls[iproc]],recvbytes[iproc],MPI_BYTE,
    	      iproc,0,comm,&request[irecv]);
  }

  // barrier to insure receives are posted

  MPI_Barrier(comm);

  // send each message, packing copy buf with needed datums

  int index = 0;
  for (int isend = 0; isend < nsend; isend++) {
    iproc = sendprocs[isend];
    ptr = copy;
    n = senddatums[iproc];
    for (i = 0; i < n; i++) {
      j = indices[index++];
      memcpy(ptr,ptrs[j],sizes[j]);
      ptr += sizes[j];
    }
    MPI_Send(copy,sendbytes[iproc],MPI_BYTE,iproc,0,comm);
  }

  // copy self data directly to recv buf

  if (self)
    ptr = &recv[rdispls[me]];
    n = senddatums[me];
    for (i = 0; i < n; i++) {
      j = indices[index++];
      memcpy(ptr,ptrs[j],sizes[j]);
      ptr += sizes[j];
    }

  // wait on all incoming messages

  if (nrecv) MPI_Waitall(nrecv,request,status);
}
