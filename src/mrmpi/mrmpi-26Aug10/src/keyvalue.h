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

#ifndef KEY_VALUE_H
#define KEY_VALUE_H

#include "mpi.h"
#include "stdio.h"
#include "stdint.h"

namespace MAPREDUCE_NS {

class KeyValue {
  friend class MapReduce;

 public:
  uint64_t nkv;                   // # of KV pairs in entire KV on this proc
  uint64_t ksize;                 // exact size of all key data
  uint64_t vsize;                 // exact size of all value data
  uint64_t esize;                 // exact size of all data in KV
  uint64_t fsize;                 // size of KV file
  int msize;                      // size of largest KV pair across all procs

  char *page;                     // memory page
  int memtag;                     // page ID
  int npage;                      // # of pages in entire KV

  KeyValue(class MapReduce *, int, int, 
	   class Memory *, class Error *, MPI_Comm);
  ~KeyValue();

  void set_page();
  void set_page(uint64_t, char *, int);
  void truncate(int, int, uint64_t);
  void copy(KeyValue *);
  void append();
  void complete();
  void complete_dummy();
  int request_info(char **);
  int request_page(int, uint64_t &, uint64_t &, uint64_t &);

  void add(char *, int, char *, int);
  void add(int, char *, int, char *, int);
  void add(int, char *, int *, char *, int *);

  void print(int, int, int);

 private:
  MapReduce *mr;
  MPI_Comm comm;
  class Memory *memory;
  class Error *error;
  int me;

  uint64_t pagesize;                // size of page
  int kalign,valign;                // alignment for keys & values
  int talign;                       // alignment of entire KV pair
  int kalignm1,valignm1,talignm1;   // alignments-1 for masking
  int twolenbytes;                  // size of single key,value lengths

  // in-memory page

  int nkey;                         // # of KV pairs in page
  uint64_t keysize;                 // exact size of key data in page
  uint64_t valuesize;               // exact size of value data in page
  uint64_t alignsize;               // current size of page with alignment

  // virtual pages

  struct Page {
    uint64_t keysize;               // exact size of keys 
    uint64_t valuesize;             // exact size of values
    uint64_t exactsize;             // exact size of all data in page
    uint64_t alignsize;             // aligned size of all data in page
    uint64_t filesize;              // rounded-up alignsize for file I/O
    uint64_t fileoffset;            // summed filesize of all previous pages
    int nkey;                       // # of KV pairs
  };

  Page *pages;                      // list of pages
  int maxpage;                      // max # of pages currently allocated

  // file info

  char *filename;                   // filename to store KV if needed
  FILE *fp;                         // file ptr
  int fileflag;                     // 1 if file exists, 0 if not

  // private methods

  void add(KeyValue *);
  void add(int, char *);
  void add(char *);
  void add(int, char *, uint64_t, uint64_t, uint64_t);
  void add(int, char *, int, int);

  void init_page();
  void create_page();
  void write_page();
  void read_page(int, int);
  uint64_t roundup(uint64_t,int);
};

}

#endif
