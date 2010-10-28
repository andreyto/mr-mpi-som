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

#ifndef SPOOL_H
#define SPOOL_H

#include "stdio.h"

namespace MAPREDUCE_NS {

class Spool {
 public:
  uint64_t nkv;                      // # of KV entries in entire spool file
  uint64_t esize;                    // size of all entries (with alignment)
  uint64_t fsize;                    // size of spool file

  char *page;                        // in-memory page
  int npage;                         // # of pages in Spool

  Spool(int, class MapReduce *, class Memory *, class Error *);
  ~Spool();

  void set_page(uint64_t, char *);
  void complete();
  void truncate(int, int, uint64_t);
  int request_info(char **);
  int request_page(int);
  void add(int, char *);
  void add(int, uint64_t, char *);

 private:
  class MapReduce *mr;
  class Memory *memory;
  class Error *error;

  uint64_t pagesize;            // size of page

  // in-memory page

  int nkey;                     // # of entries
  uint64_t size;                // current size of entries

  // virtual pages

  struct Page {
    uint64_t size;              // size of entries
    uint64_t filesize;          // rounded-up size for file I/O
    int nkey;                   // # of entries
  };

  Page *pages;                  // list of pages in Spool
  int maxpage;                  // max # of pages currently allocated

  // file info

  char *filename;               // filename to store Spool if needed
  int fileflag;                 // 1 if file exists, 0 if not
  FILE *fp;                     // file ptr

  // private methods

  void create_page();
  void write_page();
  void read_page(int);
  uint64_t roundup(uint64_t,int);
};

}

#endif
