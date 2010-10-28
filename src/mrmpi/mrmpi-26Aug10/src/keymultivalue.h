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

#ifndef KEY_MULTIVALUE_H
#define KEY_MULTIVALUE_H

#include "mpi.h"
#include "stdio.h"
#include "stdint.h"

namespace MAPREDUCE_NS {

class KeyMultiValue {
 public:
  uint64_t nkmv;                // # of KMV pairs in entire KMV on this proc
  uint64_t ksize;               // exact size of all key data
  uint64_t vsize;               // exact size of all multivalue data
  uint64_t esize;               // total exact size of entire KMV
  uint64_t fsize;               // size of KMV file

  char *page;                   // in-memory page
  int memtag;                   // page ID
  int npage;                    // # of pages in entire KMV

  KeyMultiValue(class MapReduce *, int, int,
		class Memory *, class Error *, MPI_Comm);
  ~KeyMultiValue();

  void set_page();
  void copy(KeyMultiValue *);
  void complete();
  int request_info(char **);
  int request_page(int, int, uint64_t &, uint64_t &, uint64_t &);
  uint64_t multivalue_blocks(int, int &);
  void overwrite_page(int);
  void close_file();

  void clone(class KeyValue *);
  void collapse(char *, int, class KeyValue *);
  void convert(class KeyValue *);

  void print(int, int, int);

 private:
  MapReduce *mr;
  MPI_Comm comm;
  class Memory *memory;
  class Error *error;
  int me;

  uint64_t pagesize;                 // size of page
  int kalign,valign;                 // alignment for keys & multivalues
  int talign;                        // alignment of entire KMV pair
  int ualign;                        // alignment of Unique
  int kalignm1,valignm1;             // alignments-1 for masking
  int talignm1,ualignm1;
  int twolenbytes;                   // size of key & value lengths
  int threelenbytes;                 // size of nvalue & key & value lengths

  // in-memory page

  int nkey;                          // # of KMV pairs in page
  uint64_t nvalue;                   // # of values in all KMV mvalues in page
  uint64_t keysize;                  // exact size of key data in page
  uint64_t valuesize;                // exact size of multivalue data in page
  uint64_t alignsize;                // current size of page with alignment

  // virtual pages

  struct Page {
    uint64_t keysize;           // exact size of keys 
    uint64_t valuesize;         // exact size of multivalues
    uint64_t exactsize;         // exact size of all data in page
    uint64_t alignsize;         // aligned size of all data in page
    uint64_t filesize;          // rounded-up alignsize for file I/O
    uint64_t fileoffset;        // summed filesize of all previous pages
    uint64_t nvalue_total;      // total # of values for multi-page KMV header
    int nkey;                   // # of KMV pairs
    int nblock;                 // # of value blocks for multi-page KMV header
  };

  Page *pages;                  // list of pages
  int maxpage;                  // max # of pages currently allocated

  // unique keys

  int nunique;               // current # of unique keys
  int ukeyoffset;            // offset from start of Unique to where key starts

  struct Unique {
    uint64_t nvalue;         // # of values associated with this key
    uint64_t mvbytes;        // total size of values associated with this key
    int *soffset;            // ptr to start of value sizes in KMV page
    char *voffset;           // ptr to start of values in KMV page
    Unique *next;            // ptr to next key in this hash bucket
    int keybytes;            // size of this key
    int set;                 // which KMV set this key will be part of
  };

  // hash of unique keys

  Unique **buckets;     // ptr to 1st key in each hash bucket
  int hashmask;         // bit mask for mapping hashed key into hash buckets
                        // nbuckets = hashmask + 1
  uint64_t bucketbytes; // byte size of hash buckets

  char *memunique;      // ptr to where memory for hash+Uniques starts
  char *ustart;         // ptr to where memory for Uniques starts
  char *ustop;          // ptr to where memory for Uniques stops

  // file info

  int fileflag;         // 1 if file exists, 0 if not
  char *filename;       // filename to store KMV if needed
  FILE *fp;             // file ptr

  // partitions of KV data per unique list

  struct Partition {
    class KeyValue *kv;      // primary KV storing pairs for this partition
    class Spool *sp;         // secondary Spool of pairs if re-partitioned
    class Spool *sp2;        // tertiary Spool of pairs if re-partitioned
    int sortbit;             // bit from hi-end that partitioning was done on
  };

  Partition *partitions;
  int npartition,maxpartition;

  // sets of unique keys per KMV page

  struct Set {
    class KeyValue *kv;     // KV pairs for set can be in KV and/or Spool(s)
    class Spool *sp;
    class Spool *sp2;
    Unique *first;          // ptr to first Unique in set
    int nunique;            // # of Uniques in set
    int extended;           // 1 if set contains one Unique -> multi-page KMV
  };

  Set *sets;
  int nset,maxset;

  // memory management for Spool pages

  char *readpage;           // convert() does all reading from this page
  int minspool;             // minimum allowed size for a spool page

  int npages_mr;            // # of MR pages I have allocated
  int *tag_mr;              // page IDs for MR pages
  char **page_mr;           // ptrs to MR pages
  uint64_t sizespool;       // size of spool page
  int spoolperpage;         // # of spool pages per MR page
  int nquery;               // # of requested spool pages on this iteration

  // private methods

  void add(char *, int, char *, int);
  void collapse_one(char *, int, class KeyValue *, uint64_t);
  void collapse_many(char *, int, class KeyValue *);

  void kv2unique(int);
  int unique2kmv_all();
  void unique2kmv_extended(int);
  void unique2kmv_set(int);
  void partition2sets(int);
  void kv2kmv(int);
  void kv2kmv_extended(int);

  class Spool *augment_partition(int);
  class Spool *create_partition(int);
  char *chunk_allocate();
  Unique *find(int, char *, int, Unique *&);
  int hash(char *, int);

  void init_page();
  void create_page();
  void write_page();
  void read_page(int, int);
  uint64_t roundup(uint64_t, int);

  void spool_memory(class KeyValue *);
  void spool_request(int, int);
  char *spool_malloc(int, uint64_t &);
  void spool_free();
};

}

#endif
