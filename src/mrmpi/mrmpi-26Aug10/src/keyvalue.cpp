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

#include "mpi.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "stdint.h"
#include "keyvalue.h"
#include "mapreduce.h"
#include "memory.h"
#include "error.h"

using namespace MAPREDUCE_NS;

#define MIN(A,B) ((A) < (B)) ? (A) : (B)
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

#define ROUNDUP(A,B) (char *) (((uint64_t) A + B) & ~B);

#define ALIGNFILE 512              // same as in mapreduce.cpp
#define PAGECHUNK 16
#define INTMAX 0x7FFFFFFF

enum{KVFILE,KMVFILE,SORTFILE,PARTFILE,SETFILE};   // same as in mapreduce.cpp

/* ---------------------------------------------------------------------- */

KeyValue::KeyValue(MapReduce *mr_caller, int memkalign, int memvalign,
		   Memory *memory_caller, Error *error_caller,
		   MPI_Comm comm_caller)
{
  mr = mr_caller;
  memory = memory_caller;
  error = error_caller;
  comm = comm_caller;
  MPI_Comm_rank(comm,&me);

  filename = mr->file_create(KVFILE);
  fileflag = 0;
  fp = NULL;

  pages = NULL;
  npage = maxpage = 0;

  // talign = max of (kalign,valign,int)

  kalign = memkalign;
  valign = memvalign;
  talign = MAX(kalign,valign);
  talign = MAX(talign,sizeof(int));

  kalignm1 = kalign-1;
  valignm1 = valign-1;
  talignm1 = talign-1;

  twolenbytes = 2*sizeof(int);

  nkv = ksize = vsize = esize = fsize = 0;
  init_page();
}

/* ---------------------------------------------------------------------- */

KeyValue::~KeyValue()
{
  memory->sfree(pages);
  if (fileflag) {
    remove(filename);
    mr->hiwater(1,fsize);
  }
  delete [] filename;
}

/* ----------------------------------------------------------------------
   trigger KV to request an available page of memory
------------------------------------------------------------------------- */

void KeyValue::set_page()
{
  page = mr->mymalloc(1,pagesize,memtag);
}

/* ----------------------------------------------------------------------
   directly assign a chunk of memory to be the im-memory page for the KV
------------------------------------------------------------------------- */

void KeyValue::set_page(uint64_t memsize, char *memblock, int tag)
{
  pagesize = memsize;
  page = memblock;
  memtag = tag;
}

/* ----------------------------------------------------------------------
   truncate KeyValue at ncut,pagecut entry
   called by KMV::convert()
------------------------------------------------------------------------- */

void KeyValue::truncate(int pagecut, int ncut, uint64_t sizecut)
{
  if (ncut == 0) npage = pagecut;
  else {
    npage = pagecut+1;
    pages[pagecut].alignsize = sizecut;
    pages[pagecut].filesize = roundup(sizecut,ALIGNFILE);
    pages[pagecut].nkey = ncut;
  }
}

/* ----------------------------------------------------------------------
   copy contents of another KV into me, one page at a time
   input KV should never be self
   input KV will have same pagesize and alignment as me
   called by MR::copy() and MR::map(mr)
------------------------------------------------------------------------- */

void KeyValue::copy(KeyValue *kv)
{
  if (kv == this) error->all("Cannot perform KeyValue copy on self");

  // pages will be loaded into memory assigned to other KV
  // write_page() will write them from that page to my file

  char *page_hold = page;
  int npage_other = kv->request_info(&page);

  for (int ipage = 0; ipage < npage_other-1; ipage++) {
    nkey = kv->request_page(ipage,keysize,valuesize,alignsize);
    create_page();
    write_page();
    npage++;
  }

  // last page needs to be copied to my memory before calling complete()
  // reset my page to my memory

  nkey = kv->request_page(npage_other-1,keysize,valuesize,alignsize);
  memcpy(page_hold,page,alignsize);
  msize = kv->msize;
  complete();
  page = page_hold;
}

/* ----------------------------------------------------------------------
   prepare the KV for appending of new KV pairs
   called by MR::add(), MR::gather(), MR::map(addflag=1)
------------------------------------------------------------------------- */

void KeyValue::append()
{
  if (npage == 0) return;

  int ipage = npage-1;

  // read last page from file if necessary
  // decrement MR filesize as if file were being deleted

  if (fileflag) {
    read_page(ipage,1);
    mr->hiwater(1,fsize);
  }

  // set in-memory settings from virtual page settings

  nkey = pages[ipage].nkey;
  keysize = pages[ipage].keysize;
  valuesize = pages[ipage].valuesize;
  alignsize = pages[ipage].alignsize;

  // delete last page from pages data structures since will append to it

  npage--;
}

/* ----------------------------------------------------------------------
   complete the KV after data has been added to it
   called by MR methods after creating & populating a KV
------------------------------------------------------------------------- */

void KeyValue::complete()
{
  create_page();

  // if disk file exists, write last page, close file

  if (fileflag) {
    write_page();
    fclose(fp);
    fp = NULL;
  }

  npage++;
  init_page();

  // set sizes for entire KV

  nkv = ksize = vsize = esize = fsize = 0;
  for (int ipage = 0; ipage < npage; ipage++) {
    nkv += pages[ipage].nkey;
    ksize += pages[ipage].keysize;
    vsize += pages[ipage].valuesize;
    esize += pages[ipage].exactsize;
  }

  if (fileflag) {
    fsize = pages[npage-1].fileoffset + pages[npage-1].filesize;
    mr->hiwater(0,fsize);
  }

  // msize is max across all procs, for entire KV

  int tmp = msize;
  MPI_Allreduce(&tmp,&msize,1,MPI_INT,MPI_MAX,comm);
}

/* ----------------------------------------------------------------------
   dummy complete for a KV that is already complete
   called by a proc while other procs call complete()
   simple invokes an Allreduce() to match Allreudce() in complete()
------------------------------------------------------------------------- */

void KeyValue::complete_dummy()
{
  int tmp = msize;
  MPI_Allreduce(&tmp,&msize,1,MPI_INT,MPI_MAX,comm);
}

/* ----------------------------------------------------------------------
   return # of pages and ptr to in-memory page
------------------------------------------------------------------------- */

int KeyValue::request_info(char **ptr)
{
  *ptr = page;
  return npage;
}

/* ----------------------------------------------------------------------
   ready one page of KV data
   caller is looping over data in KV
------------------------------------------------------------------------- */

int KeyValue::request_page(int ipage, uint64_t &keysize_page,
			   uint64_t &valuesize_page,
			   uint64_t &alignsize_page)
{
  // load page from file if necessary

  if (fileflag) read_page(ipage,0);

  // close file if last page

  if (ipage == npage-1 && fileflag) {
    fclose(fp);
    fp = NULL;
  }

  keysize_page = pages[ipage].keysize;
  valuesize_page = pages[ipage].valuesize;
  alignsize_page = pages[ipage].alignsize;

  return pages[ipage].nkey;
}

/* ----------------------------------------------------------------------
   add a single key/value pair
   called by user appmap() or appreduce() or appcompress()
------------------------------------------------------------------------- */

void KeyValue::add(char *key, int keybytes, char *value, int valuebytes)
{
  char *iptr = &page[alignsize];
  char *kptr = iptr + twolenbytes;
  kptr = ROUNDUP(kptr,kalignm1);
  char *vptr = kptr + keybytes;
  vptr = ROUNDUP(vptr,valignm1);
  char *nptr = vptr + valuebytes;
  nptr = ROUNDUP(nptr,talignm1);
  int kvbytes = nptr - iptr;

  // size of KV pair cannot exceed int size

  if (nptr-iptr > INTMAX)
    error->one("Single key/value pair exceeds int size");

  // page is full, write to disk
  // full page = pagesize exceeded or INTMAX KV pairs

  if (alignsize + kvbytes > pagesize || nkey == INTMAX) {
    if (alignsize == 0) {
      printf("KeyValue pair size/limit: %d %u\n",kvbytes,pagesize);
      error->one("Single key/value pair exceeds page size");
    }

    create_page();
    write_page();
    npage++;
    init_page();
    add(key,keybytes,value,valuebytes);
    return;
  }

  *((int *) iptr) = keybytes;
  *((int *) (iptr+sizeof(int))) = valuebytes;
  memcpy(kptr,key,keybytes);
  memcpy(vptr,value,valuebytes);

  nkey++;
  keysize += keybytes;
  valuesize += valuebytes;
  alignsize += kvbytes;
  msize = MAX(msize,kvbytes);
}

/* ----------------------------------------------------------------------
   add N fixed-length key/value pairs
   called by user appmap() or appreduce() or appcomress()
------------------------------------------------------------------------- */

void KeyValue::add(int n, char *key, int keybytes,
		   char *value, int valuebytes)
{
  int koffset = 0;
  int voffset = 0;

  for (int i = 0; i < n; i++) {
    add(&key[koffset],keybytes,&value[voffset],valuebytes);
    koffset += keybytes;
    voffset += valuebytes;
  }
}

/* ----------------------------------------------------------------------
   add N variable-length key/value pairs
   called by user appmap() or appreduce() or appcompress()
------------------------------------------------------------------------- */

void KeyValue::add(int n, char *key, int *keybytes,
		   char *value, int *valuebytes)
{
  uint64_t koffset = 0;
  uint64_t voffset = 0;

  for (int i = 0; i < n; i++) {
    add(&key[koffset],keybytes[i],&value[voffset],valuebytes[i]);
    koffset += keybytes[i];
    voffset += valuebytes[i];
  }
}

/* ----------------------------------------------------------------------
   add key/value pairs from another KV
   input KV should never be self
   input KV may or may not have same alignment as me
   called by MR::add()
------------------------------------------------------------------------- */

void KeyValue::add(KeyValue *kv)
{
  if (kv == this) error->all("Cannot perform KeyValue add on self");

  int kalign_other = kv->kalign;
  int valign_other = kv->valign;

  // which add() to call depends on same or different alignment

  int nkey_other;
  uint64_t keysize_other,valuesize_other,alignsize_other;

  char *page_other;
  int npage_other = kv->request_info(&page_other);

  for (int ipage = 0; ipage < npage_other; ipage++) {
    nkey_other = kv->request_page(ipage,keysize_other,valuesize_other,
				  alignsize_other);
    if (kalign == kalign_other && valign == valign_other)
      add(nkey_other,page_other,keysize_other,valuesize_other,alignsize_other);
    else
      add(nkey_other,page_other,kalign_other,valign_other);
  }

  msize = MAX(msize,kv->msize);
}

/* ----------------------------------------------------------------------
   add N KV pairs from another buffer without specified sizes
   determine sizes and call add() with sizes
   input buf should never be own in-memory page
   input buf has same alignment as me
   called by MR::aggregate()
------------------------------------------------------------------------- */

void KeyValue::add(int n, char *buf)
{
  int keybytes,valuebytes;

  uint64_t keysize_buf = 0;
  uint64_t valuesize_buf = 0;
  char *ptr = buf;

  for (int i = 0; i < n; i++) {
    keybytes = *((int *) ptr);
    valuebytes = *((int *) (ptr+sizeof(int)));;

    keysize_buf += keybytes;
    valuesize_buf += valuebytes;

    ptr += twolenbytes;
    ptr = ROUNDUP(ptr,kalignm1);
    ptr += keybytes;
    ptr = ROUNDUP(ptr,valignm1);
    ptr += valuebytes;
    ptr = ROUNDUP(ptr,talignm1);
  }

  uint64_t alignsize_buf = ptr - buf;
  add(n,buf,keysize_buf,valuesize_buf,alignsize_buf);
}

/* ----------------------------------------------------------------------
   add a single KV pair from another buffer without specified sizes
   determine 2 sizes and call add() with sizes
   input buf should never be own in-memory page
   input buf has same alignment as me
   called by MR::merge()
------------------------------------------------------------------------- */

void KeyValue::add(char *ptr)
{
  int keybytes = *((int *) ptr);
  int valuebytes = *((int *) (ptr+sizeof(int)));;
  ptr += twolenbytes;
  ptr = ROUNDUP(ptr,kalignm1);
  char *key = ptr;
  ptr += keybytes;
  char *value = ROUNDUP(ptr,valignm1);
  add(key,keybytes,value,valuebytes);
}

/* ----------------------------------------------------------------------
   add N KV pairs from another buffer with specified sizes
   input buf should never be own in-memory page
   input buf has same alignment as me so add in chunks
   called by MR::gather(), add(kv), add(n,buf)
------------------------------------------------------------------------- */

void KeyValue::add(int n, char *buf,
		   uint64_t keysize_buf, uint64_t valuesize_buf,
		   uint64_t alignsize_buf)
{
  int nkeychunk,keybytes,valuebytes,kvbytes;
  uint64_t keychunk,valuechunk,chunksize;
  char *ptr,*ptr_begin,*ptr_end,*ptr_start;

  // break data into chunks that fit into current and successive pages
  // full page = pagesize exceeded or INTMAX KV pairs
  // search for breakpoint by scanning KV pairs

  ptr = buf;
  int nlimit = INTMAX - nkey;

  while (alignsize + alignsize_buf > pagesize || n > nlimit) {
    ptr_begin = ptr;
    ptr_end = ptr_begin + (pagesize-alignsize);
    nkeychunk = 0;
    keychunk = valuechunk = 0;

    while (1) {
      ptr_start = ptr;
      keybytes = *((int *) ptr);
      valuebytes = *((int *) (ptr+sizeof(int)));;

      ptr += twolenbytes;
      ptr = ROUNDUP(ptr,kalignm1);
      ptr += keybytes;
      ptr = ROUNDUP(ptr,valignm1);
      ptr += valuebytes;
      ptr = ROUNDUP(ptr,talignm1);
      kvbytes = ptr - ptr_start;

      if (ptr > ptr_end) break;
      if (nkeychunk == nlimit) break;

      nkeychunk++;
      keychunk += keybytes;
      valuechunk += valuebytes;
    }

    if (kvbytes > pagesize) {
      printf("KeyValue pair size/limit: %d %u\n",kvbytes,pagesize);
      error->one("Single key/value pair exceeds page size");
    }

    ptr = ptr_start;
    chunksize = ptr - ptr_begin;
    memcpy(&page[alignsize],ptr_begin,chunksize);

    nkey += nkeychunk;
    keysize += keychunk;
    valuesize += valuechunk;
    alignsize += chunksize;

    create_page();
    write_page();
    npage++;
    init_page();
    
    n -= nkeychunk;
    keysize_buf -= keychunk;
    valuesize_buf -= valuechunk;
    alignsize_buf -= chunksize;
    nlimit = INTMAX;
  }

  // add remainder to in-memory page

  memcpy(&page[alignsize],ptr,alignsize_buf);

  nkey += n;
  keysize += keysize_buf;
  valuesize += valuesize_buf;
  alignsize += alignsize_buf;
}

/* ----------------------------------------------------------------------
   add N KV pairs from another buffer with specified sizes
   input buf should never be own in-memory page
   input buf has different alignment from me so must add one by one
   called by add(kv)
------------------------------------------------------------------------- */

void KeyValue::add(int n, char *buf, int kalign_buf, int valign_buf)
{
  int keybytes,valuebytes;
  char *key,*value;

  int talign_buf = MAX(kalign_buf,valign_buf);
  talign_buf = MAX(talign_buf,sizeof(int));

  int kalignm1_buf = kalign_buf-1;
  int valignm1_buf = valign_buf-1;
  int talignm1_buf = talign_buf-1;

  char *ptr = buf;

  for (int i = 0; i < n; i++) {
    keybytes = *((int *) ptr);
    valuebytes = *((int *) (ptr+sizeof(int)));;

    ptr += twolenbytes;
    ptr = ROUNDUP(ptr,kalignm1_buf);
    key = ptr;
    ptr += keybytes;
    ptr = ROUNDUP(ptr,valignm1_buf);
    value = ptr;
    ptr += valuebytes;
    ptr = ROUNDUP(ptr,talignm1_buf);

    add(key,keybytes,value,valuebytes);
  }
}

/* ----------------------------------------------------------------------
   create virtual page entry for in-memory page
------------------------------------------------------------------------- */

void KeyValue::init_page()
{
  nkey = 0;
  keysize = valuesize = 0;
  alignsize = 0;
  msize = 0;
}

/* ----------------------------------------------------------------------
   create virtual page entry for in-memory page
------------------------------------------------------------------------- */

void KeyValue::create_page()
{
  if (npage == maxpage) {
    maxpage += PAGECHUNK;
    pages = (Page *) memory->srealloc(pages,maxpage*sizeof(Page),"KV:pages");
  }

  pages[npage].nkey = nkey;
  pages[npage].keysize = keysize;
  pages[npage].valuesize = valuesize;
  pages[npage].exactsize = ((uint64_t) nkey)*twolenbytes + 
    keysize + valuesize;
  pages[npage].alignsize = alignsize;
  pages[npage].filesize = roundup(alignsize,ALIGNFILE);

  if (npage)
    pages[npage].fileoffset = 
      pages[npage-1].fileoffset + pages[npage-1].filesize;
  else
    pages[npage].fileoffset = 0;
}

/* ----------------------------------------------------------------------
   write in-memory page to disk
   do a seek since may be overwriting an arbitrary page due to append
------------------------------------------------------------------------- */

void KeyValue::write_page()
{
  if (fp == NULL) {
    fp = fopen(filename,"wb");
    if (fp == NULL) {
      char msg[1023];
      sprintf(msg, "Could not open KeyValue file %s for writing.", filename);
      error->one(msg);
    }
    fileflag = 1;
  }

  uint64_t fileoffset = pages[npage].fileoffset;
  fseek(fp,fileoffset,SEEK_SET);
  fwrite(page,pages[npage].filesize,1,fp);
  mr->wsize += pages[npage].filesize;
}

/* ----------------------------------------------------------------------
   read ipage from disk
   do a seek since may be reading last page
------------------------------------------------------------------------- */

void KeyValue::read_page(int ipage, int writeflag)
{
  if (fp == NULL) {
    if (writeflag) fp = fopen(filename,"r+b");
    else fp = fopen(filename,"rb");
    if (fp == NULL) error->one("Could not open KeyValue file for reading");
  }

  uint64_t fileoffset = pages[ipage].fileoffset;
  fseek(fp,fileoffset,SEEK_SET);
  fread(page,pages[ipage].filesize,1,fp);
  mr->rsize += pages[ipage].filesize;
}

/* ----------------------------------------------------------------------
   round N up to multiple of nalign and return it
------------------------------------------------------------------------- */

uint64_t KeyValue::roundup(uint64_t n, int nalign)
{
  if (n % nalign == 0) return n;
  n = (n/nalign + 1) * nalign;
  return n;
}

/* ----------------------------------------------------------------------
   debug print of each KV pair with nstride
------------------------------------------------------------------------- */

void KeyValue::print(int nstride, int kflag, int vflag)
{
  int keybytes,valuebytes;
  uint64_t dummy1,dummy2,dummy3;
  char *ptr,*key,*value,*ptr_start;

  int istride = 0;

  for (int ipage = 0; ipage < npage; ipage++) {
    nkey = request_page(ipage,dummy1,dummy2,dummy3);
    ptr = page;
    for (int i = 0; i < nkey; i++) {
      ptr_start = ptr;
      keybytes = *((int *) ptr);
      valuebytes = *((int *) (ptr+sizeof(int)));;

      ptr += twolenbytes;
      ptr = ROUNDUP(ptr,kalignm1);
      key = ptr;
      ptr += keybytes;
      ptr = ROUNDUP(ptr,valignm1);
      value = ptr;
      ptr += valuebytes;
      ptr = ROUNDUP(ptr,talignm1);

      istride++;
      if (istride != nstride) continue;
      istride = 0;

      printf("KV pair: proc %d, sizes %d %d",me,keybytes,valuebytes);

      printf(", key ");
      if (kflag == 0) printf("NULL");
      else if (kflag == 1) printf("%d",*(int *) key);
      else if (kflag == 2) printf("%lu",*(uint64_t *) key);
      else if (kflag == 3) printf("%g",*(float *) key);
      else if (kflag == 4) printf("%g",*(double *) key);
      else if (kflag == 5) printf("%s",key);
      else if (kflag == 6) printf("%d %d",
				  *(int *) key,
				  *(int *) (key+sizeof(int)));
      else if (kflag == 7) printf("%lu %lu",
				  *(uint64_t *) key,
				  *(uint64_t *) (key+sizeof(uint64_t)));

      printf(", value ");
      if (vflag == 0) printf("NULL");
      else if (vflag == 1) printf("%d",*(int *) value);
      else if (vflag == 2) printf("%lu",*(uint64_t *) value);
      else if (vflag == 3) printf("%g",*(float *) value);
      else if (vflag == 4) printf("%g",*(double *) value);
      else if (vflag == 5) printf("%s",value);
      else if (vflag == 6) printf("%d %d",
				  *(int *) value,
				  *(int *) (value+sizeof(int)));
      else if (vflag == 7) printf("%lu %lu",
				  *(uint64_t *) value,
				  *(uint64_t *) (value+sizeof(uint64_t)));

      printf("\n");
    }
  }
}
