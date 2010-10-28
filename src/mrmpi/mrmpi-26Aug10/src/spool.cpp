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

#include "stdlib.h"
#include "string.h"
#include "stdint.h"
#include "spool.h"
#include "mapreduce.h"
#include "memory.h"
#include "error.h"

using namespace MAPREDUCE_NS;

#define ALIGNFILE 512              // same as in mapreduce.cpp
#define PAGECHUNK 16

//#define SPOOL_DEBUG 1

/* ---------------------------------------------------------------------- */

Spool::Spool(int style, 
	     MapReduce *mr_caller, Memory *memory_caller, Error *error_caller)
{
  mr = mr_caller;
  memory = memory_caller;
  error = error_caller;

  filename = mr->file_create(style);
  fileflag = 0;
  fp = NULL;

  pages = NULL;
  npage = maxpage = 0;

  nkv = esize = fsize = 0;
  nkey = size = 0;
}

/* ---------------------------------------------------------------------- */

Spool::~Spool()
{
  memory->sfree(pages);
  if (fileflag) {
    remove(filename);
    mr->hiwater(1,fsize);
  }
  delete [] filename;
}

/* ----------------------------------------------------------------------
   directly assign a chunk of memory to be the im-memory page for the Spool
------------------------------------------------------------------------- */

void Spool::set_page(uint64_t memsize, char *memblock)
{
  pagesize = memsize;
  page = memblock;
}

/* ----------------------------------------------------------------------
   complete the Spool after data has been added to it
   always write page to disk, unlike KV and KMV which can stay in memory
------------------------------------------------------------------------- */

void Spool::complete()
{
  create_page();
  write_page();
  fclose(fp);
  fp = NULL;

  npage++;
  nkey = size = 0;

  // set sizes for entire spool file

  nkv = esize = fsize = 0;
  for (int ipage = 0; ipage < npage; ipage++) {
    nkv += pages[ipage].nkey;
    esize += pages[ipage].size;
    fsize += pages[ipage].filesize;
  }

  mr->hiwater(0,fsize);

#ifdef SPOOL_DEBUG
  printf("SP Created %s: %d pages, %u entries, %g Mb\n",
  	 filename,npage,nkv,esize/1024.0/1024.0);
#endif
}

/* ----------------------------------------------------------------------
   truncate Spool at ncut,pagecut entry
   called by KMV::convert()
------------------------------------------------------------------------- */

void Spool::truncate(int pagecut, int ncut, uint64_t sizecut)
{
  if (ncut == 0) npage = pagecut;
  else {
    npage = pagecut+1;
    pages[pagecut].size = sizecut;
    pages[pagecut].filesize = roundup(sizecut,ALIGNFILE);
    pages[pagecut].nkey = ncut;
  }
}

/* ----------------------------------------------------------------------
   return # of pages and ptr to in-memory page
------------------------------------------------------------------------- */

int Spool::request_info(char **ptr)
{
  *ptr = page;
  return npage;
}

/* ----------------------------------------------------------------------
   ready a page of entries
   caller is looping over data in Spool
------------------------------------------------------------------------- */

int Spool::request_page(int ipage)
{
  read_page(ipage);

  // close file if last request

  if (ipage == npage-1) {
    fclose(fp);
    fp = NULL;
  }

  return pages[ipage].nkey;
}

/* ----------------------------------------------------------------------
   add a single entry
------------------------------------------------------------------------- */

void Spool::add(int nbytes, char *entry)
{
  // page is full, write to disk

  if (size+nbytes > pagesize) {
    create_page();
    write_page();
    npage++;
    nkey = size = 0;

    if (nbytes > pagesize) {
      printf("Spool size/limit: %d %d\n",nbytes,pagesize);
      error->one("Single entry exceeds Spool page size");
    }
  }

  memcpy(&page[size],entry,nbytes);
  size += nbytes;
  nkey++;
}

/* ----------------------------------------------------------------------
   add N entries of total nbytes
------------------------------------------------------------------------- */

void Spool::add(int n, uint64_t nbytes, char *entries)
{
  // page is full, write to disk

  if (size+nbytes > pagesize) {
    create_page();
    write_page();
    npage++;
    nkey = size = 0;

    if (nbytes > pagesize) {
      printf("Spool size/limit: %d %d\n",nbytes,pagesize);
      error->one("Single entry exceeds Spool page size");
    }
  }

  memcpy(&page[size],entries,nbytes);
  size += nbytes;
  nkey += n;
}

/* ----------------------------------------------------------------------
   create virtual page entry for in-memory page
------------------------------------------------------------------------- */

void Spool::create_page()
{
  if (npage == maxpage) {
    maxpage += PAGECHUNK;
    pages = (Page *) memory->srealloc(pages,maxpage*sizeof(Page),"SP:pages");
  }

  pages[npage].nkey = nkey;
  pages[npage].size = size;
  pages[npage].filesize = roundup(size,ALIGNFILE);
}

/* ----------------------------------------------------------------------
   write in-memory page to disk
------------------------------------------------------------------------- */

void Spool::write_page()
{
  if (fp == NULL) {
    fp = fopen(filename,"wb");
    if (fp == NULL) error->one("Could not open Spool file for writing");
    fileflag = 1;
  }

  fwrite(page,pages[npage].filesize,1,fp);
  mr->wsize += pages[npage].filesize;
}

/* ----------------------------------------------------------------------
   read ipage from disk
------------------------------------------------------------------------- */

void Spool::read_page(int ipage)
{
  if (fp == NULL) {
    fp = fopen(filename,"rb");
    if (fp == NULL) error->one("Could not open Spool file for reading");
  }

  fread(page,pages[ipage].filesize,1,fp);
  mr->rsize += pages[ipage].filesize;
}

/* ----------------------------------------------------------------------
   round N up to multiple of nalign and return it
------------------------------------------------------------------------- */

uint64_t Spool::roundup(uint64_t n, int nalign)
{
  if (n % nalign == 0) return n;
  n = (n/nalign + 1) * nalign;
  return n;
}
