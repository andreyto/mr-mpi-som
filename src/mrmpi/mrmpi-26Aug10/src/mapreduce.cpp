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
#include "ctype.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "sys/types.h"
#include "sys/stat.h"
#include "mapreduce.h"
#include "keyvalue.h"
#include "keymultivalue.h"
#include "spool.h"
#include "irregular.h"
#include "hash.h"
#include "memory.h"
#include "error.h"

using namespace MAPREDUCE_NS;

// allocate space for static class variables and initialize them

MapReduce *MapReduce::mrptr;
int MapReduce::instances_now = 0;
int MapReduce::instances_ever = 0;
int MapReduce::mpi_finalize_flag = 0;
uint64_t MapReduce::rsize = 0;
uint64_t MapReduce::wsize = 0;
uint64_t MapReduce::cssize = 0;
uint64_t MapReduce::crsize = 0;
double MapReduce::commtime = 0.0;

// prototypes for non-class functions

void map_file_standalone(int, KeyValue *, void *);
int compare_standalone(const void *, const void *);

#define MIN(A,B) ((A) < (B)) ? (A) : (B)
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

#define ROUNDUP(A,B) (char *) (((uint64_t) A + B) & ~B);

#define MAXLINE 1024
#define ALIGNFILE 512         // same as in other classes
#define FILECHUNK 128
#define VALUECHUNK 128
#define MBYTES 64
#define ALIGNKV 4
#define INTMAX 0x7FFFFFFF

enum{KVFILE,KMVFILE,SORTFILE,PARTFILE,SETFILE};

/* ----------------------------------------------------------------------
   construct using caller's MPI communicator
   perform no MPI_init() and no MPI_Finalize()
------------------------------------------------------------------------- */

MapReduce::MapReduce(MPI_Comm caller)
{
  instances_now++;
  instances_ever++;
  instance_me = instances_ever;

  comm = caller;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  defaults();
}

/* ----------------------------------------------------------------------
   construct without MPI communicator, use MPI_COMM_WORLD
   perform MPI_Init() if not already initialized
   perform no MPI_Finalize()
------------------------------------------------------------------------- */

MapReduce::MapReduce()
{
  instances_now++;
  instances_ever++;
  instance_me = instances_ever;

  int flag;
  MPI_Initialized(&flag);

  if (!flag) {
    int argc = 0;
    char **argv = NULL;
    MPI_Init(&argc,&argv);
  }

  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  defaults();
}

/* ----------------------------------------------------------------------
   construct without MPI communicator, use MPI_COMM_WORLD
   perform MPI_Init() if not already initialized
   perform MPI_Finalize() if final instance is destructed
------------------------------------------------------------------------- */

MapReduce::MapReduce(double dummy)
{
  instances_now++;
  instances_ever++;
  instance_me = instances_ever;
  mpi_finalize_flag = 1;

  int flag;
  MPI_Initialized(&flag);

  if (!flag) {
    int argc = 0;
    char **argv = NULL;
    MPI_Init(&argc,&argv);
  }

  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  defaults();
}

/* ----------------------------------------------------------------------
   free all memory
   if finalize_flag is set and this is last instance, then finalize MPI
------------------------------------------------------------------------- */

MapReduce::~MapReduce()
{
  delete [] fpath;

  for (int i = 0; i < npage; i++)
    if (memcount[i]) memory->sfree(memptr[i]);
  memory->sfree(memptr);
  memory->sfree(memused);
  memory->sfree(memcount);

  delete memory;
  delete error;
  delete kv;
  delete kmv;

  instances_now--;
  if (verbosity) mr_stats(verbosity);
  if (instances_now == 0 && verbosity) cummulative_stats(verbosity,1);
  if (mpi_finalize_flag && instances_now == 0) MPI_Finalize();
}

/* ----------------------------------------------------------------------
   default settings
------------------------------------------------------------------------- */

void MapReduce::defaults()
{
  memory = new Memory(comm);
  error = new Error(comm);

  mapstyle = 0;
  all2all = 1;
  verbosity = 0;
  timer = 0;

#ifdef MRMPI_MEMSIZE
  memsize = MRMPI_MEMSIZE;
#else
  memsize = MBYTES;
#endif

  minpage = 0;
  maxpage = 0;
  keyalign = valuealign = ALIGNKV;

#ifdef MRMPI_FPATH
#define _QUOTEME(x) #x
#define QUOTEME(x) _QUOTEME(x)
#define QMRMPI_FPATH QUOTEME(MRMPI_FPATH)
  int n = strlen(QMRMPI_FPATH) + 1;
  fpath = new char[n];
  strcpy(fpath,QMRMPI_FPATH);
#else
  fpath = new char[2];
  strcpy(fpath,".");
#endif

  collateflag = 0;
  fcounter_kv = fcounter_kmv = fcounter_sort = 
    fcounter_part = fcounter_set = 0;

  twolenbytes = 2*sizeof(int);
  kmv_block_valid = 0;

  allocated = 0;
  memptr = NULL;
  memused = NULL;
  memcount = NULL;
  npage = 0;
  fsize = 0;
  fsizemax = 0;

  kv = NULL;
  kmv = NULL;

  if (sizeof(uint64_t) != 8 || sizeof(char *) != 8)
    error->all("Not compiled for 8-byte integers and pointers");

  int mpisize;
  MPI_Type_size(MPI_UNSIGNED_LONG,&mpisize);
  if (mpisize != 8)
    error->all("MPI_UNSIGNED_LONG is not 8-byte data type");
}

/* ----------------------------------------------------------------------
   make a copy of myself and return it
   new MR object duplicates my settings and KV/KMV
------------------------------------------------------------------------- */

MapReduce *MapReduce::copy()
{
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  MapReduce *mrnew = new MapReduce(comm);

  mrnew->mapstyle = mapstyle;
  mrnew->all2all = all2all;
  mrnew->verbosity = verbosity;
  mrnew->timer = timer;
  mrnew->memsize = memsize;
  mrnew->minpage = minpage;
  mrnew->maxpage = maxpage;

  if (allocated) {
    mrnew->keyalign = kalign;
    mrnew->valuealign = valign;
  } else {
    mrnew->keyalign = keyalign;
    mrnew->valuealign = valuealign;
  }

  delete [] mrnew->fpath;
  int n = strlen(fpath) + 1;
  mrnew->fpath = new char[n];
  strcpy(mrnew->fpath,fpath);

  if (kv) mrnew->copy_kv(kv);
  if (kmv) mrnew->copy_kmv(kmv);

  if (kv) stats("Copy",0);
  if (kmv) stats("Copy",1);

  return mrnew;
}

/* ----------------------------------------------------------------------
   create my KV as copy of kv_src
   called by other MR's copy(), so my KV will not yet exist
------------------------------------------------------------------------- */

void MapReduce::copy_kv(KeyValue *kv_src)
{
  if (!allocated) allocate();
  kv = new KeyValue(this,kalign,valign,memory,error,comm);
  kv->set_page();
  kv->copy(kv_src);
}

/* ----------------------------------------------------------------------
   create my KMV as copy of kmvsrc
   called by other MR's copy(), so my KMV will not yet exist
------------------------------------------------------------------------- */

void MapReduce::copy_kmv(KeyMultiValue *kmv_src)
{
  if (!allocated) allocate();
  kmv = new KeyMultiValue(this,kalign,valign,memory,error,comm);
  kmv->set_page();
  kmv->copy(kmv_src);
}

/* ----------------------------------------------------------------------
   add KV pairs from another MR to my KV
------------------------------------------------------------------------- */

uint64_t MapReduce::add(MapReduce *mr)
{
  if (mr->kv == NULL) 
    error->all("MapReduce passed to add() does not have KeyValue pairs");
  if (mr == this) error->all("Cannot add to self");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (!allocated) allocate();
  if (kmv) myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  if (kv == NULL) {
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else {
    kv->append();
  }

  kv->add(mr->kv);
  kv->complete();

  stats("Add",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   aggregate a KV across procs to create a new KV
   initially, key copies can exist on many procs
   after aggregation, all copies of key are on same proc
   performed via parallel distributed hashing
   hash = user hash function (NULL if not provided)
   requires irregular all2all communication
------------------------------------------------------------------------- */

uint64_t MapReduce::aggregate(int (*hash)(char *, int))
{
  int i,nkey_send,keybytes,valuebytes,nkey_recv;
  int start,stop,done,mydone;
  int memtag_cdpage,memtag_epage,memtag_fpage,memtag_gpage;
  uint64_t dummy,dummy1,dummy2,dummy3;
  double timestart,fraction,minfrac;
  char *ptr,*key;
  int *proclist,*kvsizes,*reorder;
  char **kvptrs;

  if (kv == NULL) error->all("Cannot aggregate without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (nprocs == 1) {
    stats("Aggregate",0);
    return kv->nkv;
  }

  // new KV that will be created

  KeyValue *kvnew = new KeyValue(this,kalign,valign,memory,error,comm);
  kvnew->set_page();

  // irregular communicator

  Irregular *irregular = new Irregular(all2all,memory,error,comm);

  // pages of workspace memory, including extra allocated pages

  uint64_t twopage;
  char *cdpage = mymalloc(2,twopage,memtag_cdpage);
  char *epage = mymalloc(1,dummy,memtag_epage);
  char *fpage = mymalloc(1,dummy,memtag_fpage);
  char *gpage = mymalloc(1,dummy,memtag_gpage);

  // maxpage = max # of pages in any proc's KV

  char *page_send;
  int npage_send = kv->request_info(&page_send);
  int maxpage;
  MPI_Allreduce(&npage_send,&maxpage,1,MPI_INT,MPI_MAX,comm);

  // loop over pages, perform irregular comm on each

  for (int ipage = 0; ipage < maxpage; ipage++) {

    // load page of KV pairs

    if (ipage < npage_send)
      nkey_send = kv->request_page(ipage,dummy1,dummy2,dummy3);
    else nkey_send = 0;

    // set ptrs to workspace memory

    proclist = (int *) epage;
    kvsizes = &proclist[nkey_send];
    reorder = &proclist[2 * ((uint64_t) nkey_send)];
    kvptrs = (char **) fpage;

    // hash each key to a proc ID
    // via user-provided hash function or hashlittle()

    ptr = page_send;

    for (i = 0; i < nkey_send; i++) {
      kvptrs[i] = ptr;
      keybytes = *((int *) ptr);
      valuebytes = *((int *) (ptr+sizeof(int)));;

      ptr += twolenbytes;
      ptr = ROUNDUP(ptr,kalignm1);
      key = ptr;
      ptr += keybytes;
      ptr = ROUNDUP(ptr,valignm1);
      ptr += valuebytes;
      ptr = ROUNDUP(ptr,talignm1);

      kvsizes[i] = ptr - kvptrs[i];
      if (hash) proclist[i] = hash(key,keybytes) % nprocs;
      else proclist[i] = hashlittle(key,keybytes,nprocs) % nprocs;
    }

    // perform irregular comm of each proc's page of KV pairs
    // add received KV pairs to kvnew
    // no proc can receive more than 2 pages at once, else scale back
    // iterate until entire page is communicated by every proc

    start = 0;
    stop = nkey_send;

    done = 0;
    while (!done) {

      // attempt to communicate all KVs from start to stop
      // if overflows any proc, then scale back stop until succeed
      // if setup returns any fraction < 1.0, reset stop and try again
      // 0.9 is a conservative round-down factor
      // NOTE: is scale back guaranteed to eventually be successful?
      //       is this loop guaranteed to make progress (comm something)?
      //       what if all procs want to send 1 big datum to proc 0 but
      //         call cannot do it together?
      //         then all might round down to 0 ??

      timestart = MPI_Wtime();

      while (1) {
	nkey_recv = irregular->setup(stop-start,&proclist[start],
				     &kvsizes[start],&reorder[start],
				     twopage,fraction);

	MPI_Allreduce(&fraction,&minfrac,1,MPI_DOUBLE,MPI_MIN,comm);
	if (minfrac < 1.0) 
	  stop = static_cast<int> (start + 0.9*minfrac*(stop-start));
	else break;
      }

      irregular->exchange(stop-start,&proclist[start],&kvptrs[start],
			  &kvsizes[start],&reorder[start],gpage,cdpage);
      cssize += irregular->cssize;
      crsize += irregular->crsize;
      commtime += MPI_Wtime() - timestart;

      kvnew->add(nkey_recv,cdpage);

      // set start/stop to remainder of page and iterate
      // if all procs are at end of page, then done

      start = stop;
      stop = nkey_send;
      if (start == stop) mydone = 1;
      else mydone = 0;
      MPI_Allreduce(&mydone,&done,1,MPI_INT,MPI_MIN,comm);
    }
  }

  delete irregular;
  myfree(memtag_cdpage);
  myfree(memtag_epage);
  myfree(memtag_fpage);
  myfree(memtag_gpage);

  myfree(kv->memtag);
  delete kv;
  kv = kvnew;
  kv->complete();

  stats("Aggregate",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   broadcast the KV on proc root to all other procs
------------------------------------------------------------------------- */

uint64_t MapReduce::broadcast(int root)
{
  int npage_kv,memtag;
  char *buf;
  uint64_t dummy,sizes[4];

  if (kv == NULL) error->all("Cannot broadcast without KeyValue");
  if (root < 0 || root >= nprocs) error->all("Invalid root for broadcast");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (nprocs == 1) {
    stats("Broadcast",0);
    uint64_t nkeyall;
    MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
    return nkeyall;
  }

  // on non-root procs, delete existing KV and create empty KV
  
  double timestart = MPI_Wtime();

  if (me != root) {
    myfree(kv->memtag);
    delete kv;
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
    buf = mymalloc(1,dummy,memtag);
  } else npage_kv = kv->request_info(&buf);

  MPI_Bcast(&npage_kv,1,MPI_INT,root,comm);

  // broadcast KV data, one page at a time, non-root procs add to their KV

  for (int ipage = 0; ipage < npage_kv; ipage++) {
    if (me == root)
      sizes[0] = kv->request_page(ipage,sizes[1],sizes[2],sizes[3]);
    MPI_Bcast(sizes,4,MPI_UNSIGNED_LONG,root,comm);
    MPI_Bcast(buf,sizes[3],MPI_BYTE,root,comm);
    if (me == root) cssize += sizes[3];
    else {
      crsize += sizes[3];
      kv->add(sizes[0],buf,sizes[1],sizes[2],sizes[3]);
    }
  }

  if (me != root) myfree(memtag);

  commtime += MPI_Wtime() - timestart;
  if (me != root) kv->complete();
  else kv->complete_dummy();

  stats("Broadcast",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   clone KV to KMV so that KMV pairs are one-to-one copies of KV pairs
   each proc clones only its data
   assume each KV key is unique, but is not required
------------------------------------------------------------------------- */

uint64_t MapReduce::clone()
{
  if (kv == NULL) error->all("Cannot clone without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  kmv = new KeyMultiValue(this,kalign,valign,memory,error,comm);
  kmv->set_page();

  kmv->clone(kv);
  kmv->complete();

  myfree(kv->memtag);
  delete kv;
  kv = NULL;

  stats("Clone",1);

  uint64_t nkeyall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   close a KV that KV pairs were added to by another MR's map()
------------------------------------------------------------------------- */

uint64_t MapReduce::close()
{
  if (kv == NULL) error->all("Cannot close without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  kv->complete();

  stats("Complete",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   collapse KV into a KMV with a single key/value
   each proc collapses only its data
   new key = provided key name (same on every proc)
   new value = list of old key,value,key,value,etc
------------------------------------------------------------------------- */

uint64_t MapReduce::collapse(char *key, int keybytes)
{
  if (kv == NULL) error->all("Cannot collapse without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  kmv = new KeyMultiValue(this,kalign,valign,memory,error,comm);
  kmv->set_page();

  kmv->collapse(key,keybytes,kv);
  kmv->complete();

  myfree(kv->memtag);
  delete kv;
  kv = NULL;

  stats("Collapse",1);

  uint64_t nkeyall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   collate KV to create a KMV
   aggregate followed by a convert
   hash = user hash function (NULL if not provided)
------------------------------------------------------------------------- */

uint64_t MapReduce::collate(int (*hash)(char *, int))
{
  if (kv == NULL) error->all("Cannot collate without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  collateflag = 1;
  int verbosity_hold = verbosity;
  int timer_hold = timer;
  verbosity = timer = 0;

  aggregate(hash);
  convert();

  verbosity = verbosity_hold;
  timer = timer_hold;
  stats("Collate",1);
  collateflag = 0;
  fcounter_part = fcounter_set = 0;

  uint64_t nkeyall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   compress KV to create a smaller KV
   duplicate keys are replaced with a single key/value
   each proc compresses only its data
   create a KMV temporarily
   call appcompress() with each key/multivalue in KMV
   appcompress() returns single key/value to new KV
------------------------------------------------------------------------- */

uint64_t MapReduce::compress(void (*appcompress)(char *, int, char *, int,
						 int *, KeyValue *, void *),
			     void *appptr)
{
  if (kv == NULL) error->all("Cannot compress without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  kmv = new KeyMultiValue(this,kalign,valign,memory,error,comm);
  kmv->set_page();

  // KMV will delete kv and free its memory

  kmv->convert(kv);
  kmv->complete();

  kv = new KeyValue(this,kalign,valign,memory,error,comm);
  kv->set_page();

  uint64_t dummy;
  int memtag;
  char *mvpage = mymalloc(1,dummy,memtag);

  int nkey_kmv,nvalues,keybytes,mvaluebytes;
  uint64_t dummy1,dummy2,dummy3;
  int *valuesizes;
  char *ptr,*key,*multivalue;

  char *page_kmv;
  int npage_kmv = kmv->request_info(&page_kmv);
  char *page_hold = page_kmv;

  for (int ipage = 0; ipage < npage_kmv; ipage++) {
    nkey_kmv = kmv->request_page(ipage,0,dummy1,dummy2,dummy3);
    ptr = page_kmv;

    for (int i = 0; i < nkey_kmv; i++) {
      nvalues = *((int *) ptr);
      ptr += sizeof(int);

      if (nvalues > 0) {
	keybytes = *((int *) ptr);
	ptr += sizeof(int);
	mvaluebytes = *((int *) ptr);
	ptr += sizeof(int);
	valuesizes = (int *) ptr;
	ptr += ((uint64_t) nvalues) * sizeof(int);
	
	ptr = ROUNDUP(ptr,kalignm1);
	key = ptr;
	ptr += keybytes;
	ptr = ROUNDUP(ptr,valignm1);
	multivalue = ptr;
	ptr += mvaluebytes;
	ptr = ROUNDUP(ptr,talignm1);
	
	appcompress(key,keybytes,multivalue,nvalues,valuesizes,kv,appptr);

      } else {
	keybytes = *((int *) ptr);
	ptr += sizeof(int);
	ptr = ROUNDUP(ptr,kalignm1);
	key = ptr;

	// set KMV page to mvpage so key will not be overwritten
	// when multivalue_block() loads new pages of values

	kmv->page = mvpage;
	kmv_block_valid = 1;
	kmv_key_page = ipage;
	kmv_nvalue_total = kmv->multivalue_blocks(ipage,kmv_nblock);
	appcompress(key,keybytes,NULL,0,(int *) this,kv,appptr);
	kmv_block_valid = 0;
	ipage += kmv_nblock;
	kmv->page = page_hold;
      }
    }
  }

  kv->complete();
  myfree(memtag);

  // delete KMV
  // close is necessary b/c KMV files do not close themselves
  // since users may use request_page() via multivalue_block()

  kmv->close_file();
  myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  stats("Compress",0);
  fcounter_part = fcounter_set = 0;

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   convert KV to KMV
   duplicate keys are replaced with a single key/multivalue
   each proc converts only its data
   new key = old unique key
   new multivalue = concatenated list of all values for that key in KV
------------------------------------------------------------------------- */

uint64_t MapReduce::convert()
{
  if (kv == NULL) error->all("Cannot convert without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  kmv = new KeyMultiValue(this,kalign,valign,memory,error,comm);
  kmv->set_page();

  // KMV will delete kv and free its memory

  kmv->convert(kv);
  kmv->complete();

  kv = NULL;

  stats("Convert",1);
  if (!collateflag) fcounter_part = fcounter_set = 0;

  uint64_t nkeyall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   gather a distributed KV to a new KV on fewer procs
   numprocs = # of procs new KV resides on (0 to numprocs-1)
------------------------------------------------------------------------- */

uint64_t MapReduce::gather(int numprocs)
{
  int flag,npage_kv,memtag;
  char *buf;
  uint64_t dummy,sizes[4];
  MPI_Status status;
  MPI_Request request;

  if (kv == NULL) error->all("Cannot gather without KeyValue");
  if (numprocs < 1 || numprocs > nprocs) 
    error->all("Invalid processor count for gather");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (nprocs == 1 || numprocs == nprocs) {
    stats("Gather",0);
    uint64_t nkeyall;
    MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
    return nkeyall;
  }

  // lo procs collect key/value pairs from hi procs
  // lo procs are those with ID < numprocs
  // lo procs recv from set of hi procs with same (ID % numprocs)

  double timestart = MPI_Wtime();

  if (me < numprocs) {
    kv->append();
    buf = mymalloc(1,dummy,memtag);

    for (int iproc = me+numprocs; iproc < nprocs; iproc += numprocs) {
      MPI_Send(&flag,0,MPI_INT,iproc,0,comm);
      MPI_Recv(&npage_kv,1,MPI_INT,iproc,0,comm,&status);
      
      for (int ipage = 0; ipage < npage_kv; ipage++) {
	MPI_Irecv(buf,pagesize,MPI_BYTE,iproc,1,comm,&request);
	MPI_Send(&flag,0,MPI_INT,iproc,0,comm);
	MPI_Recv(sizes,4,MPI_UNSIGNED_LONG,iproc,0,comm,&status);
	crsize += sizes[3];
	MPI_Wait(&request,&status);
	kv->add(sizes[0],buf,sizes[1],sizes[2],sizes[3]);
      }
    }

    myfree(memtag);

  } else {
    int iproc = me % numprocs;
    npage_kv = kv->request_info(&buf);

    MPI_Recv(&flag,0,MPI_INT,iproc,0,comm,&status);
    MPI_Send(&npage_kv,1,MPI_INT,iproc,0,comm);

    for (int ipage = 0; ipage < npage_kv; ipage++) {
      sizes[0] = kv->request_page(ipage,sizes[1],sizes[2],sizes[3]);
      MPI_Recv(&flag,0,MPI_INT,iproc,0,comm,&status);
      MPI_Send(sizes,4,MPI_UNSIGNED_LONG,iproc,0,comm);
      MPI_Send(buf,sizes[3],MPI_BYTE,iproc,1,comm);
      cssize += sizes[3];
    }

    // leave empty KV on vacated procs

    myfree(kv->memtag);
    delete kv;
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  }

  commtime += MPI_Wtime() - timestart;
  kv->complete();

  stats("Gather",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   create a KV via a parallel map operation for nmap tasks
   make one call to appmap() for each task
   mapstyle determines how tasks are partitioned to processors
------------------------------------------------------------------------- */

uint64_t MapReduce::map(int nmap, void (*appmap)(int, KeyValue *, void *),
			void *appptr, int addflag)
{
  MPI_Status status;

  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (!allocated) allocate();
  if (kmv) myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  if (addflag == 0) {
    if (kv) myfree(kv->memtag);
    delete kv;
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else if (kv == NULL) {
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else {
    kv->append();
  }

  // nprocs = 1 = all tasks to single processor
  // mapstyle 0 = chunk of tasks to each proc
  // mapstyle 1 = strided tasks to each proc
  // mapstyle 2 = master/slave assignment of tasks

  if (nprocs == 1) {
    for (int itask = 0; itask < nmap; itask++)
      appmap(itask,kv,appptr);

  } else if (mapstyle == 0) {
    uint64_t nmap64 = nmap;
    int lo = me * nmap64 / nprocs;
    int hi = (me+1) * nmap64 / nprocs;
    for (int itask = lo; itask < hi; itask++)
      appmap(itask,kv,appptr);

  } else if (mapstyle == 1) {
    for (int itask = me; itask < nmap; itask += nprocs)
      appmap(itask,kv,appptr);

  } else if (mapstyle == 2) {
    if (me == 0) {
      int doneflag = -1;
      int ndone = 0;
      int itask = 0;
      for (int iproc = 1; iproc < nprocs; iproc++) {
	if (itask < nmap) {
	  MPI_Send(&itask,1,MPI_INT,iproc,0,comm);
	  itask++;
	} else {
	  MPI_Send(&doneflag,1,MPI_INT,iproc,0,comm);
	  ndone++;
	}
      }
      while (ndone < nprocs-1) {
	int iproc,tmp;
	MPI_Recv(&tmp,1,MPI_INT,MPI_ANY_SOURCE,0,comm,&status);
	iproc = status.MPI_SOURCE;

	if (itask < nmap) {
	  MPI_Send(&itask,1,MPI_INT,iproc,0,comm);
	  itask++;
	} else {
	  MPI_Send(&doneflag,1,MPI_INT,iproc,0,comm);
	  ndone++;
	}
      }

    } else {
      while (1) {
	int itask;
	MPI_Recv(&itask,1,MPI_INT,0,0,comm,&status);
	if (itask < 0) break;
	appmap(itask,kv,appptr);
	MPI_Send(&itask,1,MPI_INT,0,0,comm);
      }
    }

  } else error->all("Invalid mapstyle setting");

  kv->complete();

  stats("Map",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   create a KV via a parallel map operation for list of files in file
   make one call to appmap() for each file in file
   mapstyle determines how tasks are partitioned to processors
------------------------------------------------------------------------- */

uint64_t MapReduce::map(char *file, 
			void (*appmap)(int, char *, KeyValue *, void *),
			void *appptr, int addflag)
{
  int n;
  char line[MAXLINE];
  MPI_Status status;

  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (!allocated) allocate();
  if (kmv) myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  if (addflag == 0) {
    if (kv) myfree(kv->memtag);
    delete kv;
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else if (kv == NULL) {
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else {
    kv->append();
  }

  // open file and extract filenames
  // bcast each filename to all procs
  // trim whitespace from beginning and end of filename

  int nmap = 0;
  int maxfiles = 0;
  char **files = NULL;
  FILE *fp;

  if (me == 0) {
    fp = fopen(file,"r");
    if (fp == NULL) error->one("Could not open file of file names");
  }

  while (1) {
    if (me == 0) {
      if (fgets(line,MAXLINE,fp) == NULL) n = 0;
      else n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,comm);
    if (n == 0) {
      if (me == 0) fclose(fp);
      break;
    }

    MPI_Bcast(line,n,MPI_CHAR,0,comm);

    char *ptr = line;
    while (isspace(*ptr)) ptr++;
    if (strlen(ptr) == 0) error->all("Blank line in file of file names");
    char *ptr2 = ptr + strlen(ptr) - 1;
    while (isspace(*ptr2)) ptr2--;
    ptr2++;
    *ptr2 = '\0';

    if (nmap == maxfiles) {
      maxfiles += FILECHUNK;
      files = (char **)
	memory->srealloc(files,maxfiles*sizeof(char *),"MR:files");
    }
    n = strlen(ptr) + 1;
    files[nmap] = new char[n];
    strcpy(files[nmap],ptr);
    nmap++;
  }
  
  // nprocs = 1 = all tasks to single processor
  // mapstyle 0 = chunk of tasks to each proc
  // mapstyle 1 = strided tasks to each proc
  // mapstyle 2 = master/slave assignment of tasks

  if (nprocs == 1) {
    for (int itask = 0; itask < nmap; itask++)
      appmap(itask,files[itask],kv,appptr);

  } else if (mapstyle == 0) {
    uint64_t nmap64 = nmap;
    int lo = me * nmap64 / nprocs;
    int hi = (me+1) * nmap64 / nprocs;
    for (int itask = lo; itask < hi; itask++)
      appmap(itask,files[itask],kv,appptr);

  } else if (mapstyle == 1) {
    for (int itask = me; itask < nmap; itask += nprocs)
      appmap(itask,files[itask],kv,appptr);

  } else if (mapstyle == 2) {
    if (me == 0) {
      int doneflag = -1;
      int ndone = 0;
      int itask = 0;
      for (int iproc = 1; iproc < nprocs; iproc++) {
	if (itask < nmap) {
	  MPI_Send(&itask,1,MPI_INT,iproc,0,comm);
	  itask++;
	} else {
	  MPI_Send(&doneflag,1,MPI_INT,iproc,0,comm);
	  ndone++;
	}
      }
      while (ndone < nprocs-1) {
	int iproc,tmp;
	MPI_Recv(&tmp,1,MPI_INT,MPI_ANY_SOURCE,0,comm,&status);
	iproc = status.MPI_SOURCE;

	if (itask < nmap) {
	  MPI_Send(&itask,1,MPI_INT,iproc,0,comm);
	  itask++;
	} else {
	  MPI_Send(&doneflag,1,MPI_INT,iproc,0,comm);
	  ndone++;
	}
      }

    } else {
      while (1) {
	int itask;
	MPI_Recv(&itask,1,MPI_INT,0,0,comm,&status);
	if (itask < 0) break;
	appmap(itask,files[itask],kv,appptr);
	MPI_Send(&itask,1,MPI_INT,0,0,comm);
      }
    }

  } else error->all("Invalid mapstyle setting");

  // clean up file list

  for (int i = 0; i < nmap; i++) delete [] files[i];
  memory->sfree(files);

  kv->complete();

  stats("Map",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   create a KV via a parallel map operation for nmap tasks
   nfiles filenames are split into nmap pieces based on separator char
------------------------------------------------------------------------- */

uint64_t MapReduce::map(int nmap, int nfiles, char **files,
			char sepchar, int delta,
			void (*appmap)(int, char *, int, KeyValue *, void *),
			void *appptr, int addflag)
{
  filemap.sepwhich = 1;
  filemap.sepchar = sepchar;
  filemap.delta = delta;

  return map_file(nmap,nfiles,files,appmap,appptr,addflag);
}

/* ----------------------------------------------------------------------
   create a KV via a parallel map operation for nmap tasks
   nfiles filenames are split into nmap pieces based on separator string
------------------------------------------------------------------------- */

uint64_t MapReduce::map(int nmap, int nfiles, char **files,
			char *sepstr, int delta,
			void (*appmap)(int, char *, int, KeyValue *, void *),
			void *appptr, int addflag)
{
  filemap.sepwhich = 0;
  int n = strlen(sepstr) + 1;
  filemap.sepstr = new char[n];
  strcpy(filemap.sepstr,sepstr);
  filemap.delta = delta;

  return map_file(nmap,nfiles,files,appmap,appptr,addflag);
}

/* ----------------------------------------------------------------------
   called by 2 map methods that take files and a separator
   create a KV via a parallel map operation for nmap tasks
   nfiles filenames are split into nmap pieces based on separator
   FileMap struct stores info on how to split files
   calls non-file map() to partition tasks to processors
     with callback to non-class map_file_standalone()
   map_file_standalone() reads chunk of file and passes it to user appmap()
------------------------------------------------------------------------- */

uint64_t MapReduce::map_file(int nmap, int nfiles, char **files,
			     void (*appmap)(int, char *, int, KeyValue *, void *),
			     void *appptr, int addflag)
{
  if (nfiles > nmap) error->all("Cannot map with more files than tasks");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (!allocated) allocate();
  if (kmv) myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  // copy filenames into FileMap

  filemap.filename = new char*[nfiles];
  for (int i = 0; i < nfiles; i++) {
    int n = strlen(files[i]) + 1;
    filemap.filename[i] = new char[n];
    strcpy(filemap.filename[i],files[i]);
  }

  // get filesize of each file via stat()
  // proc 0 queries files, bcasts results to all procs

  filemap.filesize = new uint64_t[nfiles];
  struct stat stbuf;

  if (me == 0) {
    for (int i = 0; i < nfiles; i++) {
      int flag = stat(files[i],&stbuf);
      if (flag < 0) error->one("Could not query file size");
      filemap.filesize[i] = stbuf.st_size;
    }
  }

  MPI_Bcast(filemap.filesize,nfiles*sizeof(uint64_t),MPI_BYTE,0,comm);

  // ntotal = total size of all files
  // nideal = ideal # of bytes per task

  uint64_t ntotal = 0;
  for (int i = 0; i < nfiles; i++) ntotal += filemap.filesize[i];
  uint64_t nideal = MAX(1,ntotal/nmap);

  // tasksperfile[i] = # of tasks for Ith file
  // initial assignment based on ideal chunk size
  // increment/decrement tasksperfile until reach target # of tasks
  // even small files must have 1 task

  filemap.tasksperfile = new int[nfiles];

  int ntasks = 0;
  for (int i = 0; i < nfiles; i++) {
    filemap.tasksperfile[i] = MAX(1,filemap.filesize[i]/nideal);
    ntasks += filemap.tasksperfile[i];
  }

  while (ntasks < nmap)
    for (int i = 0; i < nfiles; i++)
      if (filemap.filesize[i] > nideal) {
	filemap.tasksperfile[i]++;
	ntasks++;
	if (ntasks == nmap) break;
      }
  while (ntasks > nmap)
    for (int i = 0; i < nfiles; i++)
      if (filemap.tasksperfile[i] > 1) {
	filemap.tasksperfile[i]--;
	ntasks--;
	if (ntasks == nmap) break;
      }

  // check if any tasks are so small they will cause overlapping reads w/ delta
  // if so, reduce number of tasks for that file and issue warning

  int flag = 0;
  for (int i = 0; i < nfiles; i++) {
    if (filemap.filesize[i] / filemap.tasksperfile[i] > filemap.delta)
      continue;
    flag = 1;
    while (filemap.tasksperfile[i] > 1) {
      filemap.tasksperfile[i]--;
      nmap--;
      if (filemap.filesize[i] / filemap.tasksperfile[i] > filemap.delta) break;
    }
  }

  if (flag & me == 0) {
    char str[128];
    sprintf(str,"File(s) too small for file delta - decreased map tasks to %d",
	    nmap);
    error->warning(str);
  }

  // whichfile[i] = which file is associated with the Ith task
  // whichtask[i] = which task in that file the Ith task is

  filemap.whichfile = new int[nmap];
  filemap.whichtask = new int[nmap];

  int itask = 0;
  for (int i = 0; i < nfiles; i++)
    for (int j = 0; j < filemap.tasksperfile[i]; j++) {
      filemap.whichfile[itask] = i;
      filemap.whichtask[itask++] = j;
    }

  // use non-file map() to partition tasks to procs
  // it calls map_file_standalone once for each task

  int verbosity_hold = verbosity;
  int timer_hold = timer;
  verbosity = timer = 0;

  filemap.appmapfile = appmap;
  filemap.appptr = appptr;
  map(nmap,&map_file_standalone,this,addflag);

  verbosity = verbosity_hold;
  timer = timer_hold;
  stats("Map",0);

  // destroy FileMap

  if (filemap.sepwhich == 0) delete [] filemap.sepstr;
  for (int i = 0; i < nfiles; i++) delete [] filemap.filename[i];
  delete [] filemap.filename;
  delete [] filemap.filesize;
  delete [] filemap.tasksperfile;
  delete [] filemap.whichfile;
  delete [] filemap.whichtask;

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   wrappers on user-provided appmapfile function
   2-level wrapper needed b/c file map() calls non-file map()
     and cannot pass it a class method unless it were static,
     but then it couldn't access MR class data
   so non-file map() is passed standalone non-class method
   standalone calls back into class wrapper which calls user appmapfile()
------------------------------------------------------------------------- */

void map_file_standalone(int imap, KeyValue *kv, void *ptr)
{
  MapReduce *mr = (MapReduce *) ptr;
  mr->map_file_wrapper(imap,kv);
}

void MapReduce::map_file_wrapper(int imap, KeyValue *kv)
{
  // readstart = position in file to start reading for this task
  // readsize = # of bytes to read including delta

  uint64_t filesize = filemap.filesize[filemap.whichfile[imap]];
  int itask = filemap.whichtask[imap];
  int ntask = filemap.tasksperfile[filemap.whichfile[imap]];

  uint64_t readstart = itask*filesize/ntask;
  uint64_t readnext = (itask+1)*filesize/ntask;
  if (readnext - readstart + filemap.delta + 1 > INTMAX)
    error->one("Single file read exceeds int size");
  int readsize = readnext - readstart + filemap.delta;
  readsize = MIN(readsize,filesize-readstart);

  // read from appropriate file
  // terminate string with NULL

  char *str = (char *) memory->smalloc(readsize+1,"MR:fileread");
  FILE *fp = fopen(filemap.filename[filemap.whichfile[imap]],"rb");
  fseek(fp,readstart,SEEK_SET);
  fread(str,1,readsize,fp);
  str[readsize] = '\0';
  fclose(fp);

  // if not first task in file, trim start of string
  // separator can be single char or a string
  // str[strstart] = 1st char in string
  // if separator = char, strstart is char after separator
  // if separator = string, strstart is 1st char of separator

  int strstart = 0;
  if (itask > 0) {
    char *ptr;
    if (filemap.sepwhich) ptr = strchr(str,filemap.sepchar);
    else ptr = strstr(str,filemap.sepstr);
    if (ptr == NULL || ptr-str > filemap.delta)
      error->one("Could not find file separator within delta");
    strstart = ptr-str + filemap.sepwhich;
  }

  // if not last task in file, trim end of string
  // separator can be single char or a string
  // str[strstop] = last char in string = inserted NULL
  // if separator = char, NULL is char after separator
  // if separator = string, NULL is 1st char of separator

  int strstop = readsize;
  if (itask < ntask-1) {
    char *ptr;
    if (filemap.sepwhich) 
      ptr = strchr(&str[readnext-readstart],filemap.sepchar);
    else 
      ptr = strstr(&str[readnext-readstart],filemap.sepstr);
    if (ptr == NULL) error->one("Could not find file separator within delta");
    if (filemap.sepwhich) ptr++;
    *ptr = '\0';
    strstop = ptr-str;
  }

  // call user appmapfile() function with user data ptr

  int strsize = strstop - strstart + 1;
  filemap.appmapfile(imap,&str[strstart],strsize,kv,filemap.appptr);
  memory->sfree(str);
}

/* ----------------------------------------------------------------------
   create a KV via a parallel map operation from an existing MR's KV
   make one call to appmap() for each key/value pair in the input MR's KV
   each proc operates on key/value pairs it owns
------------------------------------------------------------------------- */

uint64_t MapReduce::map(MapReduce *mr, 
			void (*appmap)(uint64_t, char *, int, char *, int, 
				       KeyValue *, void *),
			void *appptr, int addflag)
{
  if (mr->kv == NULL)
    error->all("MapReduce passed to map() does not have KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  if (!allocated) allocate();
  if (kmv) myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  // kv_src = KeyValue object which sends KV pairs to appmap()
  // kv_dest = KeyValue object which stores new KV pairs
  // if mr = this and addflag, then 2 KVs are the same, copy KV first

  KeyValue *kv_src = mr->kv;
  KeyValue *kv_dest;

  if (mr == this) {
    if (addflag) {
      kv_dest = new KeyValue(this,kalign,valign,memory,error,comm);
      kv_dest->set_page();
      kv_dest->copy(kv_src);
      kv_dest->append();
    } else {
      kv_dest = new KeyValue(this,kalign,valign,memory,error,comm);
      kv_dest->set_page();
    }
  } else {
    if (addflag == 0) {
      if (kv) myfree(kv->memtag);
      delete kv;
      kv_dest = new KeyValue(this,kalign,valign,memory,error,comm);
      kv_dest->set_page();
    } else if (kv == NULL) {
      kv_dest = new KeyValue(this,kalign,valign,memory,error,comm);
      kv_dest->set_page();
    } else {
      kv->append();
      kv_dest = kv;
    }
  }

  int nkey_kv,keybytes,valuebytes;
  uint64_t dummy1,dummy2,dummy3;
  char *page_kv,*ptr,*key,*value;
  int npage_kv = kv_src->request_info(&page_kv);
  uint64_t n = 0;

  for (int ipage = 0; ipage < npage_kv; ipage++) {
    nkey_kv = kv_src->request_page(ipage,dummy1,dummy2,dummy3);
    ptr = page_kv;

    for (int i = 0; i < nkey_kv; i++) {
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
      
      appmap(n++,key,keybytes,value,valuebytes,kv_dest,appptr);
    }
  }

  if (mr == this) {
    myfree(kv_src->memtag);
    delete kv_src;
  }
  kv = kv_dest;
  kv->complete();

  stats("Map",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   open a KV so KV pairs can be added to it by another MR's map
------------------------------------------------------------------------- */

void MapReduce::open(int addflag)
{
  if (!allocated) allocate();
  if (kmv) myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  if (addflag == 0) {
    if (kv) myfree(kv->memtag);
    delete kv;
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else if (kv == NULL) {
    kv = new KeyValue(this,kalign,valign,memory,error,comm);
    kv->set_page();
  } else {
    kv->append();
  }
}

/* ----------------------------------------------------------------------
   debug print of KV or KMV pairs
   if all procs are printing, pass print token from proc to proc
------------------------------------------------------------------------- */

void MapReduce::print(int proc, int nstride, int kflag, int vflag)
{
  MPI_Status status;

  if (kv == NULL && kmv == NULL)
    error->all("Cannot print without KeyValue or KeyMultiValue");
  if (kflag < 0 || vflag < 0) error->all("Invalid print args");
  if (kflag > 7 || vflag > 7) error->all("Invalid print args");

  if (proc == me) {
    if (kv) kv->print(nstride,kflag,vflag);
    if (kmv) kmv->print(nstride,kflag,vflag);
  }

  if (proc >= 0) return;

  int token;
  MPI_Barrier(comm);
  if (me > 0) MPI_Recv(&token,0,MPI_INT,me-1,0,comm,&status);
  if (kv) kv->print(nstride,kflag,vflag);
  if (kmv) kmv->print(nstride,kflag,vflag);
  if (me < nprocs-1) MPI_Send(&token,0,MPI_INT,me+1,0,comm);
  MPI_Barrier(comm);
}

/* ----------------------------------------------------------------------
   create a KV from a KMV via a parallel reduce operation for nmap tasks
   make one call to appreduce() for each KMV pair
   each proc processes its owned KMV pairs
------------------------------------------------------------------------- */

uint64_t MapReduce::reduce(void (*appreduce)(char *, int, char *, int,
					     int *, KeyValue *, void *),
			   void *appptr)
{
  if (kmv == NULL) error->all("Cannot reduce without KeyMultiValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  kv = new KeyValue(this,kalign,valign,memory,error,comm);
  kv->set_page();

  uint64_t dummy;
  int memtag;
  char *mvpage = mymalloc(1,dummy,memtag);

  int nkey_kmv,nvalues,keybytes,mvaluebytes;
  uint64_t dummy1,dummy2,dummy3;
  int *valuesizes;
  char *ptr,*key,*multivalue;

  char *page_kmv;
  int npage_kmv = kmv->request_info(&page_kmv);
  char *page_hold = page_kmv;

  for (int ipage = 0; ipage < npage_kmv; ipage++) {
    nkey_kmv = kmv->request_page(ipage,0,dummy1,dummy2,dummy3);
    ptr = page_kmv;

    for (int i = 0; i < nkey_kmv; i++) {
      nvalues = *((int *) ptr);
      ptr += sizeof(int);

      if (nvalues > 0) {
	keybytes = *((int *) ptr);
	ptr += sizeof(int);
	mvaluebytes = *((int *) ptr);
	ptr += sizeof(int);
	valuesizes = (int *) ptr;
	ptr += ((uint64_t) nvalues) * sizeof(int);
	
	ptr = ROUNDUP(ptr,kalignm1);
	key = ptr;
	ptr += keybytes;
	ptr = ROUNDUP(ptr,valignm1);
	multivalue = ptr;
	ptr += mvaluebytes;
	ptr = ROUNDUP(ptr,talignm1);
	
	appreduce(key,keybytes,multivalue,nvalues,valuesizes,kv,appptr);

      } else {
	keybytes = *((int *) ptr);
	ptr += sizeof(int);
	ptr = ROUNDUP(ptr,kalignm1);
	key = ptr;

	// set KMV page to mvpage so key will not be overwritten
	// when multivalue_block() loads new pages of values

	kmv->page = mvpage;
	kmv_block_valid = 1;
	kmv_key_page = ipage;
	kmv_nvalue_total = kmv->multivalue_blocks(ipage,kmv_nblock);
	appreduce(key,keybytes,NULL,0,(int *) this,kv,appptr);
	kmv_block_valid = 0;
	ipage += kmv_nblock;
	kmv->page = page_hold;
      }
    }
  }

  kv->complete();
  myfree(memtag);

  // delete KMV
  // close is necessary b/c KMV files do not close themselves
  // since users may use request_page() via multivalue_block()

  kmv->close_file();
  myfree(kmv->memtag);
  delete kmv;
  kmv = NULL;

  stats("Reduce",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   scrunch KV to create a KMV on fewer processors, each with a single pair
   gather followed by a collapse
   numprocs = # of procs new KMV resides on (0 to numprocs-1)
   new key = provided key name (same on every proc)
   new value = list of old key,value,key,value,etc
------------------------------------------------------------------------- */

uint64_t MapReduce::scrunch(int numprocs, char *key, int keybytes)
{
  if (kv == NULL) error->all("Cannot scrunch without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  int verbosity_hold = verbosity;
  int timer_hold = timer;
  verbosity = timer = 0;

  gather(numprocs);
  collapse(key,keybytes);

  verbosity = verbosity_hold;
  timer = timer_hold;
  stats("Scrunch",1);

  uint64_t nkeyall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   query total # of values and # of value blocks in a single multi-page KMV
   called from user myreduce() or mycompress() function
------------------------------------------------------------------------- */

uint64_t MapReduce::multivalue_blocks(int &nblock)
{
  if (!kmv_block_valid) error->one("Invalid call to multivalue_blocks()");
  nblock = kmv_nblock;
  return kmv_nvalue_total;
}

/* ----------------------------------------------------------------------
   query info for 1 block of a single KMV that spans multiple pages
   called from user myreduce() or mycompress() function
   iblock = 0 to nblock_kmv-1
------------------------------------------------------------------------- */

int MapReduce::multivalue_block(int iblock, 
				char **pmultivalue, int **pvaluesizes)
{
  if (!kmv_block_valid) error->one("Invalid call to multivalue_block()");
  if (iblock < 0 || iblock >= kmv_nblock)
    error->one("Invalid page request to multivalue_block()");

  uint64_t dummy1,dummy2,dummy3;
  char *page_kmv;
  kmv->request_info(&page_kmv);
  kmv->request_page(kmv_key_page+iblock+1,0,dummy1,dummy2,dummy3);

  char *ptr = page_kmv;
  int nvalues = *((int *) ptr);
  ptr += sizeof(int);
  *pvaluesizes = (int *) ptr;
  ptr += nvalues*sizeof(int);
  *pmultivalue = ROUNDUP(ptr,valignm1);

  return nvalues;
}

/* ----------------------------------------------------------------------
   sort keys in a KV to create a new KV
   use appcompare() to compare 2 keys
   each proc sorts only its data
------------------------------------------------------------------------- */

uint64_t MapReduce::sort_keys(int (*appcompare)(char *, int, char *, int))
{
  if (kv == NULL) error->all("Cannot sort_keys without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  compare = appcompare;
  sort_kv(0);

  stats("Sort_keys",0);
  fcounter_sort = 0;

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   sort values in a KV to create a new KV
   use appcompare() to compare 2 values
   each proc sorts only its data
------------------------------------------------------------------------- */

uint64_t MapReduce::sort_values(int (*appcompare)(char *, int, char *, int))
{
  if (kv == NULL) error->all("Cannot sort_values without KeyValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  compare = appcompare;
  sort_kv(1);

  stats("Sort_values",0);
  fcounter_sort = 0;

  uint64_t nkeyall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   sort values within each multivalue in a KMV
   sorts in place, does not create a new KMV
   use appcompare() to compare 2 values within a multivalue
   each proc sorts only its data
------------------------------------------------------------------------- */

uint64_t MapReduce::sort_multivalues(int (*appcompare)(char *, int, 
						       char *, int))
{
  int i,j,k;
  int *order;
  uint64_t offset;

  if (kmv == NULL) error->all("Cannot sort_multivalues without KeyMultiValue");
  if (timer) start_timer();
  if (verbosity) file_stats(0);

  char *page_kmv;
  int npage_kmv = kmv->request_info(&page_kmv);

  compare = appcompare;
  mrptr = this;

  uint64_t dummy;
  int memtag1,memtag2;
  char *scratch = mymalloc(1,dummy,memtag1);
  char *twopage = mymalloc(2,dummy,memtag2);

  int nkey_kmv,nvalues,keybytes,mvaluebytes;
  uint64_t dummy1,dummy2,dummy3;
  int *valuesizes;
  char *ptr,*multivalue,*ptr2;

  for (int ipage = 0; ipage < npage_kmv; ipage++) {
    nkey_kmv = kmv->request_page(ipage,1,dummy1,dummy2,dummy3);
    ptr = page_kmv;

    for (i = 0; i < nkey_kmv; i++) {
      nvalues = *((int *) ptr);
      ptr += sizeof(int);

      if (nvalues == 0)
	error->one("Sort_multivalue of multi-page KeyMultiValue "
		   "not yet supported");

      keybytes = *((int *) ptr);
      ptr += sizeof(int);
      mvaluebytes = *((int *) ptr);
      ptr += sizeof(int);
      valuesizes = (int *) ptr;
      ptr += ((uint64_t) nvalues) * sizeof(int);
      
      ptr = ROUNDUP(ptr,kalignm1);
      ptr += keybytes;
      ptr = ROUNDUP(ptr,valignm1);
      multivalue = ptr;
      ptr += mvaluebytes;
      ptr = ROUNDUP(ptr,talignm1);

      // setup 2 arrays from 2 pages of memory
      // order = ordering of values in multivalue, initially 0 to N-1
      // dptr = ptr to each value
      // slength = length of each value = valuesizes

      offset = ((uint64_t) nvalues) * sizeof(int);
      order = (int *) twopage;
      dptr = (char **) &twopage[offset];
      slength = valuesizes;

      ptr2 = multivalue;
      for (j = 0; j < nvalues; j++) {
	order[j] = j;
	dptr[j] = ptr2;
	ptr2 += valuesizes[j];
      }

      // sort values within multivalue via qsort()
      // simply creates new order array

      qsort(order,nvalues,sizeof(int),compare_standalone);
      
      // reorder the multivalue, using scratch space
      // copy back into original page

      ptr2 = scratch;
      for (j = 0; j < nvalues; j++) {
	k = order[j];
	memcpy(ptr2,&dptr[k],slength[k]);
	ptr2 += slength[k];
      }
      memcpy(multivalue,scratch,mvaluebytes);
    }

    // overwrite the changed KMV page

    kmv->overwrite_page(ipage);
  }

  // close is necessary b/c KMV files do not close themselves

  kmv->close_file();

  // free memory pages

  myfree(memtag1);
  myfree(memtag2);

  stats("Sort_multivalues",0);

  uint64_t nkeyall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  return nkeyall;
}

/* ----------------------------------------------------------------------
   sort keys or values in a KV to create a new KV
   flag = 0 = sort keys, flag = 1 = sort values
------------------------------------------------------------------------- */

void MapReduce::sort_kv(int flag)
{
   int nkey_kv,memtag,memtag1,memtag2,memtag_twopage,src1,src2;
   uint64_t dummy,dummy1,dummy2,dummy3;
   char *page_kv;
   void *src1ptr,*src2ptr,*destptr;

   mrptr = this;
   int npage_kv = kv->request_info(&page_kv);
   memtag = kv->memtag;

   // KV has single page
   // sort into newpage, assign newpage to KV, and return

   if (npage_kv == 1) {
     char *twopage = mymalloc(2,dummy,memtag_twopage);
     char *newpage = mymalloc(1,dummy,memtag1);
     nkey_kv = kv->request_page(0,dummy1,dummy2,dummy3);
     sort_onepage(flag,nkey_kv,page_kv,newpage,twopage);
     myfree(memtag_twopage);
     myfree(memtag);
     kv->set_page(pagesize,newpage,memtag1);
     return;
   }

   // KV has multiple pages
   // perform merge sort, two sources at a time into destination
   // each pass thru while loop increments I by 2 and N by 1
   // sources can be sorted page or Spool file
   // destination can be Spool file or final sorted KV

   char *twopage = mymalloc(2,dummy,memtag_twopage);
   char *page1 = mymalloc(1,dummy,memtag1);
   char *page2 = mymalloc(1,dummy,memtag2);

   Spool **spools = new Spool*[2*npage_kv];
   int n = npage_kv;
   int i = 0;

   while (i < n) {
     if (i < npage_kv) {
       kv->set_page(pagesize,page_kv,memtag);
       nkey_kv = kv->request_page(i++,dummy1,dummy2,dummy3);
       sort_onepage(flag,nkey_kv,page_kv,page1,twopage);
       src1ptr = (void *) page1;
       src1 = nkey_kv;
     } else {
       spools[i]->set_page(pagesize,page1);
       src1ptr = (void *) spools[i++];
       src1 = 0;
     }
     if (i < npage_kv) {
       kv->set_page(pagesize,page_kv,memtag);
       nkey_kv = kv->request_page(i++,dummy1,dummy2,dummy3);
       sort_onepage(flag,nkey_kv,page_kv,page2,twopage);
       src2ptr = (void *) page2;
       src2 = nkey_kv;
     } else {
       spools[i]->set_page(pagesize,page2);
       src2ptr = (void *) spools[i++];
       src2 = 0;
     }

     if (i < n) {
       spools[n] = new Spool(SORTFILE,this,memory,error);
       spools[n]->set_page(pagesize,page_kv);
       destptr = (void *) spools[n];
       merge(flag,src1,src1ptr,src2,src2ptr,0,destptr);
       if (!src1) delete spools[i-2];
       if (!src2) delete spools[i-1];
       spools[n++]->complete();
     } else {
       delete kv;
       kv = new KeyValue(this,kalign,valign,memory,error,comm);
       kv->set_page(pagesize,page_kv,memtag);
       destptr = (void *) kv;
       merge(flag,src1,src1ptr,src2,src2ptr,1,destptr);
       if (!src1) delete spools[i-2];
       if (!src2) delete spools[i-1];
       kv->complete();
     }
   }

   delete [] spools;

   myfree(memtag_twopage);
   myfree(memtag1);
   myfree(memtag2);
}

/* ----------------------------------------------------------------------
   sort keys or values in one page of a KV to create a new KV
   flag = 0 for sort keys, flag = 1 for sort values
   unsorted KVs are in pagesrc, final sorted KVs are put in pagedest
   twopage is used for qsort() data structs
------------------------------------------------------------------------- */

void MapReduce::sort_onepage(int flag, int nkey_kv,
			     char *pagesrc, char *pagedest, char *twopage)
{
  int i,j;
  int keybytes,valuebytes;
  char *ptr,*key,*value;

  // setup 3 arrays from twopage of memory
  // order = ordering of keys or values in KV, initially 0 to N-1
  // slength = length of each key or value
  // dptr = datum ptr = ptr to each key or value
  
  uint64_t offset = ((uint64_t) nkey_kv) * sizeof(int);
  int *order = (int *) twopage;
  slength = (int *) &twopage[offset];
  dptr = (char **) &twopage[2*offset];

  ptr = pagesrc;

  for (i = 0; i < nkey_kv; i++) {
    order[i] = i;
    
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
    
    if (flag == 0) {
      slength[i] = keybytes;
      dptr[i] = key;
    } else {
      slength[i] = valuebytes;
      dptr[i] = value;
    }
  }
  
  // sort keys or values via qsort()
  // simply creates new order array
  
  qsort(order,nkey_kv,sizeof(int),compare_standalone);
  
  // dptr = start of each KV pair
  // slength = length of entire KV pair
  
  ptr = pagesrc;
  
  for (i = 0; i < nkey_kv; i++) {
    dptr[i] = ptr;
    keybytes = *((int *) ptr);
    valuebytes = *((int *) (ptr+sizeof(int)));;
    
    ptr += twolenbytes;
    ptr = ROUNDUP(ptr,kalignm1);
    ptr += keybytes;
    ptr = ROUNDUP(ptr,valignm1);
    ptr += valuebytes;
    ptr = ROUNDUP(ptr,talignm1);
    
    slength[i] = ptr - dptr[i];
  }
  
  // reorder KV pairs into dest page
  
  ptr = pagedest;
  for (i = 0; i < nkey_kv; i++) {
    j = order[i];
    memcpy(ptr,dptr[j],slength[j]);
    ptr += slength[j];
  }
}

/* ----------------------------------------------------------------------
   merge of 2 sources into a destination
   flag = 0 for key sort, flag = 1 for value sort
   src1,src2 can each be Spool file (src = 0) or KV page (src = nkey_kv)
   dest can be Spool file (dest = 0)  or final KV (dest = 1)
------------------------------------------------------------------------- */

void MapReduce::merge(int flag, int src1, void *src1ptr,
		      int src2, void *src2ptr, int dest, void *destptr)
{
  int result,ientry1,ientry2,nbytes1,nbytes2,ipage1,ipage2;
  int npage1,npage2,nentry1,nentry2;
  char *str1,*str2,*page1,*page2;
  Spool *sp1,*sp2,*spdest;
  KeyValue *kvdest;

  if (src1) {
    npage1 = 1;
    page1 = (char *) src1ptr;
    nentry1 = src1;
  } else {
    sp1 = (Spool *) src1ptr;
    npage1 = sp1->request_info(&page1);
    nentry1 = sp1->request_page(0);
  }
  if (src2) {
    npage2 = 1;
    page2 = (char *) src2ptr;
    nentry2 = src2;
  } else {
    sp2 = (Spool *) src2ptr;
    npage2 = sp2->request_info(&page2);
    nentry2 = sp2->request_page(0);
  }

  if (dest) kvdest = (KeyValue *) destptr;
  else spdest = (Spool *) destptr;

  ipage1 = ipage2 = 0;
  ientry1 = ientry2 = 0;

  char *ptr1 = page1;
  char *ptr2 = page2;
  int len1 = extract(flag,ptr1,str1,nbytes1);
  int len2 = extract(flag,ptr2,str2,nbytes2);

  int done = 0;

  while (1) {
    if (done == 0) result = compare(str1,nbytes1,str2,nbytes2);

    if (result <= 0) {
      if (dest) kvdest->add(ptr1);
      else spdest->add(len1,ptr1);
      ptr1 += len1;
      ientry1++;

      if (ientry1 == nentry1) {
	ipage1++;
	if (ipage1 < npage1) {
	  nentry1 = sp1->request_page(ipage1);
	  ientry1 = 0;
	  ptr1 = page1;
	  len1 = extract(flag,ptr1,str1,nbytes1);
	} else {
	  done++;
	  if (done == 2) break;
	  result = 1;
	}
      } else len1 = extract(flag,ptr1,str1,nbytes1);
    }

    if (result >= 0) {
      if (dest) kvdest->add(ptr2);
      else spdest->add(len2,ptr2);
      ptr2 += len2;
      ientry2++;

      if (ientry2 == nentry2) {
	ipage2++;
	if (ipage2 < npage2) {
	  nentry2 = sp2->request_page(ipage2);
	  ientry2 = 0;
	  ptr2 = page2;
	  len2 = extract(flag,ptr2,str2,nbytes2);
	} else {
	  done++;
	  if (done == 2) break;
	  result = -1;
	}
      } else len2 = extract(flag,ptr2,str2,nbytes2);
    }
  }
}

/* ----------------------------------------------------------------------
   extract datum from a KV pair beginning at ptr_start
   flag = 0, return key and keybytes as str and nbytes
   flag = 1, return value and valuebytes as str and nbytes
   also return byte increment to next entry
------------------------------------------------------------------------- */

int MapReduce::extract(int flag, char *ptr_start, char *&str, int &nbytes)
{
  char *ptr = ptr_start;
  int keybytes = *((int *) ptr);
  int valuebytes = *((int *) (ptr+sizeof(int)));;

  ptr += twolenbytes;
  ptr = ROUNDUP(ptr,kalignm1);
  char *key = ptr;
  ptr += keybytes;
  ptr = ROUNDUP(ptr,valignm1);
  char *value = ptr;
  ptr += valuebytes;
  ptr = ROUNDUP(ptr,talignm1);

  if (flag == 0) {
    str = key;
    nbytes = keybytes;
  } else {
    str = value;
    nbytes = valuebytes;
  }

  return ptr - ptr_start;
}

/* ----------------------------------------------------------------------
   wrappers on user-provided key or value comparison functions
   necessary so can extract 2 keys or values to pass back to application
   2-level wrapper needed b/c qsort() cannot be passed a class method
     unless it were static, but then it couldn't access MR class data
   so qsort() is passed standalone non-class method
   it accesses static class member mrptr, set before call to qsort()
   standalone calls back into class wrapper which calls user compare()
------------------------------------------------------------------------- */

int compare_standalone(const void *iptr, const void *jptr)
{
  return MapReduce::mrptr->compare_wrapper(*((int *) iptr),*((int *) jptr));
}

int MapReduce::compare_wrapper(int i, int j)
{
  return compare(dptr[i],slength[i],dptr[j],slength[j]);
}

/* ----------------------------------------------------------------------
   print stats for KV
------------------------------------------------------------------------- */

void MapReduce::kv_stats(int level)
{
  if (kv == NULL) error->all("Cannot print stats without KeyValue");

  double mbyte = 1024.0*1024.0;

  int npages;
  uint64_t nkeyall,ksizeall,vsizeall,esizeall;
  MPI_Allreduce(&kv->nkv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kv->ksize,&ksizeall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kv->vsize,&vsizeall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kv->esize,&esizeall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kv->npage,&npages,1,MPI_INT,MPI_SUM,comm);

  if (me == 0)
    printf("%lu pairs, %.3g Mb keys, %.3g Mb values, %.3g Mb, "
	   "%d pages\n",
	   nkeyall,ksizeall/mbyte,vsizeall/mbyte,esizeall/mbyte,npages);

  if (level == 2) {
    write_histo((double) kv->nkv,"  KV pairs:");
    write_histo(kv->ksize/mbyte,"  Kdata (Mb):");
    write_histo(kv->vsize/mbyte,"  Vdata (Mb):");
  }
}

/* ----------------------------------------------------------------------
   print stats for KMV
------------------------------------------------------------------------- */

void MapReduce::kmv_stats(int level)
{
  if (kmv == NULL) error->all("Cannot print stats without KeyMultiValue");

  double mbyte = 1024.0*1024.0;

  int npages;
  uint64_t nkeyall,ksizeall,vsizeall,esizeall;
  MPI_Allreduce(&kmv->nkmv,&nkeyall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kmv->ksize,&ksizeall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kmv->vsize,&vsizeall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kmv->esize,&esizeall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&kmv->npage,&npages,1,MPI_INT,MPI_SUM,comm);
  
  if (me == 0)
      printf("%lu pairs, %.3g Mb keys, %.3g Mb values, %.3g Mb, "
	     "%d pages\n",
	     nkeyall,ksizeall/mbyte,vsizeall/mbyte,esizeall/mbyte,npages);

  if (level == 2) {
    write_histo((double) kmv->nkmv,"  KMV pairs:");
    write_histo(kmv->ksize/mbyte,"  Kdata (Mb):");
    write_histo(kmv->vsize/mbyte,"  Vdata (Mb):");
  }
}

/* ----------------------------------------------------------------------
   print cummulative comm and file read/write stats
------------------------------------------------------------------------- */

void MapReduce::cummulative_stats(int level, int reset)
{
  double mbyte = 1024.0*1024.0;

  // communication

  uint64_t csize[2] = {cssize,crsize};
  uint64_t allcsize[2];
  MPI_Allreduce(csize,allcsize,2,MPI_UNSIGNED_LONG,MPI_SUM,comm);

  double ctime[1] = {commtime};
  double allctime[1];
  MPI_Allreduce(ctime,allctime,1,MPI_DOUBLE,MPI_SUM,comm);

  if (allcsize[0] || allcsize[1]) {
    if (me == 0) printf("Cummulative comm = "
			"%.3g Mb send, %.3g Mb recv, %.3g secs\n", 
			allcsize[0]/mbyte,allcsize[1]/mbyte,
			allctime[0]/nprocs);
    if (level == 2) {
      write_histo(csize[0]/mbyte,"  Send (Mb):");
      write_histo(csize[1]/mbyte,"  Recv (Mb):");
    }
  }

  // file I/O

  uint64_t size[2] = {rsize,wsize};
  uint64_t allsize[2];
  MPI_Allreduce(size,allsize,2,MPI_UNSIGNED_LONG,MPI_SUM,comm);

  if (allsize[0] || allsize[1]) {
    if (me == 0) printf("Cummulative I/O = %.3g Mb read, %.3g Mb write\n",
			allsize[0]/mbyte,allsize[1]/mbyte);

    if (level == 2) {
      write_histo(size[0]/mbyte,"  Read (Mb):");
      write_histo(size[1]/mbyte,"  Write (Mb):");
    }
  }

  if (reset) {
    rsize = wsize = 0;
    cssize = crsize = 0;
  }
}

/* ----------------------------------------------------------------------
   change fpath, but only if allocation has not occurred
------------------------------------------------------------------------- */

void MapReduce::set_fpath(const char *str)
{
  if (allocated) return;

  delete [] fpath;
  int n = strlen(str) + 1;
  fpath = new char[n];
  strcpy(fpath,str);
}

/* ----------------------------------------------------------------------
   print memory page and disk file stats for MR
------------------------------------------------------------------------- */

void MapReduce::mr_stats(int level)
{
  double mbyte = 1024.0*1024.0;

  int npages;
  MPI_Allreduce(&npage,&npages,1,MPI_INT,MPI_SUM,comm);
  uint64_t fsizemaxall;
  MPI_Allreduce(&fsizemax,&fsizemaxall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);

  if (me == 0)
    printf("MapReduce stats = %d pages, %.3g Mb mem, "
    	   "%.3g Mb hi-water for files\n",
    	   npages,npages*pagesize/mbyte,fsizemaxall/mbyte);
    
  if (level == 2) {
    if (npages) write_histo((double) npage,"  Pages:");
    if (fsizemaxall) write_histo(fsizemax/mbyte,"  HiWater:");
  }
}

/* ----------------------------------------------------------------------
   stats for one operation and its resulting KV or KMV
   which = 0 for KV, which = 1 for KMV
   output timer, KV/KMV, comm, I/O, or nothing depending on settings
------------------------------------------------------------------------- */

void MapReduce::stats(const char *heading, int which)
{
  if (timer) {
    if (timer == 1) {
      MPI_Barrier(comm);
      if (me == 0) printf("%s time (secs) = %g\n",
			  heading,MPI_Wtime()-time_start);
    } else if (timer == 2) {
      char str[64];
      sprintf(str,"%s time (secs) =",heading);
      write_histo(MPI_Wtime()-time_start,str);
    }
  }

  if (verbosity == 0) return;
  if (which == 0) {
    if (me == 0) printf("%s KV = ",heading);
    kv_stats(verbosity);
  } else {
    if (me == 0) printf("%s KMV = ",heading);
    kmv_stats(verbosity);
  }

  file_stats(1);

  uint64_t rall,sall,wall;
  double mbyte = 1024.0*1024.0;

  MPI_Allreduce(&cssize_one,&sall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&crsize_one,&rall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  if (sall || rall) {
    if (me == 0) printf("%s Comm = %.3g Mb send, %.3g Mb recv\n",heading,
			sall/mbyte,rall/mbyte);
    if (verbosity == 2) {
      write_histo(cssize_one/mbyte,"  Send (Mb):");
      write_histo(crsize_one/mbyte,"  Recv (Mb):");
    }
  }

  MPI_Allreduce(&rsize_one,&rall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  MPI_Allreduce(&wsize_one,&wall,1,MPI_UNSIGNED_LONG,MPI_SUM,comm);
  if (rall || wall) {
    if (me == 0) printf("%s I/O = %.3g Mb read, %.3g Mb write\n",heading,
			rall/mbyte,wall/mbyte);
    if (verbosity == 2) {
      write_histo(rsize_one/mbyte,"  Read (Mb):");
      write_histo(wsize_one/mbyte,"  Write (Mb):");
    }
  }

  int partall,setall,sortall;
  MPI_Allreduce(&fcounter_part,&partall,1,MPI_INT,MPI_SUM,comm);
  MPI_Allreduce(&fcounter_set,&setall,1,MPI_INT,MPI_SUM,comm);
  MPI_Allreduce(&fcounter_sort,&sortall,1,MPI_INT,MPI_SUM,comm);
  if (partall || setall || sortall) {
    if (me == 0) printf("%s Files = %d partition, %d set, %d sort\n",
			heading,partall,setall,sortall);
    if (verbosity == 2) {
      if (partall) write_histo((double) fcounter_part,"  Partfiles:");
      if (setall) write_histo((double) fcounter_set,"  Setfiles:");
      if (sortall) write_histo((double) fcounter_sort,"  Sortfiles:");
    }
  }
}

/* ----------------------------------------------------------------------
   create a filename with increasing counter for
     KV, KMV, Sort, Partition, Spool
   return filename, caller will delete it
------------------------------------------------------------------------- */

char *MapReduce::file_create(int style)
{
  int n = strlen(fpath) + 32;
  char *fname = new char[n];
  if (style == KVFILE)
    sprintf(fname,"%s/mrmpi.kv.%d.%d.%d",fpath,instance_me,fcounter_kv++,me);
  else if (style == KMVFILE)
    sprintf(fname,"%s/mrmpi.kmv.%d.%d.%d",fpath,instance_me,fcounter_kmv++,me);
  else if (style == SORTFILE)
    sprintf(fname,"%s/mrmpi.sort.%d.%d.%d",fpath,instance_me,
	    fcounter_sort++,me);
  else if (style == PARTFILE)
    sprintf(fname,"%s/mrmpi.part.%d.%d.%d",fpath,instance_me,
	    fcounter_part++,me);
  else if (style == SETFILE)
    sprintf(fname,"%s/mrmpi.set.%d.%d.%d",fpath,instance_me,
	    fcounter_set++,me);
  return fname;
}

/* ----------------------------------------------------------------------
   size of file read/writes from KV, KMV, and Spool files
   flag = 0 -> rsize/wsize = current size
   flag = 1 -> rsize/wsize = current size - previous size
------------------------------------------------------------------------- */

void MapReduce::file_stats(int flag)
{
  if (flag == 0) {
    rsize_one = rsize;
    wsize_one = wsize;
    cssize_one = cssize;
    crsize_one = crsize;
  } else {
    rsize_one = rsize - rsize_one;
    wsize_one = wsize - wsize_one;
    cssize_one = cssize - cssize_one;
    crsize_one = crsize - crsize_one;
  }
}

/* ---------------------------------------------------------------------- */

void MapReduce::start_timer()
{
  if (timer == 1) MPI_Barrier(comm);
  time_start = MPI_Wtime();
}

/* ----------------------------------------------------------------------
   round N up to multiple of nalign and return it
------------------------------------------------------------------------- */

uint64_t MapReduce::roundup(uint64_t n, int nalign)
{
  if (n % nalign == 0) return n;
  n = (n/nalign + 1) * nalign;
  return n;
}

/* ----------------------------------------------------------------------
   write a histogram of value to screen with title
------------------------------------------------------------------------- */

void MapReduce::write_histo(double value, const char *title)
{
  int histo[10],histotmp[10];
  double ave,max,min;
  histogram(1,&value,ave,max,min,10,histo,histotmp);
  if (me == 0) {
    printf("%-13s %g ave %g max %g min\n",title,ave,max,min);
    printf("%-13s","  Histogram:");
    for (int i = 0; i < 10; i++) printf(" %d",histo[i]);
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void MapReduce::histogram(int n, double *data, 
			  double &ave, double &max, double &min,
			  int nhisto, int *histo, int *histotmp)
{
  min = 1.0e20;
  max = -1.0e20;
  ave = 0.0;
  for (int i = 0; i < n; i++) {
    ave += data[i];
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }

  int ntotal;
  MPI_Allreduce(&n,&ntotal,1,MPI_INT,MPI_SUM,comm);
  double tmp;
  MPI_Allreduce(&ave,&tmp,1,MPI_DOUBLE,MPI_SUM,comm);
  ave = tmp/ntotal;
  MPI_Allreduce(&min,&tmp,1,MPI_DOUBLE,MPI_MIN,comm);
  min = tmp;
  MPI_Allreduce(&max,&tmp,1,MPI_DOUBLE,MPI_MAX,comm);
  max = tmp;

  for (int i = 0; i < nhisto; i++) histo[i] = 0;

  int m;
  double del = max - min;
  for (int i = 0; i < n; i++) {
    if (del == 0.0) m = 0;
    else m = static_cast<int> ((data[i]-min)/del * nhisto);
    if (m > nhisto-1) m = nhisto-1;
    histo[m]++;
  }

  MPI_Allreduce(histo,histotmp,nhisto,MPI_INT,MPI_SUM,comm);
  for (int i = 0; i < nhisto; i++) histo[i] = histotmp[i];
}

/* ----------------------------------------------------------------------
   setup memory alignment params
   setup memory page data structures and do initial allocation
------------------------------------------------------------------------- */

void MapReduce::allocate()
{
  allocated = 1;

  // check key,value alignment factors

  kalign = keyalign;
  valign = valuealign;

  int tmp = 1;
  while (tmp < kalign) tmp *= 2;
  if (tmp != kalign) error->all("Invalid alignment setting");
  tmp = 1;
  while (tmp < valign) tmp *= 2;
  if (tmp != valign) error->all("Invalid alignment setting");

  // talign = max of (kalign,valign,int)

  talign = MAX(kalign,valign);
  talign = MAX(talign,sizeof(int));

  kalignm1 = kalign - 1;
  valignm1 = valign - 1;
  talignm1 = talign - 1;

  // memory initialization

  if (memsize == 0) error->all("Invalid memsize setting");
  if (minpage < 0) error->all("Invalid minpage setting");
  if (maxpage && maxpage < minpage) error->all("Invalid maxpage setting");

  if (memsize > 0)
    pagesize = ((uint64_t) memsize) * 1024*1024;
  else
    pagesize = (uint64_t) (-memsize);

  if (pagesize < ALIGNFILE) error->all("Page size smaller than ALIGNFILE");

  if (minpage) allocate_page(minpage);
}

/* ----------------------------------------------------------------------
   allocate a contiguous set of N pages
------------------------------------------------------------------------- */

void MapReduce::allocate_page(int n)
{
  int nnew = npage + n;
  memptr = (char **) memory->srealloc(memptr,nnew*sizeof(char *),"MR:memptr");
  memused = (int *) memory->srealloc(memused,nnew*sizeof(int),"MR:memused");
  memcount = (int *) memory->srealloc(memcount,nnew*sizeof(int),"MR:memcount");

  char *ptr = (char *) memory->smalloc_align(n*pagesize,ALIGNFILE,"MR:page");
  memset(ptr,0,n*pagesize);

  for (int i = 0; i < n; i++) {
    memptr[npage+i] = ptr + i*pagesize;
    memused[npage+i] = 0;
    memcount[npage+i] = 0;
  }
  memcount[npage] = n;
  npage = nnew;
}

/* ----------------------------------------------------------------------
   request for numpages of contiguous memory
   satisfy request out of 1st available unused page(s)
   else allocate new page(s) if maxpage allows
   else throw error
   return ptr to memory and size of memory
   return tag for caller to use when releasing page(s) via myfree()
------------------------------------------------------------------------- */

char *MapReduce::mymalloc(int numpage, uint64_t &size, int &tag)
{
  int ipage,ok;

  for (tag = 0; tag < npage; tag++) {
    if (memused[tag]) continue;
    ok = 1;
    for (ipage = tag+1; ipage < tag+numpage; ipage++)
      if (ipage >= npage || memused[ipage] || memcount[ipage]) ok = 0;
    if (ok) break;
  }

  if (tag == npage) {
    if (maxpage && npage+numpage > maxpage)
      error->one("Cannot allocate requested memory page(s)");
    allocate_page(numpage);
  }

  for (ipage = 0; ipage < numpage; ipage++) memused[tag+ipage] = numpage;
  size = numpage*pagesize;

  return memptr[tag];
}

/* ----------------------------------------------------------------------
   free one or more pages of memory starting at tag
------------------------------------------------------------------------- */

void MapReduce::myfree(int tag)
{
  int n = memused[tag];
  for (int i = 0; i < n; i++) memused[tag++] = 0;
}

/* ----------------------------------------------------------------------
   query status of memory pages
   return # of free 1-pagers
   return ncontig = largest # of contiguous free pages available
   return max = # of pages that can still be allocated, -1 if infinite
------------------------------------------------------------------------- */

int MapReduce::memquery(int &maxcontig, int &max)
{
  int i,j;

  int n = 0;
  for (i = 0; i < npage; i++)
    if (memused[i] == 0) n++;

  maxcontig = 0;
  for (i = 0; i < npage; i++) {
    if (memused[i]) continue;
    for (j = i+1; j < npage; j++)
      if (memused[j] || memcount[j]) break;
    maxcontig = MAX(maxcontig,j-i);
  }

  if (maxpage == 0) max = -1;
  else max = maxpage-npage;
  return n;
}

/* ----------------------------------------------------------------------
   set hi-water mark for file sizes on disk
   flag = 0, file of fsize was written to disk
   flag = 1, file of fsize was deleted from disk
------------------------------------------------------------------------- */

void MapReduce::hiwater(int flag, uint64_t size)
{
  if (flag == 0) fsize += size;
  if (flag == 1) fsize -= size;
  fsizemax = MAX(fsizemax,fsize);
}
