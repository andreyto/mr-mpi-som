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

// C interface to MapReduce library
// ditto for Fortran, scripting language, or other hi-level languages

#include "cmapreduce.h"
#include "mapreduce.h"
#include "keyvalue.h"

using namespace MAPREDUCE_NS;

void *MR_create(MPI_Comm comm)
{
  MapReduce *mr = new MapReduce(comm);
  return (void *) mr;
}

void *MR_create_mpi()
{
  MapReduce *mr = new MapReduce();
  return (void *) mr;
}

void *MR_create_mpi_finalize()
{
  MapReduce *mr = new MapReduce(0.0);
  return (void *) mr;
}

void MR_destroy(void *MRptr)
{
  MapReduce *mr = (MapReduce *) MRptr;
  delete mr;
}

void *MR_copy(void *MRptr)
{
  MapReduce *mr = (MapReduce *) MRptr;
  MapReduce *mr2 = mr->copy();
  return (void *) mr2;
}

uint64_t MR_add(void *MRptr, void *MRptr2)
{
  MapReduce *mr = (MapReduce *) MRptr;
  MapReduce *mr2 = (MapReduce *) MRptr2;
  return mr->add(mr2);
}

uint64_t MR_aggregate(void *MRptr, int (*myhash)(char *, int))
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->aggregate(myhash);
}

uint64_t MR_broadcast(void *MRptr, int root)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->broadcast(root);
}

uint64_t MR_clone(void *MRptr)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->clone();
}

uint64_t MR_close(void *MRptr)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->close();
}

uint64_t MR_collapse(void *MRptr, char *key, int keybytes)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->collapse(key,keybytes);
}

uint64_t MR_collate(void *MRptr, int (*myhash)(char *, int))
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->collate(myhash);
}

uint64_t MR_compress(void *MRptr,
		     void (*mycompress)(char *, int, char *,
					int, int *, void *, void *),
		     void *APPptr)
{
  typedef void (CompressFunc)(char *, int, char *,
			      int, int *, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  CompressFunc *appcompress = (CompressFunc *) mycompress;
  return mr->compress(appcompress,APPptr);
}

uint64_t MR_convert(void *MRptr)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->convert();
}

uint64_t MR_gather(void *MRptr, int numprocs)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->gather(numprocs);
}

uint64_t MR_map(void *MRptr, int nmap,
		void (*mymap)(int, void *, void *),
		void *APPptr)
{
  typedef void (MapFunc)(int, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(nmap,appmap,APPptr);
}

uint64_t MR_map_add(void *MRptr, int nmap,
		    void (*mymap)(int, void *, void *),
		    void *APPptr, int addflag)
{
  typedef void (MapFunc)(int, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(nmap,appmap,APPptr,addflag);
}

uint64_t MR_map_file_list(void *MRptr, char *file,
			  void (*mymap)(int, char *, void *, void *),
			  void *APPptr)
{
  typedef void (MapFunc)(int, char *, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(file,appmap,APPptr);
}

uint64_t MR_map_file_list_add(void *MRptr, char *file,
			      void (*mymap)(int, char *, void *, void *),
			      void *APPptr, int addflag)
{
  typedef void (MapFunc)(int, char *, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(file,appmap,APPptr,addflag);
}

uint64_t MR_map_file_char(void *MRptr, int nmap, int nfiles, char **files,
			  char sepchar, int delta,
			  void (*mymap)(int, char *, int, void *, void *),
			  void *APPptr)
{
  typedef void (MapFunc)(int, char *, int, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(nmap,nfiles,files,sepchar,delta,appmap,APPptr);
}

uint64_t MR_map_file_char_add(void *MRptr, int nmap, int nfiles, char **files,
			      char sepchar, int delta,
			      void (*mymap)(int, char *, int, void *, void *),
			      void *APPptr, int addflag)
{
  typedef void (MapFunc)(int, char *, int, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(nmap,nfiles,files,sepchar,delta,appmap,APPptr,addflag);
}

uint64_t MR_map_file_str(void *MRptr, int nmap, int nfiles, char **files,
			 char *sepstr, int delta,
			 void (*mymap)(int, char *, int, void *, void *),
			 void *APPptr)
{
  typedef void (MapFunc)(int, char *, int, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(nmap,nfiles,files,sepstr,delta,appmap,APPptr);
}

uint64_t MR_map_file_str_add(void *MRptr, int nmap, int nfiles, char **files,
			     char *sepstr, int delta,
			     void (*mymap)(int, char *, int, void *, void *),
			     void *APPptr, int addflag)
{
  typedef void (MapFunc)(int, char *, int, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(nmap,nfiles,files,sepstr,delta,appmap,APPptr,addflag);
}

uint64_t MR_map_mr(void *MRptr, void *MRptr2,
		   void (*mymap)(uint64_t, char *, int, 
				 char *, int, void *, void *),
		   void *APPptr)
{
  typedef void (MapFunc)(uint64_t, char *, int, char *, int,
			 KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapReduce *mr2 = (MapReduce *) MRptr2;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(mr2,appmap,APPptr);
}

uint64_t MR_map_mr_add(void *MRptr, void *MRptr2,
		       void (*mymap)(uint64_t, char *, int, 
				     char *, int, void *, void *),
		       void *APPptr, int addflag)
{
  typedef void (MapFunc)(uint64_t, char *, int, char *, int,
			 KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  MapReduce *mr2 = (MapReduce *) MRptr2;
  MapFunc *appmap = (MapFunc *) mymap;
  return mr->map(mr2,appmap,APPptr,addflag);
}

void MR_open(void *MRptr)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->open();
}

void MR_open_add(void *MRptr, int addflag)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->open(addflag);
}

void MR_print(void *MRptr, int proc, int nstride, int kflag, int vflag)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->print(proc,nstride,kflag,vflag);
}

uint64_t MR_reduce(void *MRptr,
		   void (*myreduce)(char *, int, char *,
				    int, int *, void *, void *),
		   void *APPptr)
{
  typedef void (ReduceFunc)(char *, int, char *,
			    int, int *, KeyValue *, void *);
  MapReduce *mr = (MapReduce *) MRptr;
  ReduceFunc *appreduce = (ReduceFunc *) myreduce;
  return mr->reduce(appreduce,APPptr);
}

uint64_t MR_scrunch(void *MRptr, int numprocs, char *key, int keybytes)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->scrunch(numprocs,key,keybytes);
}

uint64_t MR_multivalue_blocks(void *MRptr, int *pnblock)
{
  MapReduce *mr = (MapReduce *) MRptr;
  int nblock;
  uint64_t nvalue_total = mr->multivalue_blocks(nblock);
  *pnblock = nblock;
  return nvalue_total;
}

int MR_multivalue_block(void *MRptr, int iblock,
			char **ptr_multivalue, int **ptr_valuesizes)
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->multivalue_block(iblock,ptr_multivalue,ptr_valuesizes);
}

uint64_t MR_sort_keys(void *MRptr, int (*mycompare)(char *, int, char *, int))
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->sort_keys(mycompare);
}

uint64_t MR_sort_values(void *MRptr, 
			int (*mycompare)(char *, int, char *, int))
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->sort_values(mycompare);
}

uint64_t MR_sort_multivalues(void *MRptr, int (*mycompare)(char *, int, 
							   char *, int))
{
  MapReduce *mr = (MapReduce *) MRptr;
  return mr->sort_multivalues(mycompare);
}

void MR_kv_stats(void *MRptr, int level)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->kv_stats(level);
}

void MR_kmv_stats(void *MRptr, int level)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->kmv_stats(level);
}

void MR_cummulative_stats(void *MRptr, int level, int reset)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->cummulative_stats(level,reset);
}

void MR_set_mapstyle(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->mapstyle = value;
}

void MR_set_all2all(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->all2all = value;
}

void MR_set_verbosity(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->verbosity = value;
}

void MR_set_timer(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->timer = value;
}

void MR_set_memsize(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->memsize = value;
}

void MR_set_minpage(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->minpage = value;
}

void MR_set_maxpage(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->maxpage = value;
}

void MR_set_keyalign(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->keyalign = value;
}

void MR_set_valuealign(void *MRptr, int value)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->valuealign = value;
}

void MR_set_fpath(void *MRptr, char *str)
{
  MapReduce *mr = (MapReduce *) MRptr;
  mr->set_fpath(str);
}

void MR_kv_add(void *KVptr, char *key, int keybytes,
	       char *value, int valuebytes)
{
  KeyValue *kv = (KeyValue *) KVptr;
  kv->add(key,keybytes,value,valuebytes);
}

void MR_kv_add_multi_static(void *KVptr, int n, 
			    char *key, int keybytes,
			    char *value, int valuebytes)
{
  KeyValue *kv = (KeyValue *) KVptr;
  kv->add(n,key,keybytes,value,valuebytes);
}

void MR_kv_add_multi_dynamic(void *KVptr, int n, 
			     char *key, int *keybytes,
			     char *value, int *valuebytes)
{
  KeyValue *kv = (KeyValue *) KVptr;
  kv->add(n,key,keybytes,value,valuebytes);
}
