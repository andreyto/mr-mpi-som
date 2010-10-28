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

/* C or Fortran style interface to MapReduce library */
/* ifdefs allow this file to be included in a C program */

#include "mpi.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

void *MR_create(MPI_Comm comm);
void *MR_create_mpi();
void *MR_create_mpi_finalize();
void MR_destroy(void *MRptr);

void *MR_copy(void *MRptr);

uint64_t MR_add(void *MRptr, void *MRptr2);
uint64_t MR_aggregate(void *MRptr, int (*myhash)(char *, int));
uint64_t MR_broadcast(void *MRptr, int);
uint64_t MR_clone(void *MRptr);
uint64_t MR_close(void *MRptr);
uint64_t MR_collapse(void *MRptr, char *key, int keybytes);
uint64_t MR_collate(void *MRptr, int (*myhash)(char *, int));
uint64_t MR_compress(void *MRptr, 
		     void (*mycompress)(char *, int, char *, int, int *, 
					void *KVptr, void *APPptr),
		     void *APPptr);
uint64_t MR_convert(void *MRptr);
uint64_t MR_gather(void *MRptr, int numprocs);

uint64_t MR_map(void *MRptr, int nmap,
		void (*mymap)(int, void *KVptr, void *APPptr),
		void *APPptr);
uint64_t MR_map_add(void *MRptr, int nmap,
		    void (*mymap)(int, void *KVptr, void *APPptr),
		    void *APPptr, int addflag);
uint64_t MR_map_file_list(void *MRptr, char *file,
			  void (*mymap)(int, char *, 
					void *KVptr, void *APPptr),
			  void *APPptr);
uint64_t MR_map_file_list_add(void *MRptr, char *file,
			      void (*mymap)(int, char *, 
					    void *KVptr, void *APPptr),
			      void *APPptr, int addflag);
uint64_t MR_map_file_char(void *MRptr, int nmap, int nfiles, char **files,
			  char sepchar, int delta,
			  void (*mymap)(int, char *, int, 
					void *KVptr, void *APPptr),
			  void *APPptr);
uint64_t MR_map_file_char_add(void *MRptr, int nmap, int nfiles, char **files,
			      char sepchar, int delta,
			      void (*mymap)(int, char *, int, 
					    void *KVptr, void *APPptr),
			      void *APPptr, int addflag);
uint64_t MR_map_file_str(void *MRptr, int nmap, int nfiles, char **files,
			 char *sepstr, int delta,
			 void (*mymap)(int, char *, int, 
				       void *KVptr, void *APPptr),
			 void *APPptr);
uint64_t MR_map_file_str_add(void *MRptr, int nmap, int nfiles, char **files,
			     char *sepstr, int delta,
			     void (*mymap)(int, char *, int, 
					   void *KVptr, void *APPptr),
			     void *APPptr, int addflag);
uint64_t MR_map_mr(void *MRptr, void *MRptr2,
		   void (*mymap)(uint64_t, char *, int, char *, int, 
				 void *KVptr, void *APPptr),
		   void *APPptr);
uint64_t MR_map_mr_add(void *MRptr, void *MRptr2,
		       void (*mymap)(uint64_t, char *, int, char *, int, 
				     void *KVptr, void *APPptr),
		       void *APPptr, int addflag);
void MR_open(void *MRptr);
void MR_open_add(void *MRptr, int addflag);
void MR_print(void *MRptr, int proc, int nstride, int kflag, int vflag);
uint64_t MR_reduce(void *MRptr,
		   void (*myreduce)(char *, int, char *,
				    int, int *, void *KVptr, void *APPptr),
		   void *APPptr);
uint64_t MR_scrunch(void *MRptr, int numprocs, char *key, int keybytes);

uint64_t MR_multivalue_blocks(void *MRptr);
int MR_multivalue_block(void *MRptr, int iblock,
			char **ptr_multivalue, int **ptr_valuesizes);

uint64_t MR_sort_keys(void *MRptr, 
		      int (*mycompare)(char *, int, char *, int));
uint64_t MR_sort_values(void *MRptr,
			int (*mycompare)(char *, int, char *, int));
uint64_t MR_sort_multivalues(void *MRptr,
			     int (*mycompare)(char *, int, char *, int));

void MR_kv_stats(void *MRptr, int level);
void MR_kmv_stats(void *MRptr, int level);
void MR_cummulative_stats(void *MRptr, int level, int reset);

void MR_set_mapstyle(void *MRptr, int value);
void MR_set_all2all(void *MRptr, int value);
void MR_set_verbosity(void *MRptr, int value);
void MR_set_timer(void *MRptr, int value);
void MR_set_memsize(void *MRptr, int value);
void MR_set_minpage(void *MRptr, int value);
void MR_set_maxpage(void *MRptr, int value);
void MR_set_keyalign(void *MRptr, int value);
void MR_set_valuealign(void *MRptr, int value);
void MR_set_fpath(void *MRptr, char *str);

void MR_kv_add(void *KVptr, char *key, int keybytes, 
	       char *value, int valuebytes);
void MR_kv_add_multi_static(void *KVptr, int n,
			    char *key, int keybytes,
			    char *value, int valuebytes);
void MR_kv_add_multi_dynamic(void *KVptr, int n,
			     char *key, int *keybytes,
			     char *value, int *valuebytes);

#ifdef __cplusplus
}
#endif
