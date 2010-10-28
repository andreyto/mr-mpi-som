# ----------------------------------------------------------------------
#   MR-MPI = MapReduce-MPI library
#   http://www.cs.sandia.gov/~sjplimp/mapreduce.html
#   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories
#
#   Copyright (2009) Sandia Corporation.  Under the terms of Contract
#   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
#   certain rights in this software.  This software is distributed under 
#   the modified Berkeley Software Distribution (BSD) License.
#
#   See the README file in the top-level MapReduce directory.
# -------------------------------------------------------------------------

# Python wrapper on MapReduce-MPI library via ctypes

import types
from ctypes import *
from cPickle import dumps,loads

class mrmpi:
  def __init__(self,comm=None):

    # attempt to load parallel library first, serial library next

    try:
      self.lib = CDLL("_mrmpi.so")
    except:
      try:
        self.lib = CDLL("_mrmpi_serial.so")
      except:
        raise StandardError,"Could not load MR-MPI dynamic library"

    # setup callbacks
    
    self.lib.MR_create.restype = c_void_p
    self.lib.MR_copy.restype = c_void_p

    COMPAREFUNC = CFUNCTYPE(c_void_p,POINTER(c_char),c_int,
                            POINTER(c_char),c_int)
    self.compare_def = COMPAREFUNC(self.compare_callback)

    COMPRESSFUNC = CFUNCTYPE(c_void_p,POINTER(c_char),c_int,
                             POINTER(c_char),c_int,POINTER(c_int),
                             c_void_p,c_void_p)
    self.compress_def = COMPRESSFUNC(self.compress_callback)

    HASHFUNC = CFUNCTYPE(c_void_p,POINTER(c_char),c_int)
    self.hash_def = HASHFUNC(self.hash_callback)
    
    MAPFUNC = CFUNCTYPE(c_void_p,c_int,c_void_p,c_void_p)
    self.map_def = MAPFUNC(self.map_callback)
    
    MAP_FILE_LIST_FUNC = CFUNCTYPE(c_void_p,c_int,c_char_p,c_void_p,c_void_p)
    self.map_file_list_def = MAP_FILE_LIST_FUNC(self.map_file_list_callback)
    
    MAP_FILE_STR_FUNC = CFUNCTYPE(c_void_p,c_int,POINTER(c_char),c_int,
                                 c_void_p,c_void_p)
    self.map_file_str_def = MAP_FILE_STR_FUNC(self.map_file_str_callback)
    
    MAP_MR_FUNC = CFUNCTYPE(c_void_p,c_int,POINTER(c_char),c_int,
                            POINTER(c_char),c_int,c_void_p,c_void_p)
    self.map_mr_def = MAP_MR_FUNC(self.map_mr_callback)

    REDUCEFUNC = CFUNCTYPE(c_void_p,POINTER(c_char),c_int,
                           POINTER(c_char),c_int,POINTER(c_int),
                           c_void_p,c_void_p)
    self.reduce_def = REDUCEFUNC(self.reduce_callback)

    if comm == None: self.mr = self.lib.MR_create_mpi()
    elif type(comm) == types.IntType: self.mr = self.lib.MR_create(comm)
    elif type(comm) == types.FloatType:
      self.mr = self.lib.MR_create_mpi_finalize()
    else: raise StandardError,"Could not create an MR library instance"

    # hardwire keyalign and valuealign to 1 because of pickling

    self.lib.MR_set_keyalign(self.mr,1)
    self.lib.MR_set_valuealign(self.mr,1)

  def __del__(self):
    if self.mr: self.lib.MR_destroy(self.mr)

  def destroy(self):
    self.lib.MR_destroy(self.mr)
    self.mr = None
    
  def copy(self,mr):
    cmr = self.lib.MR_copy(self.mr,mr.mr)
    pymr = mrmpi()
    self.lib.MR_destroy(pymr.mr)
    pymr.mr = cmr
    return pymr

  def add(self,mr):
    n = self.lib.MR_add(self.mr,mr.mr)
    return n
  
  def aggregate(self,hash=None):
    if hash:
      self.hash_caller = hash
      n = self.lib.MR_aggregate(self.mr,self.hash_def)
    else:
      n = self.lib.MR_aggregate(self.mr,None)
    return n

  def broadcast(self,root):
    n = self.lib.MR_broadcast(self.mr,root)
    return n

  def clone(self):
    n = self.lib.MR_clone(self.mr)
    return n

  def close(self):
    n = self.lib.MR_close(self.mr)
    return n

  def collapse(self,key):
    ckey = dumps(key,1)
    n = self.lib.MR_collapse(self.mr,ckey,len(ckey))
    return n

  def collate(self,hash=None):
    if hash:
      self.hash_caller = hash
      n = self.lib.MR_collate(self.mr,self.hash_def)
    else:
      n = self.lib.MR_collate(self.mr,None)
    return n

  def compress(self,commpress,ptr=None):
    self.compress_caller = compress
    self.compress_argcount = compress.func_code.co_argcount
    self.compress_ptr = ptr
    n = self.lib.MR_compress(self.mr,self.compress_def,None)
    return n

  def compress_callback(self,ckey,keybytes,multivalue,nvalues,valuesizes,
                        kv,dummy):
    self.kv = kv
    key = loads(ckey[:keybytes])
    mvalue = []
    start = 0
    for i in xrange(nvalues):
      stop = start + valuesizes[i]
      value = loads(multivalue[start:stop])
      mvalue.append(value)
      start = stop
    if self.compress_argcount == 3: self.compress_caller(key,mvalue,self)
    else: self.compress_caller(key,mvalue,self,self.compress_ptr)

  def convert(self):
    n = self.lib.MR_convert(self.mr)
    return n

  def gather(self,nprocs):
    n = self.lib.MR_gather(self.mr,nprocs)
    return n

  def hash_callback(self,ckey,keybytes):
    key = loads(ckey[:keybytes])
    return self.hash_caller(key,self)

  def map(self,nmap,map,ptr=None,addflag=0):
    self.map_caller = map
    self.map_argcount = map.func_code.co_argcount
    self.map_ptr = ptr
    if not addflag:
      n = self.lib.MR_map(self.mr,nmap,self.map_def,None)
    else:
      n = self.lib.MR_map_add(self.mr,nmap,self.map_def,None,addflag)
    return n

  def map_callback(self,itask,kv,dummy):
    self.kv = kv
    if self.map_argcount == 2: self.map_caller(itask,self)
    else: self.map_caller(itask,self,self.map_ptr)
    
  def map_file_list(self,file,map,ptr=None,addflag=0):
    self.map_caller = map
    self.map_argcount = map.func_code.co_argcount
    self.map_ptr = ptr
    if not addflag:
      n = self.lib.MR_map_file_list(self.mr,file,self.map_file_list_def,None)
    else:
      n = self.lib.MR_map_file_list_add(self.mr,file,self.map_file_list_def,
                                        None,addflag)
    return n

  def map_file_list_callback(self,itask,file,kv,dummy):
    self.kv = kv
    if self.map_argcount == 3: self.map_caller(itask,file,self)
    else: self.map_caller(itask,file,self,self.map_ptr)
    
  def map_file_char(self,nmap,files,sepchar,delta,map,
                    ptr=None,addflag=0):
    self.map_caller = map
    self.map_argcount = map.func_code.co_argcount
    self.map_ptr = ptr
    cfiles = (c_char_p*len(files))(*files)   # array of C strings from list
    if not addflag:
      n = self.lib.MR_map_file_char(self.mr,nmap,len(files),cfiles,
                                    ord(sepchar),delta,
                                    self.map_file_str_def,None)
    else:
      n = self.lib.MR_map_file_char_add(self.mr,nmap,len(files),cfiles,
                                        ord(sepchar),delta,
                                        self.map_file_str_def,None,addflag)
    return n
    
  def map_file_str(self,nmap,files,sepstr,delta,map,
                   ptr=None,addflag=0):
    self.map_caller = map
    self.map_argcount = map.func_code.co_argcount
    self.map_ptr = ptr
    cfiles = (c_char_p*len(files))(*files)   # array of C strings from list
    if not addflag:
      n = self.lib.MR_map_file_str(self.mr,nmap,len(files),cfiles,
                                   sepstr,delta,
                                   self.map_file_str_def,None)
    else:
      n = self.lib.MR_map_file_str_add(self.mr,nmap,len(files),cfiles,
                                       sepstr,delta,
                                       self.map_file_str_def,None,addflag)
    return n

  def map_file_str_callback(self,itask,cstr,size,kv,dummy):
    self.kv = kv
    str = cstr[:size]
    if self.map_argcount == 3: self.map_caller(itask,str,self)
    else: self.map_caller(itask,str,self,self.map_ptr)
    
  def map_mr(self,mr,map,ptr=None,addflag=0):
    self.map_caller = map
    self.map_argcount = map.func_code.co_argcount
    self.map_ptr = ptr
    if not addflag:
      n = self.lib.MR_map_mr(self.mr,mr.mr,self.map_mr_def,None)
    else:
      n = self.lib.MR_map_mr_add(self.mr,mr.mr,self.map_mr_def,None,addflag)
    return n

  def map_mr_callback(self,itask,ckey,keybytes,cvalue,valuebytes,kv,dummy):
    self.kv = kv
    key = loads(ckey[:keybytes])
    value = loads(cvalue[:valuebytes])
    if self.map_argcount == 4: self.map_caller(itask,key,value,self)
    else: self.map_caller(itask,key,value,self,self.map_ptr)

  def open(self,addflag=0):
    if not addflag: self.lib.MR_open(self.mr)
    else: self.lib.MR_open_add(self.mr,addflag)

  def print(self,proc,nstride,kflag,vflag):
    self.lib.print(self.mr,proc,nstride,kflag,vflag)

  def reduce(self,reduce,ptr=None):
    self.reduce_caller = reduce
    self.reduce_argcount = reduce.func_code.co_argcount
    self.reduce_ptr = ptr
    n = self.lib.MR_reduce(self.mr,self.reduce_def,None)
    return n

  def reduce_callback(self,ckey,keybytes,multivalue,nvalues,valuesizes,
                      kv,dummy):
    self.kv = kv
    key = loads(ckey[:keybytes])
    mvalue = []
    start = 0
    for i in xrange(nvalues):
      stop = start + valuesizes[i]
      value = loads(multivalue[start:stop])
      mvalue.append(value)
      start = stop
    if self.reduce_argcount == 3: self.reduce_caller(key,mvalue,self)
    else: self.reduce_caller(key,mvalue,self,self.reduce_ptr)

  def scrunch(self,nprocs,key):
    ckey = dumps(key,1)
    n = self.lib.scrunch(self.mr,nprocs,ckey,len(ckey))
    return n

  def multivalue_blocks(self):
    n = self.lib.multivalue_blocks(self.mr)
    return n

  def multivalue_block(self,iblock,multivalue,valuesizes):
    # might need to pass back a tuple of 3 things?
    # doc these 2 funcs in Python interface
    # check that reduce and compress callbacks can handle NULL case and MR ptr
    n = self.lib.multivalue_block(self.mr,iblock,multivalue,valuesizes)
    return n

  def sort_keys(self,compare):
    self.compare_caller = compare
    n = self.lib.MR_sort_keys(self.mr,self.compare_def)
    return n

  def sort_values(self,compare):
    self.compare_caller = compare
    n = self.lib.MR_sort_values(self.mr,self.compare_def)
    return n

  def sort_multivalues(self,compare):
    self.compare_caller = compare
    n = self.lib.MR_sort_multivalues(self.mr,self.compare_def)
    return n

  def compare_callback(self,cobj1,len1,cobj2,len2):
    obj1 = loads(cobj1[:len1])
    obj2 = loads(cobj2[:len2])
    return self.compare_caller(obj1,obj2)

  def kv_stats(self,level):
    self.lib.MR_kv_stats(self.mr,level)

  def kmv_stats(self,level):
    self.lib.MR_kmv_stats(self.mr,level)

  def mapstyle(self,value):
    self.lib.MR_set_mapstyle(self.mr,value)

  def all2all(self,value):
    self.lib.MR_set_all2all(self.mr,value)

  def verbosity(self,value):
    self.lib.MR_set_verbosity(self.mr,value)

  def timer(self,value):
    self.lib.MR_set_timer(self.mr,value)

  def memsize(self,value):
    self.lib.MR_set_memsize(self.mr,value)

  def minpage(self,value):
    self.lib.MR_set_minpage(self.mr,value)

  def maxpage(self,value):
    self.lib.MR_set_maxpage(self.mr,value)

  def add(self,key,value):
    ckey = dumps(key,1)
    cvalue = dumps(value,1)
    self.lib.MR_kv_add(self.kv,ckey,len(ckey),cvalue,len(cvalue))

  def add_multi_static(self,keys,values):
    n = len(keys)
    ckeys = ""
    cvalues = ""
    for i in xrange(n):
      ckey = dumps(keys[i],1)
      cvalue = dumps(values[i],1)
      ckeys += ckey
      cvalues += cvalue
    keybytes = len(ckeys)/n
    valuebytes = len(cvalues)/n
    self.lib.MR_kv_add_multi_dynamic(self.kv,n,
                                     ckeys,keybytes,cvalues,valuebytes)

  def add_multi_dynamic(self,keys,values):
    n = len(keys)
    ckeys = ""
    cvalues = ""
    keybytes = (c_int*n)()
    valuebytes = (c_int*n)()
    for i in xrange(n):
      ckey = dumps(keys[i],1)
      cvalue = dumps(values[i],1)
      keybytes[i] = len(ckey)
      valuebytes[i] = len(cvalue)
      ckeys += ckey
      cvalues += cvalue
    self.lib.MR_kv_add_multi_dynamic(self.kv,n,
                                     ckeys,keybytes,cvalues,valuebytes)
