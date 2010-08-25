
# 
CC=cc
CFLAGS=-O
LDFLAGS=-s
LDLIBS=-lm
LD=$(CC)

## NetBSD-1.0 (tested on Amiga)
## 
#CC=gcc
#CFLAGS=-O2
#LDFLAGS=-s
#LDLIBS=-lm
#LD=$(CC)

# DEC Alpha/OSF
#
#CC=cc
#CFLAGS=-O2 
#LDFLAGS=-s
#LDLIBS=-lm
#LD=$(CC)

## Amiga/AmigaDOS
##  Tested on an Amiga 2000 w/ 68030+68882 with gcc and dmake, OS 3.1
#CC=gcc
#CFLAGS=-O2 -m68020 -m68881
#LDFLAGS=-s
#LDLIBS=-lm
#LD=$(CC)

## HPUX
##
#CC=c89
#CFLAGS=-O
#LDFLAGS=-s
#LDLIBS=-lm
#LD=$(CC)

## Linux
##
#CC=gcc
#CFLAGS=-O2
#LDFLAGS=-s
#LDLIBS=-lm
#LD=$(CC)

## SGI
##
#CC=cc
#CFLAGS=-O2 -mips1
#LDFLAGS=-s
#LDLIBS=-lm
#LD=$(CC)

TESTFILES=ex.dat ex_fts.dat ex_ndy.dat ex_fdy.dat
OBJS=som_rout.o lvq_pak.o fileio.o labels.o datafile.o version.o
UMATOBJS=umat.o map.o median.o header.o
OTHERFILES=header.ps

all: vcal mapinit vsom qerror randinit lininit visual sammon planes vfind umat

vsom:	vsom.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ vsom.o $(OBJS) $(LDLIBS)

vsomtest:	vsomtest.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ vsomtest.o $(OBJS) $(LDLIBS)

qerror:	qerror.o $(OBJS) 
	$(LD) $(LDFLAGS) -o $@ qerror.o $(OBJS) $(LDLIBS)

mapinit:	mapinit.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ mapinit.o $(OBJS) $(LDLIBS)

randinit: mapinit
	rm -f $@
	ln mapinit $@

lininit: mapinit 
	rm -f $@
	ln mapinit $@

vcal:	vcal.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ vcal.o $(OBJS) $(LDLIBS)

visual:	visual.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ visual.o $(OBJS) $(LDLIBS)

sammon:	sammon.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ sammon.o $(OBJS) $(LDLIBS)

planes: planes.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ planes.o $(OBJS) $(LDLIBS)
	
vfind:	vfind.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ vfind.o $(OBJS) $(LDLIBS)

# Umat
umat:	$(UMATOBJS) $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(UMATOBJS) $(OBJS) $(LDLIBS)

map.o umat.o:	umat.h fileio.h datafile.o lvq_pak.h labels.h
median.o:	umat.h

# for making development distribution
dist:
	rm -f som_pak-3.0.tar.gz som_pak-3.0.tar 
	tar -cvf som_pak-3.0.tar *.c *.h Makefile.* $(TESTFILES) $(OTHERFILES)
	gzip -v -9 som_pak-3.0.tar

version.o: version.h

# for testing
#TRAND=-rand 1
#ALPHA_TYPE=-alpha_type inverse_t
#BUFFER=-buffer 500

example :
	./randinit -din ex.dat -cout ex.cod -xdim 12 -ydim 8 -topol hexa \
  -neigh bubble -rand 123
	./vsom  -din ex.dat -cin ex.cod  -cout ex.cod -rlen 1000 \
  -alpha 0.05 -radius 10 $(TRAND) $(ALPHA_TYPE) $(BUFFER)
	./vsom     -din ex.dat -cin ex.cod  -cout ex.cod -rlen 10000 \
  -alpha 0.02 -radius 3 $(TRAND) $(ALPHA_TYPE) $(BUFFER)
	./qerror   -din     ex.dat -cin ex.cod
	./vcal     -din ex_fts.dat -cin ex.cod -cout ex.cod
	./visual   -din ex_ndy.dat -cin ex.cod -dout ex.nvs
	./visual   -din ex_fdy.dat -cin ex.cod -dout ex.fvs

fileio.o:	fileio.h
datafile.o:	lvq_pak.h datafile.h fileio.h
labels.o:	labels.h lvq_pak.h
lvq_pak.o:	lvq_pak.h datafile.h fileio.h labels.h
som_rout.o:	som_rout.h lvq_pak.h datafile.h fileio.h labels.h

vcal.o mapinit.o vsom.o qerror.o visual.o sammon.o:\
	lvq_pak.h datafile.h fileio.h labels.h som_rout.h 

