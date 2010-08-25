#ifndef _INCLUDED_UMAT_H_
#define _INCLUDED_UMAT_H_
/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  umat.h                                                              *
 *  - header file for SOM to PS converter                               *
 *                                                                      *
 *  Version 1.0                                                         *
 *  Date: 1 Mar 1995                                                    *
 *                                                                      *
 *  NOTE: This program package is copyrighted in the sense that it      *
 *  may be used for scientific purposes. The package as a whole, or     *
 *  parts thereof, cannot be included or used in any commercial         *
 *  application without written permission granted by its producents.   *
 *  No programs contained in this package may be copied for commercial  *
 *  distribution.                                                       *
 *                                                                      *
 *  All comments  concerning this program package may be sent to the    *
 *  e-mail address 'lvq@cochlea.hut.fi'.                                *
 *                                                                      *
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "lvq_pak.h"
#include "datafile.h"

#define ON 1
#define OFF 0

#define RHOMB 0
#define HEXAGON 1

#define X 1
#define Y 2

#define AVERAGE 1
#define MEDIAN 2

#define DEFAULTFONT "Helvetica"

struct umatrix {
  /* codebook part */
  int mxdim, mydim;      /* map dimensions */
  int topol;             /* topology type */
  int dim;               /* dimension of cedebook vectors */
  float ***mvalue;       /* codebook vector values */
  struct entries *codes; /* the codebook */

  /* the umatrix */
  int uxdim, uydim;      /* dimensions */
  float **uvalue;        /* values */
};

/* functions in 'map.c' */ 

struct umatrix *alloc_umat(void);
int free_umat(struct umatrix *);
struct umatrix *read_map(char *mapfile, int swapx, int swapy);
int calc_umatrix(struct umatrix *, int, int);
void swap_umat(struct umatrix *,int,int);
int average_umatrix(struct umatrix *umat);
int median_umatrix(struct umatrix *umat);

/* functions in 'median.c' */
float median3(float,float, float );
float median4(float,float, float , float );
float median5(float,float, float , float , float );
float median6(float,float, float , float , float , float );
float median7(float,float, float , float , float , float , float );

/* paper sizes in points */

#define A4HEIGHT 841 
#define A4WIDTH 595
#define A3WIDTH A4HEIGHT
#define A3HEIGHT (2*A4WIDTH)

/* paper types */
#define PAPER_A4 1
#define PAPER_A3 2

/* margins (in points) */
#define LMARGIN 36
#define RMARGIN 36
#define BMARGIN 36
#define TMARGIN 36

/* orientations */
#define PORTRAIT 1
#define LANDSCAPE 2
#define BEST 0  /* decide orientation of image based on image aspect */

/* output mode */
#define OUTPUT_PS 1 /* output PS page */
#define OUTPUT_EPS 2 /* output EPS image */

struct paper_info {
  char *name;
  int id;
  int width, height;
};

struct eps_info {
  struct umatrix *umat;
  float width, height;
  float xstep, ystep, radius;
  float x0, y0;
  char *title;
};

#define min(a,b) ((a) < (b) ? (a) : (b))

#endif /* _INCLUDED_UMAT_H_ */










