/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  sammon.c                                                            *
 *  - generates a Sammon mapping from a given list                      *
 *                                                                      *
 *  Version 3.0                                                         *
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
#include <math.h>
#include <float.h>
#include <string.h>
#include "lvq_pak.h"
#include "datafile.h"
#include "labels.h"
  
#define TRUE  1
#define FALSE 0
#define MAGIC 0.2
  
#define max(x,y) ((x) > (y) ? (x) : (y))
#define min(x,y) ((x) < (y) ? (x) : (y))

static char *usage[] = {
  "sammon - generates a Sammon mapping from a given list\n",
  "Required parameters:\n",
  "  -cin filename         input codebook file\n",
  "  -cout filename        output codebook filename\n",
  "  -rlen integer         running length\n",
  "Optional parameters:\n",
  "  -eps                  produce an EPS picture of map\n",
  "  -ps                   produce an PS picture of map\n",
  "  -rand integer         seed for random number generator. 0 is current time\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  NULL};


char *base_name;

/* ps_string_filter: escape ps special characters in string. Returns a
   pointer to a static buffer containing the escaped string. */

char *ps_string_filter(char *text)
{

  static char ps_string[1024]; /* This should be enough for most purposes */
  char *s = ps_string, c;

  if (text)
    while ((c = *text++))
      {
	switch (c) 
	  {
	  case '(':
	  case ')':
	  case '\\':
	    *s++ = '\\';
	    break;
	  default:
	    break;
	  }
	*s++ = c;
      }

  *s++ = 0;

  return ps_string;
}

struct entries *remove_identicals(struct entries *codes, int *rem)
{
  struct data_entry *entr, *entr1, *prev;
  DIST_FUNCTION *distance = vector_dist_euc; /* tama parameettereista */
  eptr p1, p2;
  int dim = codes->dimension;
  int ii, ij;

  /* Compute the mutual distances between entries */
  /* Remove the identical entries from the list */

  *rem = 0;
  entr = rewind_entries(codes, &p1);
  ii = 1; ij = 1;
  p2.parent = p1.parent;
  while (entr != NULL) {
    p2.current = p1.current;
    p2.index = p1.index;
    entr1 = next_entry(&p2);
    ij = ii + 1;
    prev = p1.current;
    while (entr1 != NULL) {
      if (distance(entr, entr1, dim) == 0.0) {
	fprintf(stderr, "Identical entries in codebook ");
        fprintf(stderr, "(entries %d, %d), removing one.\n", ii, ij);
        *rem = 1;
	codes->num_entries--;
	prev->next = entr1->next;
	p2.current = prev;
	free_entry(entr1);
	entr1 = next_entry(&p2);
        ij++; ij++;
      }
      else {
	prev = entr1;
        entr1 = next_entry(&p2);
        ij++;
      }
    }
    entr = next_entry(&p1);
    ii++;
  }
  
  return(codes);
}

struct entries *sammon_iterate(struct entries *codes, int length)
{
  int i, j, k;
  long noc = 0;
  float e1x, e1y, e2x, e2y;
  float dpj;
  float dq, dr, dt;
  struct data_entry *entr, *entr1, *prev, tmp;
  float *x, *y;
  float *xu, *yu, *dd;
  float xd, yd;
  float xx, yy;
  float e, tot;
  int mutual;
  float d, ee;
  DIST_FUNCTION *distance = vector_dist_euc; /* tama jotenkin parametrina */
  eptr p1, p2;
  struct entries *newent;
  int dim = codes->dimension;

  /* How many entries? */
  noc = codes->num_entries;

  ifverbose(3)
    fprintf(stderr, "%ld entries in codebook\n", noc);

  /* Allocate dynamical memory */
  x = (float *) oalloc(sizeof(float) * noc);
  y = (float *) oalloc(sizeof(float) * noc);
  xu = (float *) oalloc(sizeof(float) * noc);
  yu = (float *) oalloc(sizeof(float) * noc);
  dd = (float *) oalloc(sizeof(float) * (noc * (noc - 1) / 2));

  /* Initialize the tables */
  for (i = 0; i < noc; i++) {
    x[i] = (float) (orand() % noc) / noc;
    y[i] = (float) (i) / noc;
  }

  /* Compute the mutual distances between entries */
  mutual = 0;
  entr = rewind_entries(codes, &p1);
  entr = next_entry(&p1);
  while (entr != NULL) {
    entr1 = rewind_entries(codes, &p2);
    while (entr1 != entr) {
      dd[mutual] = distance(entr, entr1, dim);
      if (dd[mutual] == 0.0) {
	fprintf(stderr, "Identical entries in codebook\n");
      }
      mutual++;
      entr1 = next_entry(&p2);
    }
    entr = next_entry(&p1);
  }
  
  /* Iterate */
  for (i = 0; i < length; i++) {
    for (j = 0; j < noc; j++) {
      e1x = e1y = e2x = e2y = 0.0;
      for (k = 0; k < noc; k++) {
	if (j == k)
	  continue;
	xd = x[j] - x[k];
	yd = y[j] - y[k];
	dpj = (float) sqrt((double) xd * xd + yd * yd);

	/* Calculate derivatives */
	if (k > j)
	  dt = dd[k * (k - 1) / 2 + j];
	else
	  dt = dd[j * (j - 1) / 2 + k];
	dq = dt - dpj;
	dr = dt * dpj;
	e1x += xd * dq / dr;
	e1y += yd * dq / dr;
	e2x += (dq - xd * xd * (1.0 + dq / dpj) / dpj) / dr;
	e2y += (dq - yd * yd * (1.0 + dq / dpj) / dpj) / dr;
      }
      /* Correction */
      xu[j] = x[j] + MAGIC * e1x / fabs(e2x);
      yu[j] = y[j] + MAGIC * e1y / fabs(e2y);
    }
    
    /* Move the center of mass to the center of picture */
    xx = yy = 0.0;
    for (j = 0; j < noc; j ++) { 
      xx += xu[j];
      yy += yu[j];
    }
    xx /= noc;
    yy /= noc;
    for (j = 0; j < noc; j ++) {
      x[j] = xu[j] - xx;
      y[j] = yu[j] - yy;
    }
    
    /* Error in distances */
    e = tot = 0.0;
    mutual = 0;
    for (j = 1; j < noc; j ++)
      for (k = 0; k < j; k ++) {
	d = dd[mutual];
	tot += d;
	xd = x[j] - x[k];
	yd = y[j] - y[k];
	ee = d - (float) sqrt((double) xd * xd + yd * yd);
	e += (ee * ee / d);
	mutual++;
      }
    e /= tot;
    ifverbose(2)
      fprintf(stdout, "Mapping error: %7.3f\n", e);
    if (verbose(-1) == 1)
      mprint((long) (length-i));
  }
  if (verbose(-1) == 1)
    mprint((long) 0);
  if (verbose(-1) == 1)
    fprintf(stderr, "\n");

  newent = alloc_entries();
  newent->dimension = 2;
  newent->xdim = codes->xdim;
  newent->ydim = codes->ydim;
  newent->topol = codes->topol;
  newent->neigh = codes->neigh;

  /* Copy the data to return variable */

  prev = &tmp;
  prev->next = NULL;
  for (i = 0, entr = rewind_entries(codes, &p1); i < noc; i++, entr = next_entry(&p1)) {
    entr1 = init_entry(newent, NULL);
    prev->next = entr1;
    prev = entr1;
    entr1->points[0] = x[i];
    entr1->points[1] = y[i];
    copy_entry_labels(entr1, entr);
  }
  newent->entries = tmp.next;
  newent->num_entries = noc;

  return(newent);
}

void save_entries_in_eps(struct entries *spics, char *filename, int ps, int rem)
{
  char str[100];
  FILE *fp;
  struct data_entry *spetr;
  eptr p;
  float xmi, xma, ymi, yma;
  float frac;
  int label;
  int xc, yc, ec;

  xmi = FLT_MAX;
  xma = FLT_MIN;
  ymi = FLT_MAX;
  yma = FLT_MIN;

  if (ps)
    sprintf(str, "%s_sa.ps", filename);
  else
    sprintf(str, "%s_sa.eps", filename);

  fp = fopen(str, "w");
  if (fp == NULL) {
    printf("Can't open file%s\n", str);
    return;
  }

  spetr = rewind_entries(spics, &p);
  while (spetr != NULL) {
    if (xmi > spetr->points[0])
      xmi = spetr->points[0];
    if (xma < spetr->points[0])
      xma = spetr->points[0];
    if (ymi > spetr->points[1])
      ymi = spetr->points[1];
    if (yma < spetr->points[1])
      yma = spetr->points[1];
    spetr = next_entry(&p);
  }

  if ((xma - xmi)*1.5 > (yma - ymi))
    frac = 510.0 / (xma - xmi);
  else
    frac = 760.0 / (yma - ymi);

  spetr = rewind_entries(spics, &p);
  while (spetr != NULL) {
    spetr->points[0] = spetr->points[0] - xmi;
    spetr->points[1] = spetr->points[1] - ymi;
    spetr = next_entry(&p);
  }

  if (ps) {
    /* print ps header */
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    fprintf(fp, "%%%%Title: %s\n%%%%Creator: sammon\n", "undefined");
    fprintf(fp, "%%%%Pages: 1\n%%%%EndComments\n");
    fprintf(fp, "40 40 translate\n");
    fprintf(fp, "/gscale %f def\n", frac);
    fprintf(fp, "gscale dup scale\n");
  }
  else {
    /* print eps header */
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    fprintf(fp, "%%%%Title: %s\n%%%%Creator: sammon\n", "undefined");
    fprintf(fp, "%%%%BoundingBox: 0 0 %f %f\n", xma - xmi, yma - ymi);
    fprintf(fp, "%%%%Pages: 0\n%%%%EndComments\n");
    fprintf(fp, "/gscale %f def\n", frac);
  }

  fprintf(fp, "/Helvetica findfont 12 gscale div scalefont setfont\n");
  fprintf(fp, "/radius %f def\n", 2.0/frac);
  fprintf(fp, "/LN\n");
  fprintf(fp, "{newpath\n");
  fprintf(fp, "radius 0 360 arc fill\n");
  fprintf(fp, "} def\n");
  fprintf(fp, "/LP\n");
  fprintf(fp, "{dup stringwidth pop\n");
  fprintf(fp, "-2 div 0 rmoveto show} def\n");
  fprintf(fp, "%f setlinewidth\n", 0.2/frac);
  fprintf(fp, "0 setgray\n");

  spetr = rewind_entries(spics, &p);
  while (spetr != NULL) {
    fprintf(fp, "%f %f LN\n", spetr->points[0], spetr->points[1]);
    if ((label = get_entry_label(spetr)) != LABEL_EMPTY) {
      fprintf(fp, "%f %f moveto\n", spetr->points[0], spetr->points[1]);
      fprintf(fp, "(%s) LP\n", ps_string_filter(find_conv_to_lab(label)));
    }

    spetr = next_entry(&p);
  }

  if (!rem) {
    xc = 0;
    spetr = rewind_entries(spics, &p);
    while (spetr != NULL) {
      if (xc == 0) {
	fprintf(fp, "newpath\n");
	fprintf(fp, "%f %f moveto\n", spetr->points[0], spetr->points[1]);
      }
      else {
	fprintf(fp, "%f %f lineto\n", spetr->points[0], spetr->points[1]);
	if (xc == spics->xdim-1)
	  fprintf(fp, "stroke\n");
      }
  
      spetr = next_entry(&p);
      xc++;
      if (xc == spics->xdim)
	xc = 0;
    }
  
    yc = 0;
    while (yc < spics->xdim) {
      xc = 0;
      ec = 0;
      spetr = rewind_entries(spics, &p);
      while (spetr != NULL) {
	if ((ec == 0) && (xc == yc)) {
	  fprintf(fp, "newpath\n");
	  fprintf(fp, "%f %f moveto\n", spetr->points[0], spetr->points[1]);
	}
	else if (xc == yc) {
	  fprintf(fp, "%f %f lineto\n", spetr->points[0], spetr->points[1]);
	  if (ec == spics->ydim-1)
	    fprintf(fp, "stroke\n");
	}
    
	spetr = next_entry(&p);
	xc++;
	if (xc == spics->xdim) {
	  xc = 0;
	  ec++;
	}
      }
      yc++;
    }
  }

  if (ps)
    fprintf(fp, "showpage\n");

  fclose(fp);
}

int main(int argc, char **argv)
{
  long length;
  int randomize;
  int eps = 0, ps = 0;
  char *in_code_file;
  char *out_code_file;
  struct entries *codes;
  struct entries *spics;
  int removed;
  
  global_options(argc, argv);
  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }
  in_code_file = extract_parameter(argc, argv, IN_CODE_FILE, ALWAYS);
  out_code_file = extract_parameter(argc, argv, OUT_CODE_FILE, ALWAYS);
  length = (int) oatoi(extract_parameter(argc, argv, RUNNING_LENGTH, ALWAYS),
                       1);
  randomize = (int) oatoi(extract_parameter(argc, argv, RANDOM, OPTION), 0);
  eps = (extract_parameter(argc, argv, "-eps", OPTION2) != NULL);
  ps = (extract_parameter(argc, argv, "-ps", OPTION2) != NULL);
  
  label_not_needed(1);
  
  ifverbose(2)
    fprintf(stderr, "Code entries from file %s\n", in_code_file);
  if ((codes = open_entries(in_code_file)) == NULL)
    {
      fprintf(stderr, "can't open code file %s\n", in_code_file);
      exit(1);
    }
  
  init_random(randomize);

  /* Remove identical entries from the codebook */
  codes = remove_identicals(codes, &removed);
  
  spics = sammon_iterate(codes, length);
  
  ifverbose(2)
    fprintf(stderr, "Save code entries to file %s\n", out_code_file);
  save_entries(spics, out_code_file);

  {
    char *p;

    base_name = ostrdup(out_code_file);
    p = strrchr(base_name, '.');
    if (p != NULL)
      *p = '\0';
  }

  /* Don't draw lines when the file is not a map file */
  if ((codes->topol != TOPOL_RECT)  && (codes->topol != TOPOL_HEXA))
    removed = 1;

  if (ps || eps)
    save_entries_in_eps(spics, base_name, ps, removed);

  close_entries(codes);
  close_entries(spics);

  return(0);
}


