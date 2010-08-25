/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  planes.c                                                            *
 *  -convert the map planes into postscript format for printing         *
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
#include <float.h>
#include <string.h>
#include <math.h>
#include "lvq_pak.h"
#include "datafile.h"
#include "labels.h"
#include "som_rout.h"

#define XMSTEP 40
#define YMSTEP ((int) ((int) XSTEP * 0.87))

struct traj_point {
  int index;
  struct traj_point *next;
};

struct teach_params params;
int plane;
char *mapname, *base_name;
int traj_ready = 0;
int *dens;
struct traj_point traj_root;

int ps = 0;

int XSTEP, YSTEP;

static char *usage[] = {
  "planes - produce EPS images of self-organizing maps\n",
  "Required parameters:\n",
  "  -cin filename         codebook file\n",
  "Optional parameters:\n",
  "  -din filename         input data\n",
  "  -plane integer        map reference vector component to be used (0=all)\n",
  "  -ps integer           produce PS-code (instead of EPS)\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  NULL};

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

void print_plane(int plane)
{
  float cv;
  int xp, yp;
  int index, offset = 0;
  int label;
  float maxval, minval;
  struct teach_params *teach = &params;
  struct entries *codes = teach->codes;
  struct data_entry *codetmp;
  FILE *fp;
  char str[100];
  int xsize, ysize;
  eptr p;

  if (ps)
    sprintf(str, "%s_p%d.ps", base_name, plane+1);
  else
    sprintf(str, "%s_p%d.eps", base_name, plane+1);

  fp = fopen(str, "w");

  if (codes->topol == topol_type("hexa"))
    offset = XSTEP/2;

  xsize = XSTEP * codes->xdim + offset;
  ysize = YSTEP * codes->ydim;

  if (ps) {
    /* print ps header */
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    fprintf(fp, "%%%%Title: %s\n%%%%Creator: planes\n", "undefined");
    fprintf(fp, "%%%%Pages: 1\n%%%%EndComments\n");
    fprintf(fp, "550 40 translate\n");
    fprintf(fp, "90 rotate\n");
    fprintf(fp, "760 %d div 510 %d div lt\n", xsize, ysize);
    fprintf(fp, "   {760 %d 0 sub div} {510 %d div} ifelse\n", xsize, ysize);
    fprintf(fp, "/gscale exch def\n");
    fprintf(fp, "gscale dup scale\n");
  }
  else {
    /* print eps header */
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    fprintf(fp, "%%%%Title: %s\n%%%%Creator: planes\n", "undefined");
    fprintf(fp, "%%%%BoundingBox: 0 0 %d %d\n", xsize, ysize);
    fprintf(fp, "%%%%Pages: 0\n%%%%EndComments\n");
  }

  /* print header */
  fprintf(fp, "/fontsize %d def\n", (int) (XSTEP/3));
  fprintf(fp, "0 %d translate\n", ysize);
  fprintf(fp, "1 -1 scale\n");

  /* Find the lower and upper limits for gray level scaling */
  minval = FLT_MAX;
  maxval = -FLT_MAX;
  codetmp = rewind_entries(codes, &p);
  while (codetmp != NULL) {
    if (maxval < codetmp->points[plane])
      maxval = codetmp->points[plane];
    if (minval > codetmp->points[plane])
      minval = codetmp->points[plane];

    codetmp = next_entry(&p);
  }

  /* initialize gray circles */
  fprintf(fp, "/radius %d def\n", (int) (XSTEP/2.2));
  fprintf(fp, "/LN\n");
  fprintf(fp, "{ setgray\n");
  fprintf(fp, "newpath\n");
  fprintf(fp, "radius 0 360 arc fill\n");
  fprintf(fp, "} def\n");

  /* Print the gray level values */
  index = 0;
  codetmp = rewind_entries(codes, &p);
  while (codetmp != NULL) {
    /* Compute the gray level of next unit */
    if ((maxval - minval) != 0.0)
      cv = 0.05 + 0.9 * (codetmp->points[plane] - minval) /
	  (maxval - minval);
    else
      cv = 0.5;
    xp = XSTEP * (index % codes->xdim) + XSTEP/2;
    yp = YSTEP * (index / codes->xdim) + YSTEP/2;
    if ((index / codes->xdim) % 2) xp += offset;

    /* Print the value */
    fprintf(fp, "%d %d %f LN\n", xp, yp, cv);

    codetmp = next_entry(&p);
    index++;
  }

  /* initialize label printing */
  fprintf(fp, "0 setgray\n");
  fprintf(fp, "/Helvetica findfont fontsize scalefont setfont\n");
  fprintf(fp, "/LP\n");
  fprintf(fp, "{ \n");
  fprintf(fp, "1 -1 scale dup stringwidth pop\n");
  fprintf(fp, "-2 div 0 rmoveto show\n");
  fprintf(fp, "1 -1 scale } def\n");

  /* Print the labels */
  index = 0;
  codetmp = rewind_entries(codes, &p);
  while (codetmp != NULL) {
    /* If there is label, then print it */
    xp = XSTEP * (index % codes->xdim) + XSTEP/2;
    yp = YSTEP * (index / codes->xdim) + YSTEP/2;
    if ((index / codes->xdim) % 2) xp += offset;

    if ((label = get_entry_label(codetmp)) != LABEL_EMPTY)
      fprintf(fp, "%d %d moveto (%s) LP\n", xp, yp, ps_string_filter(find_conv_to_lab(label)));

    codetmp = next_entry(&p);
    index++;
  }

  if (ps) {
    fprintf(fp, "showpage\n");
  }

  fclose(fp);
}

void scan_data_traj(struct entries *codes, struct entries *data)
{
  struct teach_params *teach = &params;
  WINNER_FUNCTION *find_winner = teach->winner;
  int i;
  struct data_entry *datatmp;
  struct traj_point *traj_po;
  struct winner_info winner;
  eptr p;

  dens = (int *) malloc(sizeof(int) * codes->xdim * codes->ydim);
  for (i = 0; i < codes->xdim * codes->ydim; i++)
    dens[i] = 0;
  traj_root.next = NULL;
  traj_root.index = 0;
  traj_po = &traj_root;

  datatmp = rewind_entries(data, &p);
  /* Scan all input entries */
  while (datatmp != NULL) {

    /* check if the vector is empty */
    if (find_winner(codes, datatmp, &winner, 1) == 0) {
       traj_po->next =
	   (struct traj_point *) malloc(sizeof(struct traj_point) * 1);
      traj_po = traj_po->next;
      traj_po->next = NULL;
  
      /* save the winner index */
      traj_po->index = -1;
   }
    else {
      traj_po->next =
	   (struct traj_point *) malloc(sizeof(struct traj_point) * 1);
      traj_po = traj_po->next;
      traj_po->next = NULL;
  
      /* save the winner index */
      traj_po->index = winner.index;

      /* Increment the winner */
      dens[winner.index]++;
    }

    /* Take the next input entry */
    datatmp = next_entry(&p);;
  }
  traj_ready = 1;
}

void print_trajectory()
{
  int i, j;
  int offset = 0;
  struct teach_params *teach = &params;
  struct entries *data = teach->data;
  struct entries *codes = teach->codes;
  FILE *fp;
  char str[100];
  int xsize, ysize;

  if (ps)
    sprintf(str, "%s_tr.ps", base_name);
  else
    sprintf(str, "%s_tr.eps", base_name);
  fp = fopen(str, "w");

  if (codes->topol == TOPOL_HEXA)
    offset = XSTEP/2;

  xsize = XSTEP * codes->xdim + offset;
  ysize = YSTEP * codes->ydim;

  if (ps) {
    /* print ps header */
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    fprintf(fp, "%%%%Title: %s\n%%%%Creator: planes\n", "undefined");
    fprintf(fp, "%%%%Pages: 1\n%%%%EndComments\n");
    fprintf(fp, "550 40 translate\n");
    fprintf(fp, "90 rotate\n");
    fprintf(fp, "760 %d div 510 %d div lt\n", xsize, ysize);
    fprintf(fp, "   {760 %d 0 sub div} {510 %d div} ifelse\n", xsize, ysize);
    fprintf(fp, "/gscale exch def\n");
    fprintf(fp, "gscale dup scale\n");
  }
  else {
    /* print eps header */
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    fprintf(fp, "%%%%Title: %s\n%%%%Creator: planes\n", "undefined");
    fprintf(fp, "%%%%BoundingBox: 0 0 %d %d\n", xsize, ysize);
    fprintf(fp, "%%%%Pages: 0\n%%%%EndComments\n");
  }

  /* print header */
  fprintf(fp, "0 %d translate\n", ysize);
  fprintf(fp, "1 -1 scale\n");

  /* draw empty circles */
  fprintf(fp, "1 setlinewidth\n");
  fprintf(fp, "0.8 setgray\n");
  fprintf(fp, "/radius %d def\n", (int) (XSTEP/2.2));
  fprintf(fp, "/LN\n");
  fprintf(fp, "{ newpath\n");
  fprintf(fp, "radius 0 360 arc\n");
  fprintf(fp, "stroke } def\n");
  for (i = 0; i < codes->xdim; i++)
    for (j = 0; j < codes->ydim; j++) {
      fprintf(fp, "%d %d LN\n",
           ((int) i * XSTEP + XSTEP/2 + ((j % 2) ? offset : 0)),
           ((int) j * (int) YSTEP + YSTEP/2));
    }

  /* Set linetype for trajectory */
  fprintf(fp, "%d setlinewidth\n", XSTEP/10);
  fprintf(fp, "1 setlinejoin\n");
  fprintf(fp, "1 setlinecap\n");
  fprintf(fp, "0 setgray\n");

  /* If data values are present, print trajectory */
  if (data != NULL) {
    int cul[2];
    int bpos;
    int first = 1;
    struct traj_point *traj_po;

    if (traj_ready == 0)
      scan_data_traj(codes,data);

    traj_po = traj_root.next;
    /* Scan all input entries */
    while (traj_po != NULL) {
  
      bpos = traj_po->index;
      if (bpos == -1) {
        if (!first) {
          fprintf(fp, "stroke\n");
        }
        first = 1;

        /* Take the next input entry */
        traj_po = traj_po->next;

        /* Skip until next real data point */
        continue;
      }
 
      /* Draw the trajectory */
      if (first) {
	first = 0;

	cul[0] = XSTEP * (bpos % codes->xdim) + XSTEP/2;
	cul[1] = YSTEP * (bpos / codes->xdim) + YSTEP/2;
	if (((bpos / codes->xdim) % 2)) cul[0] += offset;

        /* print the initialization */
        fprintf(fp, "newpath\n");
        fprintf(fp, "%d %d moveto\n", cul[0], cul[1]);
      }
      else {
	cul[0] = XSTEP * (bpos % codes->xdim) + XSTEP/2;
	cul[1] = YSTEP * (bpos / codes->xdim) + YSTEP/2;
	if (((bpos / codes->xdim) % 2)) cul[0] += offset;

	/* Print one segment of trajectory */
        fprintf(fp, "%d %d lineto\n", cul[0], cul[1]);
      }
  
      /* Take the next input entry */
      traj_po = traj_po->next;
    }
    fprintf(fp, "stroke\n");
  }

  if (ps) {
    fprintf(fp, "showpage\n");
  }

  fclose(fp);
}

void print_all_planes()
{
  int i;
  struct teach_params *teach = &params;
  struct entries *codes = teach->codes;

  for (i = 0; i < codes->dimension; i++)
    print_plane(i);
}

int main(int argc, char **argv)
{
  int error, buffer;
  char *in_code_file;
  char *in_data_file = NULL;
  struct entries *codes = NULL;
  struct entries *data = NULL;
  int xsize, ysize;
  int offset = 0;

  error = 0;
  global_options(argc, argv);
  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }

  in_code_file = extract_parameter(argc, argv, IN_CODE_FILE, ALWAYS);
  in_data_file = extract_parameter(argc, argv, IN_DATA_FILE, OPTION);
  plane = (int) oatoi(extract_parameter(argc, argv, PLANE, OPTION), 1);
  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);
  ps = oatoi(extract_parameter(argc, argv, "-ps", OPTION), 0);

  mapname = in_code_file;
  {
    char *p;

    base_name = ostrdup(mapname);
    p = strrchr(base_name, '.');
    if (p != NULL)
      *p = (char) NULL;
  }

  label_not_needed(1);
  ifverbose(2)
    fprintf(stdout, "Codebook entries are read from file %s\n", in_code_file);
  codes = open_entries(in_code_file);
  if (codes == NULL)
    {
      fprintf(stderr, "cant open code file '%s'\n", in_code_file);
      error = 1;
      goto cleanup;
    }

  if (codes->topol < TOPOL_HEXA) {
    printf("File %s is not a map file\n", in_code_file);
    error = 1;
    goto cleanup;
  }

  if (in_data_file != NULL) {
    ifverbose(2)
      fprintf(stdout, "Data entries are read from file %s\n", in_data_file);
    if ((data = open_entries(in_data_file)) == NULL)
      {
	fprintf(stderr, "cant open data file '%s'\n", in_data_file);
	error = 1;
	goto cleanup;
      }

    if (data->dimension > codes->dimension) {
      fprintf(stderr, "Dimensions in data and codebook files are different");
      error = 1;
      goto cleanup;
    }
    data->flags.skip_empty = 0;
  }

  if (plane > codes->dimension) {
    fprintf(stderr, "Required plane is bigger than codebook vector dimension");
    error = 1;
    goto cleanup;
  }

  set_teach_params(&params, codes, data, buffer);
  set_som_params(&params);

  XSTEP = XMSTEP;
  YSTEP = XMSTEP;
  if (codes->topol == topol_type("hexa")) {
    offset = XSTEP/2;
    YSTEP = YMSTEP;
  }

  /* Every unit gets a fixed size in the picture */
  xsize = XSTEP * codes->xdim + offset;
  ysize = YSTEP * codes->ydim;

  /* indexing the tables by plane-1 */
  plane--;

  if (plane == -1)
    print_all_planes();
  else
    print_plane(plane);

  if (data != NULL)
    print_trajectory();

 cleanup:
  if (codes)
    close_entries(codes);
  if (data)
    close_entries(data);

  return(error);
}
