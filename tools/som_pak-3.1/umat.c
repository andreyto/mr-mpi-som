/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  umat                                                                *
 *  - convert a SOM codebook to PS/EPS file                             *
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
#include <string.h>
#include <math.h>
#include "config.h"
#include "umat.h"

char *usage[] = {
  "umat - produce EPS/PS picture of a SOM\n",
  "Required options:\n",
  "  -cin              input codebook file\n",
  "Optional parameters:\n",
  "  -o filename       output filename (default is stdout)\n", 
  "  -eps              output EPS file (the default)\n",
  "  -ps               output PS file\n",
  "  -portrait         output PS picture in portrait mode\n",
  "  -landscape        output PS picture in landscape mode\n",
  "  -border           draw border around map units\n",
  "  -onlylabs         draw only labels\n",
  "  -nolabs           don't draw labels\n", 
  "  -W float          white treshold\n",
  "  -B float          black treshold\n",
  "  -title string     set title (default is input codebook name)\n", 
  "  -notitle          do not print title line on PS picture\n", 
  "  -font fontname    PS fontname for labels\n",
  "  -fontsize float   fontsize relative to the radius of an unit\n",
  "  -paper type       select paper size for PS output, A4 (default) or A3\n",
  "  -average          average the umatrix\n",
  "  -median           median filter the umatrix\n",
  "  -headerfile fname specify alternative postscript header file\n",
  NULL};

extern char *psheader[];

char *newstring(char *text, int len);
#define freestring(str) free((str))

int print_eps(FILE *fp, struct eps_info *einfo);
int print_page(FILE *fp, struct eps_info *einfo);
int image_size(struct eps_info *, int);
char *ps_string_filter(char *text);
int change_suffix(char *s, char *dest, char *end);

struct paper_info *get_paper_by_name(char *);
struct paper_info *get_paper_by_id(int);
int set_paper(struct paper_info* p);

int guess_mode(char *);

/* info on paper sizes */

struct paper_info papers[] = {
  {"A4", PAPER_A4, A4WIDTH, A4HEIGHT},
  {"A3", PAPER_A3, A3WIDTH, A3HEIGHT},
  {NULL, -1, 0, 0}};

int uxdim, uydim, mxdim,mydim;

int paper_type = PAPER_A4;
int page_width = A4WIDTH;
int page_height = A4HEIGHT;
int orientation = BEST;
int old_or;

float hexa_radius;
float eps_width = 1000.0;
float eps_height;
float radius;

char *fontname = DEFAULTFONT;
float fontsize = -1.0;
char *title = NULL;

int xstart = LMARGIN;
int ystart = RMARGIN;
int lmargin = LMARGIN;
int rmargin = RMARGIN;
int tmargin = TMARGIN;
int bmargin = BMARGIN;

/* int mode = OUTPUT_EPS; */
int mode = 0;
int notitle = 0; /* doesn't print file name if 1 */

int doborder = 0;

int drawblocks = 1;
int drawlabels = 1;

float white_treshold = 1.0;
float black_treshold = 0.0;

char *headerfile = NULL;

char filenamebuf[1000];

int main(int argc, char **argv)
{
  char *in_name, *out_name, *s;
  struct umatrix *umat;
  struct eps_info einfo;
  int average = 0;
  int median = 0;
  FILE *out_fp;
  struct file_info *out_fi;

  global_options(argc, argv);
  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }
  
  in_name = NULL;
  out_name = NULL;

  /* default page size is A4 */
  set_paper(get_paper_by_id(PAPER_A4));

  /* parse args */
  
  if (extract_parameter(argc, argv, "-border", OPTION2))
    doborder = 1;

  if (extract_parameter(argc, argv, "-portrait", OPTION2))
    orientation = PORTRAIT;

  if (extract_parameter(argc, argv, "-landscape", OPTION2))
    orientation = LANDSCAPE;

  if (extract_parameter(argc, argv, "-ps", OPTION2))
    mode = OUTPUT_PS;

  if (extract_parameter(argc, argv, "-eps", OPTION2))
    mode = OUTPUT_EPS;

  white_treshold = oatof(extract_parameter(argc, argv, "-W", OPTION), 1.0);

  black_treshold = oatof(extract_parameter(argc, argv, "-B", OPTION), 0.0);

  out_name = extract_parameter(argc, argv, "-o", OPTION);
  if (mode == 0)
    mode = guess_mode(out_name);

  if ((s = extract_parameter(argc, argv, "-font", OPTION)))
    fontname = s;

  fontsize = oatof(extract_parameter(argc, argv, "-fontsize", OPTION), -1.0);
  
  title = extract_parameter(argc, argv, "-title", OPTION);
  
  if (extract_parameter(argc, argv, "-notitle", OPTION2))
    notitle = 1;

  if ((s = extract_parameter(argc, argv, "-paper", OPTION)))
    {
      struct paper_info *p = get_paper_by_name(s);
      if (p == NULL)
	{
	  fprintf(stderr, "Unknown paper type: %s\n", s);
	  exit(1);
	}
      set_paper(p);
    }

  if (extract_parameter(argc, argv, "-average", OPTION2))
    average = 1;

  if (extract_parameter(argc, argv, "-median", OPTION2))
    median = 1;

  if (extract_parameter(argc, argv, "-onlylabs", OPTION2))
    drawblocks = 0;

  if (extract_parameter(argc, argv, "-nolabs", OPTION2))
    drawlabels = 0;

  in_name = extract_parameter(argc, argv, IN_CODE_FILE, ALWAYS);

  if ((s = getenv("UMAT_HEADERFILE")))
    headerfile = s;

  if ((s = extract_parameter(argc, argv, "-headerfile", OPTION)))
    headerfile = s;

  label_not_needed(1);

  if ((umat = read_map(in_name, 0, 0)) == NULL)
    {
      fprintf(stderr, "Can't load file\n");
      exit(1);
    }

  calc_umatrix(umat, 0, 0);

  if (average)
    average_umatrix(umat);

  if (median)
    median_umatrix(umat);

  if (mode == 0)
    mode = OUTPUT_EPS;

  if (orientation == BEST)
    orientation =  (umat->mxdim >= umat->mydim) ? LANDSCAPE : PORTRAIT;

  if ((out_fi = open_file(out_name, "w")) == NULL)
    {
      fprintf(stderr, "can't open output file\n");
      free_umat(umat);
      exit(1);
    }
  out_fp = fi2fp(out_fi);

  einfo.umat = umat;
  einfo.title = title ? title : in_name;
      
  image_size(&einfo, 0);
      
  if (mode == OUTPUT_EPS)
    print_eps(out_fp, &einfo);
  else
    print_page(out_fp, &einfo);

  free_umat(umat);
  umat = NULL;

  if (out_fi)
    close_file(out_fi);

  return 0;
}

/* get info about paper types by name */

struct paper_info *get_paper_by_name(char *name)
{
  struct paper_info *p;
  
  for (p = papers; p->name != NULL; p++)
    if (strcasecmp(p->name, name) == 0)
      return p;

  return NULL;
}

/* get info about paper types by type id */

struct paper_info *get_paper_by_id(int id)
{
  struct paper_info *p;
  
  for (p = papers; p->name != NULL; p++)
    if (p->id == id)
      return p;

  return NULL;
}

int set_paper(struct paper_info* p)
{
  paper_type = p->id;
  page_width = p->width;
  page_height = p->height;

  return 0;
}

int change_suffix(char *s, char *dest, char *end)
{
  char *s1;
  strcpy(dest, s);
  
  s1 = strrchr(dest, '.');
  if (s1 == NULL)
    strcat(dest, end);
  else
    strcpy(s1, end);
  
  return 0;
}

char *get_date(void)
{
  time_t curtime;

  curtime = time(NULL);
  return ctime(&curtime);
}

/* guess printing mode from filename (EPS/PS) */

int guess_mode(char *name)
{
  char *s;
  if (name == NULL)
    return 0;

  s = strrchr(name, '.');
  if (s == NULL)
    return 0;

  s++;
  if (strcasecmp(s, "ps") == 0)
    return OUTPUT_PS;

  if (strcasecmp(s, "eps") == 0)
    return OUTPUT_EPS;

  return 0;
}

int print_page(FILE *fp, struct eps_info *einfo)
{
  int w, h, xs, ys, tmp, pw, ph;
  float scale, scale1, scale2;
  char *title_str;
  struct umatrix *umat = einfo->umat;
  w = einfo->width;
  h = einfo->height;

  title_str = einfo->title;

  if (title_str && (!notitle))
    w += 24;

  pw = page_width - lmargin - rmargin;
  ph = page_height - bmargin - tmargin;

  fprintf(fp, "%%!PS-Adobe-2.0\n%%%%Pages: 1\n");
  fprintf(fp, "%%%%Creator: umat V1.0\n");
  fprintf(fp, "%%%%CreationDate: %s", ps_string_filter(get_date()));

  if (orientation == LANDSCAPE)
    {
      fprintf(fp, "%d %d translate 90 rotate\n",
	      lmargin + pw, bmargin);
      tmp = pw;
      pw = ph;
      ph = tmp;
    }
  else /* PORTRAIT */
    {
      fprintf(fp, "%d %d translate\n", lmargin, bmargin);
    }

  scale1 = (float)pw / (float)w;
  scale2 = (float)ph / (float)h;
  scale = min(scale1, scale2);

  xs = (pw - scale * w) * 0.5;
  ys = (ph - scale * h) * 0.5;

  fprintf(fp, "gsave %d %d translate %f dup scale\n",
	  xs, ys, scale);

  if ((title_str) && (!notitle))
    {
      fprintf(fp, "gsave /Helvetica findfont 18 scalefont setfont\n");
      fprintf(fp, "0 setgray %f %f 8 add moveto\n",
	      (float)2.0, einfo->height);
      fprintf(fp, "(%s - Dim: %d, Size: %d*%d units, %s neighborhood) show\n",
	      ps_string_filter(title_str), umat->dim, umat->mxdim, umat->mydim,
	      umat->codes->neigh == NEIGH_GAUSSIAN ? "gaussian" : "bubble");
      fprintf(fp, "grestore\n");
    }

  /* print map */
  print_eps(fp, einfo);

  fprintf(fp, "grestore\nshowpage\n");

  return 0;
}


char *newstring(char *text, int len)
{
  char *s, *s2;
  if (text == NULL)
    return NULL;

  if (len < 0)
    {
      fprintf(stderr, "warning: negative length string '%s'\n", text);
      len = 0;
    }

  if (len == 0)
    len = strlen(text);
  
  s = malloc(len + 1);
  if (s == NULL)
    {
      fprintf(stderr, "can't alloc memory for string '%s'\n", text);
      return NULL;
    }      

  s2 = s;
  while ((*s++ = *text++));
  return s2;
}
  
int print_header(FILE *fp, char *hname)
{
  int c;
  struct file_info *fi;
  FILE *header_file;

  if (hname)
    {
      if ((fi = open_file(hname, "r")) == NULL)
	{
	  fprintf(stderr, "umat: can't read PS header file %s\n", hname);
	  return 1;
	}
      header_file = fi2fp(fi);
      while ((c = fgetc(header_file)) != EOF)
	fputc(c, fp);
      close_file(fi);
    }
  else
    print_lines(fp, psheader);

  return 0;
}

/* calculate image size */
int image_size(struct eps_info *einfo, int width)
{
  struct umatrix *umat = einfo->umat;

  if (width <= 0) 
    width = 1000;

  einfo->width = width;
  switch (umat->topol)
    {
    case TOPOL_RECT:
      einfo->xstep = (float)width / (float)umat->uxdim;
      einfo->ystep = einfo->xstep;
      einfo->height = umat->uydim * einfo->ystep;
      einfo->x0 = einfo->xstep * 0.5;
      einfo->y0 = einfo->ystep * 0.5;
      einfo->radius = einfo->xstep * 0.5;
      break;
    case TOPOL_HEXA:
      einfo->xstep = (float)width / (float)(umat->uxdim + 1);
      einfo->ystep = einfo->xstep * sqrt(3) * 0.5;
      einfo->radius = einfo->xstep / sqrt(3);
      einfo->height = (umat->uydim - 1) * einfo->ystep + 2.0 * einfo->radius;
      einfo->x0 = einfo->xstep * 0.5;
      einfo->y0 = einfo->radius;
      break;
    default:
      fprintf(stderr, "unknown topology %d\n", umat->topol);
      return 1;
      break;
    }
      
  return 0;
}

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


/* print an eps object of file */

int print_eps(FILE *fp, struct eps_info *einfo)
{
  float color;
  float xstep, ystep, radius;
  int x, y, numlabs, i;
  char *label;
  eptr p;
  char *select_topol_s, *draw_block_s, *start_row_s, *end_row_s;
  struct data_entry *dtmp;
  struct umatrix *umat = einfo->umat;

  end_row_s = "NL";

  switch (umat->topol) 
    {
    case TOPOL_HEXA:
      select_topol_s = "topol_hexa";
      draw_block_s = "H";
      start_row_s = "XSH";
      break;
    case TOPOL_RECT:
      select_topol_s = "topol_rect";
      draw_block_s = "R";
      start_row_s = "XSR";
      break;
    default:
      fprintf(stderr, "can't print topology %d\n", umat->topol);
      return 0;
      break;
    }

  /*  radius = eps_width / (2.0 * sqrt(3) * (float)umat->mxdim); */
  xstep = einfo->xstep;
  ystep = einfo->ystep;
  radius = einfo->radius;

  /* print eps headers */
  fprintf(fp, "%%!PS-Adobe-3.0 EPSF-3.0\n");
  fprintf(fp, "%%%%BoundingBox: 0 0 %d %d\n", 
	  (int)ceil(einfo->width), (int)ceil(einfo->height));
  
  fprintf(fp, "%%%%Title: %s\n%%%%Creator: umat V1.0\n", 
	  ps_string_filter(einfo->title));
  
  fprintf(fp, "%%%%CreationDate: %s", ps_string_filter(get_date()));
  fprintf(fp, "%%%%Pages: 0\n");
  fprintf(fp, "%%%%DocumentFonts: %s\n%%%%DocumentNeededFonts: %s\n",
	  fontname, fontname);
  fprintf(fp, "%%%%EndComments\n");
  
  /* COPY HEADER */
  if (print_header(fp, headerfile))
    return 1; /* error */

  fprintf(fp, "/radius %f def\n/xstep %f def\n/ystep %f def\n",
	  radius, xstep, ystep);

  /* font selection */
  fprintf(fp, "%%%%IncludeFont: %s\n", fontname);
  fprintf(fp, "/fontname /%s def\n", fontname);
  if (fontsize > 0.0) 
    fprintf(fp, "/fontsize %f def\n", fontsize);
  fprintf(fp, "selfont\n");
  
  fprintf(fp, "/doborder %s def\n", doborder ? "true" : "false");
  /* print umat */

  fprintf(fp, "/wt %f def /bt %f def\n", white_treshold, black_treshold);

  fprintf(fp, "/y 0 def\n/xoff %f def\n/yoff %f def\n", 
	  einfo->x0, (einfo->height - einfo->y0));

  if (drawblocks)
    for (y = 0; y < umat->uydim; y++)
      {
	fprintf(fp, "%s ", start_row_s);
	for (x = 0; x < umat->uxdim; x++)
	  fprintf(fp, "%d %s ", (int)(100 * umat->uvalue[x][y]), draw_block_s);
	fprintf(fp, "%s\n", end_row_s);
      }
  
  /* print labels */

  fprintf(fp, "/y 0 def\n/xoff %f def\n/yoff %f def\n", 
	  einfo->x0, (einfo->height - einfo->y0));

  if (drawlabels)
    {
      dtmp = rewind_entries(umat->codes, &p);
      
      for (y = 0; y < umat->mydim; y++)
	{
	  fprintf(fp, "%s ", start_row_s);
	  for (x = 0; x < umat->mxdim; x++)
	    {
	      numlabs = dtmp->num_labs;
	      if (!drawblocks)
		color = 100;
	      else
		color = (umat->uvalue[2 * x][2 * y] * 100);
	      
	      if (numlabs)
		{
		  if (numlabs == 1)
		    {
		      /* one label */
		      label = find_conv_to_lab(get_entry_label(dtmp));
		      fprintf(fp, "(%s) %d LAB ", 
			      ps_string_filter(label), (int)color);
		    }
		  else
		    {
		      /* multiple labels */
		      for (i = 0; i < numlabs; i++)
			{
			  label = find_conv_to_lab(get_entry_labels(dtmp, i));
			  
			  if (label == LABEL_EMPTY) 
			    {
			      numlabs = i;
			      break;
			    }
			  
			  fprintf(fp, "(%s) ", ps_string_filter(label));
			}
		      
		      fprintf(fp, "%d %d ML ", numlabs, (int)color);
		    }
		}
	      else
		fprintf(fp, "%d LN ", (int)color);
	      dtmp = next_entry(&p);
	    }
	  /* do newline twice because labels are printed on every other row */
	  fprintf(fp, "%s %s\n", end_row_s, end_row_s);
	}
    }
  fprintf(fp, "end\n");
  fprintf(fp, "%% end of EPS object\n");

  return 0;
}


