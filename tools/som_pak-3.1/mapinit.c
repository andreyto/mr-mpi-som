/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  mapinit.c (for randinit and lininit)                                *
 *  - initializes the codebook vectors for SOM learning                 *
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
#include "lvq_pak.h"
#include "som_rout.h"
#include "datafile.h"

#define IT_UNKNOWN 0
#define IT_LIN     1  /* lininit */
#define IT_RAND    2  /* randinit */

static char *usage[] = {
  "mapinit/randinit/lininit - initializes the codebook vectors for SOM learning\n",
  "Initialzation type is determined from program name (randinit or lininit)\n",
  "or is selected with the -init option.\n",
  "Required parameters:\n",
  "  -din filename         input data\n",
  "  -cout filename        output codebook filename\n",
  "  -topol type           topology type of map, hexa or rect\n",
  "  -neigh type           neighborhood type, bubble or gaussian\n",
  "  -xdim integer\n",
  "  -ydim integer         dimensions of the map\n",
  "Optional parameters:\n",
  "  -init type            initialization type, rand or lin. Overrides the\n",
  "                        type determined from the program name\n",
  "  -rand integer         seed for random number generator. 0 is current time\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  NULL};


int main(int argc, char **argv)
{
  int number_of_codes, randomize, xdim, ydim, buffer;
  int topol;
  int neigh;
  char *in_data_file, *out_code_file, *s, *progname;
  struct entries *data, *codes;
  int init_type = IT_UNKNOWN;
  char comments[256];

  global_options(argc, argv);

  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }

  progname = getprogname();

  if (strcasecmp(progname, "lininit") == 0)
    init_type = IT_LIN;
  else if (strcasecmp(progname, "randinit") == 0)
    init_type = IT_RAND;

  in_data_file = extract_parameter(argc, argv, IN_DATA_FILE, ALWAYS);
  out_code_file = extract_parameter(argc, argv, OUT_CODE_FILE, ALWAYS);

  randomize = (int) oatoi(extract_parameter(argc, argv, RANDOM, OPTION), 0);
  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);

  /* the topology type of the map */
  s = extract_parameter(argc, argv, TOPOLOGY, ALWAYS);
  topol = topol_type(s);
  if (topol == TOPOL_UNKNOWN) 
    {
      fprintf(stderr, "Unknown topology type %s\n", s);
      exit(1);
    }

  /* the neighbourhood type */
  s = extract_parameter(argc, argv, NEIGHBORHOOD, ALWAYS);
  neigh = neigh_type(s);
  if (neigh == NEIGH_UNKNOWN) 
    {
      fprintf(stderr, "Unknown neighborhood type %s\n", s);
      exit(1);
    }

  xdim = (int) oatoi(extract_parameter(argc, argv, XDIM, ALWAYS), 0);
  ydim = (int) oatoi(extract_parameter(argc, argv, YDIM, ALWAYS), 0);

  /* initialization type, overrides the one guessed from program name */
  s = extract_parameter(argc, argv, "-init", OPTION);
  if (s)
    if (strcmp(s, "lin") == 0)
      init_type = IT_LIN;
    else if (strcmp(s, "rand") == 0)
      init_type = IT_RAND;

  if (parameters_left()) {
    fprintf(stderr, "Extra parameters in command line ignored\n");
  }

  if (init_type == IT_UNKNOWN)
    {
      fprintf(stderr, "Unknown initialization type %s\n", s ? s : progname);
      exit(1);
    }
  
  label_not_needed(1);

  number_of_codes = xdim * ydim;
  if (number_of_codes <= 0) {
    fprintf(stderr, "Dimensions of map (%d %d) are incorrect\n", xdim, ydim);
    exit(1);
  }
  if (xdim < 0) {
    fprintf(stderr, "Dimensions of map (%d %d) are incorrect\n", xdim, ydim);
    exit(1);
  }
  
  ifverbose(2)
    fprintf(stderr, "Input entries are read from file %s\n", in_data_file);
  if ((data = open_entries(in_data_file)) == NULL)
    {
      fprintf(stderr, "Can't open data file '%s'\n", in_data_file);
      exit(1);
    }
  
  /* set options for data file */
  set_buffer(data, buffer);
      
  init_random(randomize);

  /* do initialization */
  switch (init_type) 
    {
    case IT_RAND:
      ifverbose(2)
	fprintf(stderr, "initializing codes (random)\n");
      codes = randinit_codes(data, topol, neigh, xdim, ydim);
      break;
    case IT_LIN:
      ifverbose(2)
	fprintf(stderr, "initializing codes (linear)\n");
      codes = lininit_codes(data, topol, neigh, xdim, ydim);
      break;
    default:
      fprintf(stderr, "Unknown initialization type %d\n", init_type);
      codes = NULL;
      break;
    }

  close_entries(data);

  if (codes == NULL)
    {
      fprintf(stderr, "initialization failure\n");
      exit(1);
    }

  ifverbose(2)
    fprintf(stderr, "Codebook entries are saved to file %s\n", out_code_file);

  sprintf(comments, "# random seed: %d\n", randomize);
  save_entries_wcomments(codes, out_code_file, comments);

  close_entries(codes);

  return(0);
}

