/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  qerror.c                                                            *
 *  -give the quantization error for the data entries                   *
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
#include <math.h>
#include "lvq_pak.h"
#include "som_rout.h"
#include "datafile.h"


static char *usage[] = {
  "qerror - calculate quantization error for the data entries\n",
  "Required parameters:\n",
  "  -cin filename         codebook file\n",
  "  -din filename         test data\n",
  "Optional parameters:\n",
  "  -qetype 1             another way to calculate the error\n",
  "  -radius float         radius of neighborhood for alternative above\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  NULL};

int main(int argc, char **argv)
{
  long nod, buffer;
  float qerror;
  char *in_data_file;
  char *in_code_file;
  struct teach_params teach;
  int qmode;
  struct entries *data, *codes;
  
  global_options(argc, argv);
  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }

  in_data_file = extract_parameter(argc, argv, IN_DATA_FILE, ALWAYS);
  in_code_file = extract_parameter(argc, argv, IN_CODE_FILE, ALWAYS);
  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);
  teach.radius = oatof(extract_parameter(argc, argv, TRAINING_RADIUS, OPTION), 1.0);
  qmode = oatoi(extract_parameter(argc, argv, "-qetype", OPTION), 0);

  label_not_needed(1);

  ifverbose(2)
    fprintf(stdout, "Input entries are read from file %s\n", in_data_file);
  data = open_entries(in_data_file);
  
  if (data == NULL)
    {
      fprintf(stderr, "Can't open datafile: %s\n", in_data_file);
      exit(1);
    }

  ifverbose(2)
    fprintf(stdout, "Codebook entries are read from file %s\n", in_code_file);
  codes = open_entries(in_code_file);
  
  if (codes == NULL)
    {
      fprintf(stderr, "Can't open code file '%s'\n", in_code_file);
      close_entries(data);
      exit(1);
    }

  if (codes->topol < TOPOL_HEXA) {
    fprintf(stderr, "File %s is not a map file\n", in_code_file);
    close_entries(data);
    close_entries(codes);
    exit(1);
  }

  if (data->dimension != codes->dimension) {
    fprintf(stderr, "Data and codebook vectors have different dimensions (%d != %d)",
	    data->dimension, codes->dimension);
    close_entries(data);
    close_entries(codes);
    exit(1);
  }

  set_teach_params(&teach, codes, data, buffer);
  set_som_params(&teach);

  if (qmode > 0)
    qerror = find_qerror2(&teach);
  else
    qerror = find_qerror(&teach);

  nod = teach.data->num_entries;

  ifverbose(1)
    fprintf(stdout, "Quantization error of %s with map %s is %f per sample (%d samples)\n",
	    in_data_file, in_code_file, qerror / (float) nod, nod);
  else
    fprintf(stdout, "%f\n", qerror / (float) nod);

  return(0);
}
