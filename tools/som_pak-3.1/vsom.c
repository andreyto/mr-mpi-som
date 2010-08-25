/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  vsom.c                                                              *
 *  -Visualization Self-Organizing Map                                  *
 *                                                                      *
 *  Version 3.1                                                         *
 *  Date: 7 Apr 1995                                                    *
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
#include <stdlib.h>
#include <math.h>
#include "lvq_pak.h"
#include "som_rout.h"
#include "datafile.h"


static char *usage[] = {
  "vsom - teach self-organizing map\n",
  "Required parameters:\n",
  "  -cin filename         initial codebook file\n",
  "  -din filename         teaching data\n",
  "  -cout filename        output codebook filename\n",
  "  -rlen integer         running length of teaching\n",
  "  -alpha float          initial alpha value\n",
  "  -radius float         initial radius of neighborhood\n",
  "Optional parameters:\n",
  "  -rand integer         seed for random number generator. 0 is current time\n",
  "  -fixed                use fixed points\n",
  "  -weights              use weights\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  "  -alpha_type type      type of alpha decrease, linear (def) or inverse_t.\n",
  
  "  -snapfile filename    snapshot filename\n",
  "  -snapinterval integer interval between snapshots\n",
  NULL};

int main(int argc, char **argv)
{
  char *in_data_file;
  char *in_code_file;
  char *out_code_file;
  char *snapshot_file;
  char *alpha_s, *rand_s;
  struct entries *data = NULL, *codes = NULL;
  int randomize;
  int fixed;
  int weights;
  struct teach_params params;
  long buffer = 0;
  long snapshot_interval;
  struct snapshot_info *snap = NULL;
  int snap_type;
  struct typelist *type_tmp;
  int error = 0;

  data = codes = NULL;

  global_options(argc, argv);
  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }

  in_data_file = extract_parameter(argc, argv, IN_DATA_FILE, ALWAYS);

  in_code_file = extract_parameter(argc, argv, IN_CODE_FILE, ALWAYS);
  out_code_file = extract_parameter(argc, argv, OUT_CODE_FILE, ALWAYS);

  params.length = oatoi(extract_parameter(argc, argv, RUNNING_LENGTH, ALWAYS),
                        1);
  
  params.alpha = atof(extract_parameter(argc, argv, TRAINING_ALPHA, ALWAYS));
	
  params.radius = atof(extract_parameter(argc, argv, TRAINING_RADIUS, ALWAYS));

  rand_s = extract_parameter(argc, argv, RANDOM, OPTION);
  randomize = oatoi(rand_s, 0);

  fixed = (extract_parameter(argc, argv, FIXPOINTS, OPTION2) != NULL);
  weights = (extract_parameter(argc, argv, WEIGHTS, OPTION2) != NULL);

  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);

  alpha_s = extract_parameter(argc, argv, "-alpha_type", OPTION);


  /* snapshots */
  snapshot_file = extract_parameter(argc, argv, "-snapfile", OPTION);
  snapshot_interval = 
    oatoi(extract_parameter(argc, argv, "-snapinterval", OPTION), 0);

  snap_type =
    get_id_by_str(snapshot_list, 
		  extract_parameter(argc, argv, "-snaptype", OPTION));

  
  use_fixed(fixed);
  use_weights(weights);

  label_not_needed(1);

  if (snapshot_interval)
    {
      if (snapshot_file == NULL)
	{
	  snapshot_file = out_code_file;
	  fprintf(stderr, "snapshot file not specified, using '%s'", snapshot_file);
	}
      snap = get_snapshot(snapshot_file, snapshot_interval, snap_type);
      if (snap == NULL)
	exit(1);
    }

  ifverbose(2)
    fprintf(stderr, "Input entries are read from file %s\n", in_data_file);
  data = open_entries(in_data_file);
  if (data == NULL)
    {
      fprintf(stderr, "cant open data file '%s'\n", in_data_file);
      error = 1;
      goto end;
    }

  ifverbose(2)
    fprintf(stderr, "Codebook entries are read from file %s\n", in_code_file);
  codes = open_entries(in_code_file);
  if (codes == NULL)
    {
      fprintf(stderr, "Can't open code file '%s'\n", in_code_file);
      error = 1;
      goto end;
    }

  if (codes->topol < TOPOL_HEXA) {
    fprintf(stderr, "File %s is not a map file\n", in_code_file);
    error = 1;
    goto end;
  }

  if (data->dimension != codes->dimension) {
    fprintf(stderr, "Data and codebook vectors have different dimensions");
    error = 1;
    goto end;
  }

  set_teach_params(&params, codes, data, buffer);
  set_som_params(&params);
  params.snapshot = snap;

  init_random(randomize);

  /* take teaching vectors in random order */
  if (rand_s)
    data->flags.random_order = 1;

  if (alpha_s)
    {
      type_tmp = get_type_by_str(alpha_list, alpha_s);
      if (type_tmp->data == NULL)
	{
	  fprintf(stderr, "Unknown alpha type %s\n", alpha_s);
	  error = 1;
	  goto end;
	}
    }
  else
    type_tmp = get_type_by_id(alpha_list, ALPHA_LINEAR);

  params.alpha_type = type_tmp->id;
  params.alpha_func = type_tmp->data;

  codes = som_training(&params);

  ifverbose(2)
    fprintf(stderr, "Codebook entries are saved to file %s\n", out_code_file);
  save_entries(codes, out_code_file);
 end:

  if (data)
    close_entries(data);
  if (codes)
    close_entries(codes);

  if (snap)
    free_snapshot(snap);

  return(error);
}
