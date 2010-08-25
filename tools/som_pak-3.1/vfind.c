/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  vfind.c                                                             *
 *  - find best map with many trials                                    *
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
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "lvq_pak.h"
#include "som_rout.h"
#include "datafile.h"

#ifndef max
#define max(x,y) ((x)<(y) ? (y):(x))
#endif


/*---------------------------------------------------------------------------*/

void print_description(void)
{
  printf("This program will repeatedly run the initialization, training\n");
  printf("and testing cycle for Self-Organizing Map algorithm.\n");
  printf("\n");
  printf("In the following the training file name, the test file name\n");
  printf("(that can be the same) and the map save file name are asked.\n");
  printf("After them the type of map topology is asked, as well as\n");
  printf("the type of neighborhood function. The x- and y-dimension\n");
  printf("of the map should be integers and prefereably x-dimension\n");
  printf("should be larger than y-dimension.\n");
  printf("\n");
  printf("The training is done in two parts. First an ordering phase\n");
  printf("that is usually shorter than the following converging phase.\n");
  printf("The number of training cycles, the training rates and\n");
  printf("the radius of the adaptation area are asked separately for\n");
  printf("both phases. The fixed point qualifiers and weighting qualifiers\n");
  printf("are used if the corresponding parameters were given.\n");
  printf("\n");
  printf("The quantization error is computed for each map and\n");
  printf("the best map (smallest quantization error) is saved to\n");
  printf("the given file. If the verbose parameter allows the quantization\n");
  printf("error is given for each separate trial.\n");
  printf("\n");
  printf("After the answers have been given the training begins\n");
  printf("and depending on the size of problem it may take a long time.\n");
  printf("\n");
}

long get_int(char *ch, int no)
{
  int len = 100;
  char *tstr;
  char str[100];

  printf("%s: ", ch);

  tstr = fgets(str, len, stdin);
  if (tstr == NULL)
    return(no);

  return(oatoi(str, no));
}

float get_float(char *ch, float no)
{
  int len = 100;
  char *tstr;
  char str[100];

  printf("%s: ", ch);

  tstr = fgets(str, len, stdin);
  if (tstr == NULL)
    return(no);

  return(atof(str));
}

char *get_str(char *ch)
{
  int len = 100;
  char *tstr, *tmp;
  char str[100];

  printf("%s: ", ch);

  tstr = fgets(str, len, stdin);
  if (tstr == NULL) {
    printf("Can't read required data\n");
    exit(1);
  }

  tstr = ostrdup(str);
  tmp = strchr(tstr, ' ');
  if (tmp != (char) NULL)
    tmp[0] = '\0';
  tmp = strchr(tstr, '\n');
  if (tmp != (char) NULL)
    tmp[0] = '\0';

  return(tstr);
}

int main(int argc, char **argv)
{
  int not, bnot, error;
  int xdim, ydim;
  int topol, neigh;
  float alpha1, radius1;
  int fixed, weights;
  float alpha2, radius2;
  float qerror, qerrorb;
  char *in_data_file, *in_test_file, *out_code_file, *alpha_s;
  struct entries *data = NULL;
  struct entries *testdata = NULL;
  struct entries *codes = NULL;
  struct entries *codess = NULL;
  struct entries *tmp;
  struct teach_params params;
  long buffer, length1, length2, noc, nod;
  int qmode;
  struct typelist *type_tmp;

  error = 0;
  global_options(argc, argv);
  print_description();

  not = get_int("Give the number of trials", 0);
  in_data_file = get_str("Give the input data file name");
  in_test_file = get_str("Give the input test file name");
  out_code_file = get_str("Give the output map file name");

  topol = topol_type(get_str("Give the topology type"));
  if (topol == TOPOL_UNKNOWN) {
    ifverbose(2)
      fprintf(stderr, "Unknown topology type, using hexagonal\n");
    topol = TOPOL_HEXA;
  }
  neigh = neigh_type(get_str("Give the neighborhood type"));
  if (neigh == NEIGH_UNKNOWN) {
    ifverbose(2)
      fprintf(stderr, "Unknown neighborhood type, using bubble\n");
    neigh = NEIGH_BUBBLE;
  }

  xdim = get_int("Give the x-dimension", 0);
  ydim = get_int("Give the y-dimension", 0);

  length1 = get_int("Give the training length of first part", 0);
  alpha1 = get_float("Give the training rate of first part", 0.0);
  radius1 = get_float("Give the radius in first part", 0.0);
  length2 = get_int("Give the training length of second part", 0);
  alpha2 = get_float("Give the training rate of second part", 0.0);
  radius2 = get_float("Give the radius in second part", 0.0);

  printf("\n");

  fixed = (int) oatoi(extract_parameter(argc, argv, FIXPOINTS, OPTION), 0);
  weights = (int) oatoi(extract_parameter(argc, argv, WEIGHTS, OPTION), 0);
  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);
  alpha_s = extract_parameter(argc, argv, "-alpha_type", OPTION);
  qmode = oatoi(extract_parameter(argc, argv, "-qetype", OPTION), 0);

  use_fixed(fixed);
  use_weights(weights);

  label_not_needed(1);

  ifverbose(2)
    fprintf(stderr, "Input entries are read from file %s\n", in_data_file);
  data = open_entries(in_data_file);
  if (data == NULL)
    {
      fprintf(stderr, "Can't open data file '%s'\n", in_data_file);
      error = 1;
      goto end;
    }
  set_buffer(data, buffer);

  ifverbose(2)
    fprintf(stderr, "Test entries are read from file %s\n", in_test_file);
  testdata = open_entries(in_test_file);
  if (testdata == NULL)
    {
      fprintf(stderr, "Can't open test data file '%s'\n", in_test_file);
      error = 1;
      goto end;
    }
  set_buffer(testdata, buffer);

  noc = xdim * ydim;
  if (noc <= 0) 
    {
      fprintf(stderr, "Dimensions of map (%d %d) are incorrect\n", xdim, ydim);
      error = 1;
      goto end;
    }
  if (xdim < 0) 
    {
      fprintf(stderr, "Dimensions of map (%d %d) are incorrect\n", xdim, ydim);
      error = 1;
      goto end;
    }

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


  codess = NULL;
  qerrorb = FLT_MAX;
  bnot = 0;
  while (not) {
    init_random(not);

    ifverbose(2)
      fprintf(stderr, "Initializing codebook\n");
    
    codes = randinit_codes(data, topol, neigh, xdim, ydim);
    if (codes == NULL)
      {
	fprintf(stderr, "Error initializing random codebook, aborting\n");
	error = 1;
	goto end;
      }

    set_teach_params(&params, codes, NULL, 0);
    set_som_params(&params);
    params.data = data;
    
    params.length = length1;
    params.alpha = alpha1;
    params.radius = radius1;

    ifverbose(2)
      fprintf(stderr, "Training map, first part, rlen: %ld alpha: %f\n", 
	      params.length, params.alpha);
    codes = som_training(&params);

    params.length = length2;
    params.alpha = alpha2;
    params.radius = radius2;
    ifverbose(2)
      fprintf(stderr, "Training map, second part, rlen: %ld alpha: %f\n", 
	      params.length, params.alpha);
    codes = som_training(&params);

    params.data = testdata;
    ifverbose(2)
      fprintf(stderr, "Calculating quantization error\n");

    if (qmode > 0)
      qerror = find_qerror2(&params);
    else
      qerror = find_qerror(&params);
    nod = testdata->num_entries;

    if (qerror < qerrorb) {
      qerrorb = qerror;
      bnot = not;

      tmp = codess;
      codess = codes;
      codes = tmp;
    }

    close_entries(codes);
    codes = NULL;
    ifverbose(1)
      fprintf(stderr, "%3d: %f\n", not, qerror/(float) nod);
    not--;
  }

  if (codess != NULL) 
    {
      ifverbose(2)
	fprintf(stdout, "Codebook entries are saved to file %s\n", out_code_file);
      save_entries(codess, out_code_file);
      ifverbose(1)
	fprintf(stdout, "Smallest error with random seed %3d: %f\n",
                bnot, qerrorb/(float) nod);
    }

 end:
  if (codess)
    close_entries(codess);
  if (data)
    close_entries(data);
  if (testdata)
    close_entries(testdata);
  
  return(error);
}



