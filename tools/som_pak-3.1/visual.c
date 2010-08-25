/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  visual.c                                                            *
 *  -finds out the best matching unit in map and produces a file        *
 *   containing a line for each input sample, in each line the          *
 *   coordinates of the best match and the quantization error           *
 *   are given as well as the label if there is any                     *
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
  "visual - find best matching unit for each data sample\n", 
  "Required parameters:\n",
  "  -cin filename         codebook file\n",
  "  -din filename         input data\n",
  "  -dout filename        output filename\n",
  "Optional parameters:\n",
  "  -noskip               do not skip data vectors that have all components\n",
  "                        masked off\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  NULL};



struct entries *compute_visual_data(struct teach_params *teach, 
				    char *out_file_name)
{
  int length_known;
  struct data_entry *datatmp;
  struct entries *data = teach->data;
  struct entries *codes = teach->codes;
  struct entries *fake_data;
  struct data_entry *fake_entry;
  struct file_info *fi;
  WINNER_FUNCTION *winner = teach->winner;
  struct winner_info win_info;
  int emptylab = LABEL_EMPTY;
  long index, nod, bpos;
  eptr p;

  emptylab = find_conv_to_ind("EMPTY_LINE");

  /* initialize fake entries table */
  fake_data = alloc_entries();
  if (fake_data == NULL)
    {
      fprintf(stderr, "Can't allocate memory for entries\n");
      return NULL;
    }

  fake_data->dimension = 3;
  fake_data->topol = codes->topol;
  fake_data->neigh = codes->neigh;
  fake_data->xdim = codes->xdim;
  fake_data->ydim = codes->ydim;

  fake_entry = alloc_entry(fake_data);
  if (fake_entry == NULL)
    {
      fprintf(stderr, "Can't allocate memory for fake entry\n");
      free_entries(fake_data);
      return NULL;
    }
  fake_data->entries = fake_entry;

  /* open output file */
  fi = open_file(out_file_name, "w");
  if (fi == NULL)
    {
      fprintf(stderr, "can't open file for output: '%s'\n", out_file_name);
      free_entries(fake_data);
      return NULL;
    }

  fake_data->fi = fi;
  /* write header of output file */
  write_header(fi, fake_data);

  /* Scan all input entries. */
  datatmp = rewind_entries(data, &p);

  if ((length_known = data->flags.totlen_known))
    nod = data->num_entries;
  else
    nod = 0;

  while (datatmp != NULL) {

    /* bpos = winner(codes, datatmp); */
    if (winner(codes, datatmp, &win_info, 1) == 0)
      {
	/* empty sample */
	/* Save the classification and coordinates */
	set_entry_label(fake_entry, emptylab); /* labels */
	fake_entry->points[0] = -1;
	fake_entry->points[1] = -1;
	/* And the quantization error */
	fake_entry->points[2] = -1.0;
      }
    else
      {
	bpos = win_info.index;
	index = get_entry_label(win_info.winner);
	
	/* Save the classification and coordinates */
	copy_entry_labels(fake_entry, win_info.winner); /* labels */
	fake_entry->points[0] = bpos % codes->xdim;
	fake_entry->points[1] = bpos / codes->xdim;
	/* And the quantization error */
	fake_entry->points[2] = sqrt(win_info.diff);
      }
    /* write new entry */
    write_entry(fi, fake_data, fake_entry);

    /* Take the next input entry */
    datatmp = next_entry(&p);

    if (length_known)
      ifverbose(1)
	mprint((long) nod--);
  }

  if (length_known)
    ifverbose(1)
      {
	mprint((long) 0);
	fprintf(stderr, "\n");
      }

  free_entries(fake_data);

  return(data);
}


int main(int argc, char **argv)
{
  char *in_data_file;
  char *in_code_file;
  char *out_data_file;
  struct entries *data, *codes;
  struct teach_params params;
  int retcode, noskip;
  long buffer;

  data = codes = NULL;
  retcode = 0;

  global_options(argc, argv);
  if (extract_parameter(argc, argv, "-help", OPTION2))
    {
      printhelp();
      exit(0);
    }

  in_data_file = extract_parameter(argc, argv, IN_DATA_FILE, ALWAYS);
  in_code_file = extract_parameter(argc, argv, IN_CODE_FILE, ALWAYS);
  out_data_file = extract_parameter(argc, argv, OUT_DATA_FILE, ALWAYS);
  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);
  noskip = (extract_parameter(argc, argv, "-noskip", OPTION2) != NULL);

  label_not_needed(1);

  data = open_entries(in_data_file);
  if (data == NULL)
    {
      fprintf(stderr, "Can't open data file '%s'\n", in_data_file);
      retcode = 1;
      goto end;
    }
  
  ifverbose(2)
    fprintf(stderr, "Codebook entries are read from file %s\n", in_code_file);
  codes = open_entries(in_code_file);
  if (codes == NULL)
    {
      fprintf(stderr, "can't open code file '%s'\n", in_code_file);
      retcode = 1;
      goto end;
    }

  if (codes->topol < TOPOL_HEXA) {
    fprintf(stderr, "File %s is not a map file\n", in_code_file);
    retcode = 1;
    goto end;
  }

  if (data->dimension != codes->dimension) {
    fprintf(stderr, "Data and codebook vectors have different dimensions");
    retcode = 1;
    goto end;
  }

  set_teach_params(&params, codes, data, buffer);
  set_som_params(&params);
  /* does not skip empty samples if wanted */
  if (noskip)
    data->flags.skip_empty = 0;
  data = compute_visual_data(&params, out_data_file);

 end:
  if (data)
    close_entries(data);
  if (codes)
    close_entries(codes);
  return(retcode);
}
