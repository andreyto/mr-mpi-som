/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  vcal.c                                                              *
 *  - sets the labels of entries by the majority voting                 *
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
#include "lvq_pak.h"
#include "datafile.h"
#include "som_rout.h"

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

static char *usage[] = {
  "vcal - sets the labels of entries by the majority voting\n",
  "Required parameters:\n",
  "  -cin filename         codebook file\n",
  "  -din filename         labeling data\n",
  "  -cout filename        labeled output codebook filename\n",
  "Optional parameters:\n",
  "  -numlabs integer      maximum number of labels to assign to a codebook\n",
  "                        vector. Default is 1, 0 gives all.\n",
  "  -buffer integer       buffered reading of data, integer lines at a time\n",
  NULL};

struct entries *find_labels(struct teach_params *teach, int numlabs)

{
  long noc, nol, index;
  int i, labs;
  int datalabel, ind;
  struct data_entry *codetmp, *datatmp;
  WINNER_FUNCTION *winner = teach->winner;
  struct entries *data = teach->data;
  struct entries *codes = teach->codes;
  struct winner_info win_info;
  struct hitlist **hits = NULL;
  struct hit_entry *hit;
  int showmeter = 0;
  eptr p;

  if (numlabs < 0)
    numlabs = 0;

  data = teach->data;
  codes = teach->codes;

  if (rewind_entries(codes, &p) == NULL)
    return NULL;
  
  noc = codes->num_entries;

  /* allocate a hitlist for all codebook units */
  hits = calloc(noc, sizeof(struct hitlist *));
  if (hits == NULL)
    return NULL;

  for (i = 0; i < noc; i++)
    {
      hits[i] = new_hitlist();
      if (hits[i] == NULL)
	{
	  fprintf(stderr, "Can't get hitlist[%d]\n", i);
	  /* free allocated hitlists */
	  while (--i >= 0)
	    free_hitlist(hits[i]);
	  free(hits);
	  return NULL;
	}
    }

  /* Scan all data entries */

  if ((datatmp = rewind_entries(data, &p)) == NULL)
    return NULL;

  /* show progress meter only if number of data vectors is known (when
     not using buffered input) */

  showmeter = data->flags.totlen_known;
  if (showmeter)
    nol = data->num_entries;
  else
    nol = 0;
  ind = 0;

  while (datatmp != NULL) {
    datalabel = get_entry_label(datatmp);

    if (winner(codes, datatmp, &win_info, 1) == 0)
      goto skip_hit; /* winner not found -> assume that all components
			of sample vector were masked off -> skip this
			sample */

    /* add a hit in the winning unit's hitlist for the class of the
       sample. Ignores samples with no class (= empty label) */

    index = win_info.index;
    if (datalabel != LABEL_EMPTY)
      add_hit(hits[index], datalabel);

  skip_hit:
    /* Take the next data entry */
    datatmp = next_entry(&p);
    ind++;

    if (showmeter)
      ifverbose(1)
	mprint(nol--);
  }

  ifverbose(1)
    {
      mprint(0);
      fprintf(stderr, "\n");
    }

  /* Set the label of codebook entries according the
     selections. Numlabs tells how many labels at maximum to assign to
     a certain codebook vector. 0 means all */

  codetmp = rewind_entries(codes, &p);
  index = 0;

  while (codetmp != NULL) {

    if (numlabs == 0)
      labs = hits[index]->entries;
    else
      labs = min(hits[index]->entries, numlabs);

    /* remove previous labels from codebook vector */
    clear_entry_labels(codetmp);

    for (i = 0, hit = hits[index]->head; i < labs; i++, hit = hit->next)
      add_entry_label(codetmp, hit->label);

    free_hitlist(hits[index]);
    hits[index] = NULL;

    codetmp = next_entry(&p);
    index++;
  }

  free(hits);

  return(codes);
}

int main(int argc, char **argv)
{
  char *in_data_file;
  char *in_code_file;
  char *out_code_file;
  struct entries *data, *codes;
  struct teach_params params;
  int retcode = 0;
  int numlabels = 1;
  long buffer;

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
  buffer = oatoi(extract_parameter(argc, argv, "-buffer", OPTION), 0);
  numlabels = oatoi(extract_parameter(argc, argv, "-numlabs", OPTION), 1);
  
  ifverbose(2)
    fprintf(stderr, "Data entries are read from file %s\n", in_data_file);
  data = open_entries(in_data_file);
  if (data == NULL)
    {
      fprintf(stderr, "cant open data file '%s'\n", in_data_file);
      exit(1);
    }

  label_not_needed(1);

  ifverbose(2)
    fprintf(stderr, "Codebook entries are read from file %s\n", in_code_file);
  codes = open_entries(in_code_file);
  if (codes == NULL)
    {
      fprintf(stderr, "can't open code file '%s'\n", in_code_file);
      close_entries(data);
      exit(1);
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

  if (find_labels(&params, numlabels) == NULL)
    {
      fprintf(stderr, "find_labels failed\n");
      retcode = 1;
      goto end;
    }

  ifverbose(2)
    fprintf(stderr, "Codebook entries are saved to file %s\n", out_code_file);
  save_entries(codes, out_code_file);

 end:

  close_entries(data);
  close_entries(codes);

  return(retcode);
}
