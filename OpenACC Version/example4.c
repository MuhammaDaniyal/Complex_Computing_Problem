/**********************************************************************
Reads the feature table from "features.txt", copies the features from 
the second frame to those of the third frame, writes the features to 
"feat2.txt", and writes the new feature table to "ft2.txt".  Then the
eighth feature is overwritten with the fifth feature, and the resulting
table is saved to "ft3.txt".
**********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "klt.h"

#ifdef WIN32
int RunExample4()
#else
int main()
#endif
{
  KLT_FeatureList fl;
  KLT_FeatureHistory fh;
  KLT_FeatureTable ft;
  int i;
  char imgPath[100], fnameIn[200], fnameOut[200];

  // Choose image set (set1, set2, etc.)
  const char *setName = "set1";
  sprintf(imgPath, "images/%s/", setName);

  // Read feature table from selected image set
  sprintf(fnameIn, "%sfeatures.txt", imgPath);
  ft = KLTReadFeatureTable(NULL, fnameIn);

  fl = KLTCreateFeatureList(ft->nFeatures);
  KLTExtractFeatureList(fl, ft, 1);

  // Write and read feature list (frame 1)
  sprintf(fnameOut, "%sfeat1.txt", imgPath);
  KLTWriteFeatureList(fl, fnameOut, "%3d");
  KLTReadFeatureList(fl, fnameOut);

  // Store feature list into frame 2 and write new table
  KLTStoreFeatureList(fl, ft, 2);
  sprintf(fnameOut, "%sft2.txt", imgPath);
  KLTWriteFeatureTable(ft, fnameOut, "%3d");

  // Create and print feature history for feature 5
  fh = KLTCreateFeatureHistory(ft->nFrames);
  KLTExtractFeatureHistory(fh, ft, 5);

  printf("The feature history of feature number 5:\n\n");
  for (i = 0; i < fh->nFrames; i++)
    printf("%d: (%5.1f,%5.1f) = %d\n",
           i, fh->feature[i]->x, fh->feature[i]->y,
           fh->feature[i]->val);

  // Store feature 5's history into feature 8 and write to ft3.txt
  KLTStoreFeatureHistory(fh, ft, 8);
  sprintf(fnameOut, "%sft3.txt", imgPath);
  KLTWriteFeatureTable(ft, fnameOut, "%6.1f");

  return 0;
}
