/**********************************************************************
Finds the 100 best features in an image, and tracks these
features to the next image.  Saves the feature
locations (before and after tracking) to text files and to PPM files, 
and prints the features to the screen.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32
int RunExample1()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  int nFeatures = 100;
  int ncols, nrows;
  int i;
  char imgPath[100], fname1[200], fname2[200], featOut[200];

  // Choose image set
  const char *setName = "set1";   // set1, set2, set3, ...
  sprintf(imgPath, "images/%s/", setName);

  // Initialize tracking context and feature list
  tc = KLTCreateTrackingContext();
  KLTPrintTrackingContext(tc);
  fl = KLTCreateFeatureList(nFeatures);

  // Read two images from chosen set
  sprintf(fname1, "%simg0.pgm", imgPath);
  sprintf(fname2, "%simg1.pgm", imgPath);
  img1 = pgmReadFile(fname1, NULL, &ncols, &nrows);
  img2 = pgmReadFile(fname2, NULL, &ncols, &nrows);

  // Select good features in first image
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

  printf("\nIn first image:\n");
  for (i = 0; i < fl->nFeatures; i++) {
    printf("Feature #%d:  (%f, %f) with value of %d\n",
           i, fl->feature[i]->x, fl->feature[i]->y,
           fl->feature[i]->val);
  }

  // Write first image features
  sprintf(featOut, "%sfeat1.ppm", imgPath);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, featOut);
  sprintf(featOut, "%sfeat1.txt", imgPath);
  KLTWriteFeatureList(fl, featOut, "%3d");

  // Track features to second image
  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

  printf("\nIn second image:\n");
  for (i = 0; i < fl->nFeatures; i++) {
    printf("Feature #%d:  (%f, %f) with value of %d\n",
           i, fl->feature[i]->x, fl->feature[i]->y,
           fl->feature[i]->val);
  }

  // Write tracked features
  sprintf(featOut, "%sfeat2.ppm", imgPath);
  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, featOut);
  sprintf(featOut, "%sfeat2.fl", imgPath);
  KLTWriteFeatureList(fl, featOut, NULL);     /* binary file */
  sprintf(featOut, "%sfeat2.txt", imgPath);
  KLTWriteFeatureList(fl, featOut, "%5.1f");  /* text file */

  return 0;
}
