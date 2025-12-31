/**********************************************************************
Finds the 100 best features in an image, tracks these
features to the next image, and replaces the lost features with new
features in the second image.  Saves the feature
locations (before and after tracking) to text files and to PPM files.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32
int RunExample2()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  int nFeatures = 100;
  int ncols, nrows;
  char imgPath[100], fname1[200], fname2[200], featOut[200];

  // Choose image set
  const char *setName = "set1";   // set1, set2, set3, ...
  sprintf(imgPath, "images/%s/", setName);

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);

  // Read two consecutive images
  sprintf(fname1, "%simg0.pgm", imgPath);
  sprintf(fname2, "%simg1.pgm", imgPath);
  img1 = pgmReadFile(fname1, NULL, &ncols, &nrows);
  img2 = pgmReadFile(fname2, NULL, &ncols, &nrows);

  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

  sprintf(featOut, "%sfeat1.ppm", imgPath);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, featOut);
  sprintf(featOut, "%sfeat1.txt", imgPath);
  KLTWriteFeatureList(fl, featOut, "%3d");

  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
  KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);

  sprintf(featOut, "%sfeat2.ppm", imgPath);
  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, featOut);
  sprintf(featOut, "%sfeat2.txt", imgPath);
  KLTWriteFeatureList(fl, featOut, "%3d");

  return 0;
}
