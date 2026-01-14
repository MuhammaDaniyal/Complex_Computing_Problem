/**********************************************************************
Demonstrates manually tweaking the tracking context parameters.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include <stdio.h>

#ifdef WIN32
int RunExample5()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  int nFeatures = 100;
  int ncols, nrows;

  // Choose image set
  const char *setName = "set1";  // "set1", "set2", "set3", ...
  char imgPath[100], fname1[200], fname2[200], out1[200], out2[200];
  sprintf(imgPath, "images/%s/", setName);

  tc = KLTCreateTrackingContext();
  tc->mindist = 20;
  tc->window_width  = 9;
  tc->window_height = 9;
  KLTChangeTCPyramid(tc, 15);
  KLTUpdateTCBorder(tc);
  fl = KLTCreateFeatureList(nFeatures);

  // Read images from selected set
  sprintf(fname1, "%simg0.pgm", imgPath);
  sprintf(fname2, "%simg2.pgm", imgPath);
  img1 = pgmReadFile(fname1, NULL, &ncols, &nrows);
  img2 = pgmReadFile(fname2, NULL, &ncols, &nrows);

  // Feature detection and tracking
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

  sprintf(out1, "%sfeat1b.ppm", imgPath);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, out1);

  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

  sprintf(out2, "%sfeat2b.ppm", imgPath);
  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, out2);

  return 0;
}
