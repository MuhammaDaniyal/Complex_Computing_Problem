/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "pnmio.h"
#include "klt.h"

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main(int argc, char *argv[])
#endif
{
  //KLT_ResetPerformanceStats();
  int nFrames = 10;

  if(argc < 3)
  {
    printf("No arguments provided! Using default image set -> set1\n");
    argv[1] = "set1";
  }
  nFrames = atoi(argv[2]);
  //nFrames = 10;  // Currently keeping same number of frames for all
  
  unsigned char *img1, *img2;
  char fnamein[200], fnameout[200], imgPath[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150;
  int ncols, nrows;
  int i;

  /* Choose image set */
  char setName[100];   // set1, set2, set3, ...
  strcpy(setName, argv[1]);
  sprintf(imgPath, "images/%s/", setName);
  printf("\n--- Using image set: %s ---", setName);

    tc = KLTCreateTrackingContext();
    fl = KLTCreateFeatureList(nFeatures);
    ft = KLTCreateFeatureTable(nFrames, nFeatures);
    tc->sequentialMode = TRUE;
    tc->writeInternalImages = FALSE;
    tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
    // Reading first image
    sprintf(fnamein, "%simg0.pgm", imgPath);
    img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
    img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));

    // Select and store good features
    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
    KLTStoreFeatureList(fl, ft, 0);
    sprintf(fnameout, "%sfeat0.ppm", imgPath);
    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);


    // ========== START TIMER ==========
    struct timeval start, end;
    gettimeofday(&start, NULL);
    // =================================

  // Track features across subsequent frames
  for (i = 1 ; i < nFrames ; i++)  {
    sprintf(fnamein, "%simg%d.pgm", imgPath, i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "%sfeat%d.ppm", imgPath, i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }

    // ========== STOP TIMER & CALCULATE ==========
    gettimeofday(&end, NULL);
    
    double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms

    // Write results
    KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
    KLTWriteFeatureTable(ft, "features.ft", NULL);

    // Free memory
    KLTFreeFeatureTable(ft);
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);
    free(img1);
    free(img2);
    

  printf("\n");
  printf("╔════════════════════════════════════════════╗\n");
  printf("║   TOTAL PROGRAM EXECUTION TIME (CPU)       ║\n");
  printf("╠════════════════════════════════════════════╣\n");
  printf("║   Time: %10.2f ms                      ║\n", elapsed_time);
  printf("║   Time: %10.3f seconds                 ║\n", elapsed_time / 1000.0);
  printf("╚════════════════════════════════════════════╝\n");

  //KLT_PrintPerformanceStats();

  return 0;
}