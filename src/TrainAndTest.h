#ifndef TrainAndTest_h
#define TrainAndTest_h

#include "MLCoursework.h"

int train(double** trainingSamples, char* trainingLabels, int numSamples,
          int numFeatures);

char predictLabel(double* sample, int numFeatures);

#endif /* TrainAndTest_h */
