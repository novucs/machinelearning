#include <math.h>
#include <stdio.h>
#include "TrainAndTest.h"

// Totals
#define HIDDEN_LAYER_SIZE 3
#define OUTPUT_LAYER_SIZE 1

static int totalSamples = 0;
static int totalFeatures = NUM_FEATURES;

// Models
static double model[NUM_TRAINING_SAMPLES][NUM_FEATURES] = { 0 };
static double modelNormals[NUM_TRAINING_SAMPLES][NUM_FEATURES] = { 0 };
static double featureCaps[NUM_FEATURES] = { 0 };

// Labels
static char labels[NUM_TRAINING_SAMPLES] = { 0 };
static double labelNormals[NUM_TRAINING_SAMPLES] = { 0 };
static double labelCap;

// Weights
static double weights1[NUM_FEATURES][HIDDEN_LAYER_SIZE] = { 0 };
static double weights2[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE] = { 0 };

// Neurons
static double activation2[NUM_TRAINING_SAMPLES][HIDDEN_LAYER_SIZE] = { 0 };
static double activity2[NUM_TRAINING_SAMPLES][HIDDEN_LAYER_SIZE] = { 0 };
static double activation3[NUM_TRAINING_SAMPLES][OUTPUT_LAYER_SIZE] = { 0 };

// Estimation
double yHat[NUM_TRAINING_SAMPLES][OUTPUT_LAYER_SIZE] = { 0 };

// Normalisation
void normalise() {
  int sample;
  int feature;
  int sum;

  for (sample = 0; sample < totalSamples; sample++) {
    if (labels[sample] > labelCap) {
      labelCap = labels[sample];
    }

    for (feature = 0; feature < totalFeatures; feature++) {
      if (model[sample][feature] > featureCaps[feature]) {
        featureCaps[feature] = model[sample][feature];
      }
    }
  }

  for (sample = 0; sample < totalSamples; sample++) {
    labelNormals[sample] = (labels[sample] - 'a') / (labelCap - 'a');

    for (feature = 0; feature < totalFeatures; feature++) {
      sum = model[sample][feature] / featureCaps[feature];
      modelNormals[sample][feature] = sum;
    }
  }
}

// Forward propagation
void forward() {
  int sample;
  int feature;
  int hidden_layer;
  int output_layer;
  int sum;

  // Dot product of model and weights1
  for (sample = 0; sample < totalSamples; sample++) {
    for (feature = 0; feature < totalFeatures; feature++) {
      for (hidden_layer = 0; hidden_layer < HIDDEN_LAYER_SIZE; hidden_layer++) {
        sum = modelNormals[sample][feature] * weights1[feature][hidden_layer];
        activation2[sample][hidden_layer] += sum;
      }
    }
  }

  // Sigmoid activation2 for activity2
  for (sample = 0; sample < totalSamples; sample++) {
    for (hidden_layer = 0; hidden_layer < HIDDEN_LAYER_SIZE; hidden_layer++) {
      sum = 1 / (1 + pow(M_E, -(activation2[sample][hidden_layer])));
      activity2[sample][hidden_layer] = sum;
    }
  }

  // Dot product of activity2 and weights2
  for (sample = 0; sample < totalSamples; sample++) {
    for (hidden_layer = 0; hidden_layer < HIDDEN_LAYER_SIZE; hidden_layer++) {
      for (output_layer = 0; output_layer < OUTPUT_LAYER_SIZE; output_layer++) {
        sum = activity2[sample][hidden_layer] * weights2[hidden_layer][output_layer];
        activation3[sample][output_layer] += sum;
      }
    }
  }

  // Sigmoid activation3 for yHat
  for (sample = 0; sample < totalSamples; sample++) {
    for (output_layer = 0; output_layer < OUTPUT_LAYER_SIZE; output_layer++) {
      sum = 1 / (1 + pow(M_E, -(activation3[sample][output_layer])));
      yHat[sample][output_layer] = sum;
    }
  }
}

int train(double** trainingSamples, char* trainingLabels, int numSamples,
          int numFeatures) {
  int returnval = 1;
  int sample;
  int feature;

  // Sanity checking.
  if (numFeatures > NUM_FEATURES || numSamples > NUM_TRAINING_SAMPLES) {
    fprintf(stdout, "error: called train with data set larger than spaced allocated to store it");
    returnval = 0;
  }

  if (returnval == 1) {
    // Store the labels and the feature values.
    totalSamples = numSamples;
    totalFeatures = numFeatures;

    for (sample = 0; sample < numSamples; sample++) {
      labels[sample] = trainingLabels[sample];

      for (feature = 0; feature < numFeatures; feature++) {
        model[sample][feature] = trainingSamples[sample][feature];
      }
    }

    printf("Data stored locally\n");
  }

  normalise();
  forward();

  return returnval;
}

char predictLabel(double* sample, int numFeatures) {
  // This is a silly trivial test function obviously you need to replace this
  // with something that uses the model you built in your train() function.
  char prediction = 'c';
  return prediction;
}
