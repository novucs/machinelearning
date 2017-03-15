#include <math.h>
#include <stdio.h>
#include "TrainAndTest.h"

// Totals
#define HIDDEN_LAYERS 3

static int totalSamples = 0;
static int totalFeatures = 0;

// Models
static double model[NUM_SAMPLES][NUM_FEATURES] = { 0 };
static double modelNormals[NUM_SAMPLES][NUM_FEATURES] = { 0 };
static double featureCaps[NUM_FEATURES] = { 0 };

// Labels
static char labels[NUM_SAMPLES] = { 0 };
static double labelNormals[NUM_SAMPLES] = { 0 };
static char labelCap;

static double predictions[NUM_SAMPLES] = { 0 };
static double weights[NUM_FEATURES] = { 0 };

// Helpers
static int sample = 0;
static int feature = 0;

// Normalisation
void normalise() {
  // Get label and feature max values
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

  // Divide each label and feature by either max values
  for (sample = 0; sample < totalSamples; sample++) {
    labelNormals[sample] = (labels[sample] - 'a') / (labelCap - 'a');

    for (feature = 0; feature < totalFeatures; feature++) {
      modelNormals[sample][feature] = model[sample][feature] / featureCaps[feature];
    }
  }
}

// Forward propagation
void learn() {
  int i;
  int error = 1;
  int maxIterations = 25;
  double alpha = 0.025;

  for (i = 0; error == 1 && i < maxIterations; i++) {
    error = 0;

    for (sample = 0; sample <= totalSamples - 1; sample++) {
      predictions[sample] = 0;

      for (feature = 0; feature < totalFeatures; feature++) {
        predictions[sample] += weights[feature] * modelNormals[sample][feature];
      }

      if (predictions[sample] == labelNormals[sample]) {
        continue;
      }

      error = 1;

      for (feature = 0; feature < totalFeatures; feature++) {
        weights[feature] += alpha * (labelNormals[sample] - predictions[sample]) * modelNormals[sample][feature];
      }
    }
  }
}

void predict() {
  for (sample = 0; sample <= totalSamples - 1; sample++) {
    predictions[sample] = 0;

    for (feature = 0; feature < totalFeatures; feature++) {
      predictions[sample] += weights[feature] * modelNormals[sample][feature];
    }
  }
}

int train(double** trainingSamples, char* trainingLabels, int numSamples,
          int numFeatures) {
  // Sanity checking.
  if (numFeatures > NUM_FEATURES || numSamples > NUM_TRAINING_SAMPLES) {
    printf("Too many features and/or samples provided\n");
    return 0;
  }

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

  weights[0] = 0.2;
  weights[1] = 0.4;
  weights[2] = 0.6;
  weights[3] = 0.8;

  normalise();
  learn();

  return 1;
}

char predictLabel(double* sample, int numFeatures) {
  totalSamples = 1;
  totalFeatures = numFeatures;

  for (feature = 0; feature < totalFeatures; feature++) {
    model[0][feature] = sample[feature];
  }

  normalise();
  predict();

  char prediction;

  if (predictions[0] <= 0.25) {
    prediction = 'a';
  } else if (predictions[0] <= 0.75) {
    prediction = 'b';
  } else {
    prediction = 'c';
  }

  return prediction;
}
