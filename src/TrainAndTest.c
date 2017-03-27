#include <stdio.h>

#include "TrainAndTest.h"

static int totalSamples = 0;
static int totalFeatures = 0;
static int totalLabels = 0;

static double model[NUM_SAMPLES][NUM_FEATURES] = { 0 };
static char labels[NUM_SAMPLES] = { 0 };

int train(double** trainingSamples, char* trainingLabels, int numSamples,
          int numFeatures) {
  int sample = 0;
  int feature = 0;

  // Sanity checking.
  if (numFeatures > NUM_FEATURES || numSamples > NUM_TRAINING_SAMPLES) {
    printf("Too many features and/or samples provided\n");
    return 0;
  }

  // Store the labels and the feature values.
  totalSamples = numSamples;
  totalFeatures = numFeatures;
  totalLabels = numSamples;

  for (sample = 0; sample < numSamples; sample++) {
    labels[sample] = trainingLabels[sample];

    for (feature = 0; feature < numFeatures; feature++) {
      model[sample][feature] = trainingSamples[sample][feature];
    }
  }

  return 1;
}

char predictLabel(double* sampleData, int numFeatures) {
  int sample;
  int feature;
  int difference;
  int distance;
  int closestDistance;
  int closestSample;

  // Sanity checking.
  if (numFeatures != totalFeatures) {
    printf("Training data feature count not equal to testing data\n");
    return 0;
  }

  if (totalSamples <= 0) {
    printf("kNN has not yet been trained\n");
    return 0;
  }

  // Calculate the distance for the first sample and set it.
  distance = 0;

  for (feature = 0; feature < numFeatures; feature++) {
    difference = model[0][feature] - sampleData[feature];
    distance += difference * difference;
  }

  closestDistance = distance;
  closestSample = 0;

  // Calculate the distance for each subsequent sample and update if its closer.
  for (sample = 1; sample < totalSamples; sample++) {
    distance = 0;

    for (feature = 0; feature < numFeatures; feature++) {
      difference = model[sample][feature] - sampleData[feature];
      distance += difference * difference;
    }

    if (distance < closestDistance) {
      closestDistance = distance;
      closestSample = sample;
    }
  }

  return labels[closestSample];
}
