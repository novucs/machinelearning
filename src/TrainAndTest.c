#include <stdio.h>

#include "TrainAndTest.h"

#define HIDDEN_LAYERS 3
#define TOLERANCE 0.01

static int totalSamples = 0;
static int totalFeatures = 0;
static int totalLabels = 0;

// Models
static double model[NUM_SAMPLES][NUM_FEATURES] = { 0 };
static double modelNormals[NUM_SAMPLES][NUM_FEATURES] = { 0 };
static double featureCaps[NUM_FEATURES] = { 0 };

// Labels
static char labels[NUM_SAMPLES] = { 0 };
static double labelNormals[NUM_SAMPLES] = { 0 };
static char highestLabel;
static char lowestLabel;

static double predictions[NUM_SAMPLES] = { 0 };
static double weights[NUM_FEATURES] = { 0 };

// Helpers
static int sample = 0;
static int feature = 0;
static int label = 0;

// Normalisation
void normalise() {
  // Divide each label and feature by either max values
  for (sample = 0; sample < totalSamples; sample++) {
    labelNormals[sample] = (double) (labels[sample] - lowestLabel) / (highestLabel - lowestLabel);

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
  double alpha = totalFeatures / (double) 1000;

  for (i = 0; error == 1 && i < maxIterations; i++) {
    error = 0;

    for (sample = 0; sample <= totalSamples - 1; sample++) {
      predictions[sample] = 0;

      for (feature = 0; feature < totalFeatures; feature++) {
        predictions[sample] += weights[feature] * modelNormals[sample][feature];
      }

      if (predictions[sample] > labelNormals[sample] - TOLERANCE &&
          predictions[sample] < labelNormals[sample] + TOLERANCE) {
        continue;
      }

      error = 1;

      for (feature = 0; feature < totalFeatures; feature++) {
        if (labelNormals[sample] - predictions[sample] < 0) {
          weights[feature] -= alpha * modelNormals[sample][feature];
        } else {
          weights[feature] += alpha * modelNormals[sample][feature];
        }
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
  totalLabels = numSamples;

  for (sample = 0; sample < numSamples; sample++) {
    labels[sample] = trainingLabels[sample];

    for (feature = 0; feature < numFeatures; feature++) {
      model[sample][feature] = trainingSamples[sample][feature];
    }
  }

  printf("Data stored locally\n");

  for (feature = 0; feature < NUM_FEATURES; feature++) {
    weights[feature] = ((double) feature / NUM_FEATURES) + 0.1;
  }

  // Get label and feature max values
  for (sample = 0; sample < totalSamples; sample++) {
    if (sample == 0 || labels[sample] > highestLabel) {
      highestLabel = labels[sample];
    }

    if (sample == 0 || labels[sample] < lowestLabel) {
      lowestLabel = labels[sample];
    }

    for (feature = 0; feature < totalFeatures; feature++) {
      if (model[sample][feature] > featureCaps[feature]) {
        featureCaps[feature] = model[sample][feature];
      }
    }
  }

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
  double closestLabelNormal;
  int hasFoundLabel = 0;
  double difference = 0;

  for (label = 0; label < totalLabels; label++) {
    difference = (labelNormals[label] - predictions[0]);

    if (difference < 0) {
      difference = -difference;
    }

    if (hasFoundLabel == 0 || closestLabelNormal > difference) {
      closestLabelNormal = difference;
      hasFoundLabel = 1;
      prediction = labels[label];
    }
  }
  // printf("%c %f %f %f %f\n", prediction, predictions[0], labelNormals[0], labelNormals[totalLabels / 2], labelNormals[totalLabels - 1]);

  return prediction;
}
