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

// Weights
static double weights1[NUM_FEATURES][HIDDEN_LAYERS] = { 0 };
static double weights2[HIDDEN_LAYERS] = { 0 };

// Neurons
static double activation2[NUM_SAMPLES][HIDDEN_LAYERS] = { 0 };
static double activity2[NUM_SAMPLES][HIDDEN_LAYERS] = { 0 };
static double activation3[NUM_SAMPLES] = { 0 };

// Estimation
static double yHat[NUM_SAMPLES] = { 0 };

// Costs
static double costWeights2[HIDDEN_LAYERS] = { 0 };
static double costWeights1[NUM_FEATURES][HIDDEN_LAYERS] = { 0 };
static double backpropError2[NUM_SAMPLES][HIDDEN_LAYERS] = { 0 };
static double backpropError1[NUM_SAMPLES] = { 0 };

// Helpers
static int sample = 0;
static int feature = 0;
static int layer = 0;
static double sum = 0;
static double temp = 0;

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
      sum = model[sample][feature] / featureCaps[feature];
      modelNormals[sample][feature] = sum;
    }
  }
}

// Forward propagation
void forward() {
  // Dot product of model and weights1
  for (sample = 0; sample < totalSamples; sample++) {
    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      activation2[sample][layer] = 0;

      for (feature = 0; feature < totalFeatures; feature++) {
        sum = modelNormals[sample][feature] * weights1[feature][layer];
        activation2[sample][layer] += sum;
      }
    }
  }

  // Sigmoid activation2 for activity2
  for (sample = 0; sample < totalSamples; sample++) {
    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      sum = 1 / (1 + pow(M_E, -activation2[sample][layer]));
      activity2[sample][layer] = sum;
    }
  }

  // Dot product of activity2 and weights2
  for (sample = 0; sample < totalSamples; sample++) {
    activation3[sample] = 0;

    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      sum = activity2[sample][layer] * weights2[layer];
      activation3[sample] += sum;
    }
  }

  // Sigmoid activation3 for yHat
  for (sample = 0; sample < totalSamples; sample++) {
    sum = 1 / (1 + pow(M_E, -activation3[sample]));
    yHat[sample] = sum;
  }
}

double costFunction() {
  forward();
  sum = 0;

  for (sample = 0; sample < totalSamples; sample++) {
    temp = labelNormals[sample] - yHat[sample];
    sum += temp * temp;
  }

  return sum * 0.5;
}

void costFunctionPrime() {
  forward();

  // Calculate sigmoid prime of activation 3
  double temp1[NUM_SAMPLES] = { 0 };

  for (sample = 0; sample < totalSamples; sample++) {
    sum = pow(M_E, -activation3[sample]) / (1 + pow(M_E, -activation3[sample]));
    temp1[sample] = sum;
  }

  // Calculate negative label normals subtract yHat
  double temp2[NUM_SAMPLES] = { 0 };

  for (sample = 0; sample < totalSamples; sample++) {
    sum = -(labelNormals[sample] - yHat[sample]);
    temp2[sample] = sum;
  }

  // Multiply temp1 and temp2 to find backpropagating error 1
  for (sample = 0; sample < totalSamples; sample++) {
    sum = temp1[sample] * temp2[sample];
    backpropError1[sample] = sum;
  }

  // Calculate the dot product of activity2 (transpose) and backpropError1
  for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
    costWeights2[layer] = 0;

    for (sample = 0; sample < totalSamples; sample++) {
      sum = activity2[sample][layer] * backpropError1[sample];
      costWeights2[layer] += sum;
    }
  }

  // Calculate the dot product of backpropError1 and weights 2 (transpose)
  double temp3[NUM_SAMPLES][HIDDEN_LAYERS] = { 0 };

  for (sample = 0; sample < totalSamples; sample++) {
    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      sum = backpropError1[sample] * weights2[layer];
      temp3[sample][layer] = sum;
    }
  }

  // Calculate the sigmoid prime of activation 2
  double temp4[NUM_SAMPLES][HIDDEN_LAYERS] = { 0 };

  for (sample = 0; sample < totalSamples; sample++) {
    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      sum = activation2[sample][layer];
      temp4[sample][layer] = pow(M_E, -sum) / (1 + pow(M_E, -sum));
    }
  }

  // Multiply temp3 and temp4 to find backpropagating error 2
  for (sample = 0; sample < totalSamples; sample++) {
    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      sum = temp3[sample][layer] * temp4[sample][layer];
      backpropError2[sample][layer] = sum;
    }
  }

  // Calculate the dot product of model normals (transpose) and backpropError2
  for (feature = 0; feature < totalFeatures; feature++) {
    for (layer = 0; layer < HIDDEN_LAYERS; layer++) {
      costWeights1[feature][layer] = 0;

      for (sample = 0; sample < totalSamples; sample++) {
        sum = backpropError2[sample][layer] * modelNormals[sample][feature];
        costWeights1[feature][layer] += sum;
      }
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

  normalise();
  forward();

  return 1;
}

char predictLabel(double* sample, int numFeatures) {
  totalSamples = 1;
  totalFeatures = numFeatures;

  for (feature = 0; feature < totalFeatures; feature++) {
    model[0][feature] = sample[feature];
  }

  normalise();
  forward();

  char prediction;

  if (yHat[0] <= 0.25) {
    prediction = 'a';
  } else if (yHat[0] <= 0.75) {
    prediction = 'b';
  } else {
    prediction = 'c';
  }

  return prediction;
}
