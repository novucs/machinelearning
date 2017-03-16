#include <stdio.h>
#include <stdlib.h>

#include "MLCoursework.h"
#include "TrainAndTest.h"
#include "AbaloneData.h"
#include "BalanceData.h"
#include "BreastCancerData.h"
#include "IrisData.h"
#include "LetterData.h"
#include "SeedsData.h"

// This is just to show how your programme might be called. You should not
// assume that any particular data set is being used to test your code BUT you
// can assume that NUM_TRAINING_SAMPLES, NUM_FEATURES and NUM_TEST_SAMPLES are
// defined.

int main(int argc, const char *argv[]) {
  int sample = 0;
  int trainingsamples = 0;
  int testsamples = 0;
  int feature = 0;

  int ok = 1;

  int correct = 0;
  int wrong = 0;

  char prediction;
  int returnval = 0;
  double mySample[NUM_FEATURES];

  // Allocate space for data storage
  double **trainingSet = calloc(NUM_TRAINING_SAMPLES, sizeof(double *));

  for (sample = 0; sample < NUM_TRAINING_SAMPLES; sample++) {
    trainingSet[sample] = calloc(NUM_FEATURES, sizeof(double));
  }

  char *trainingLabels = calloc(NUM_TRAINING_SAMPLES, sizeof(char));
  double **testSet = calloc(NUM_TEST_SAMPLES, sizeof(double *));

  for (sample = 0; sample < NUM_TEST_SAMPLES; sample++) {
    testSet[sample] = calloc(NUM_FEATURES, sizeof(double));
  }

  char *testLabels = calloc(NUM_TEST_SAMPLES, sizeof(char));

  // Simple 2/3: 1/3 split of iris data into training and test set matrices

  if (USETEST == 0) {
    for (sample = 0; sample < IRIS_SET_SIZE; sample++) {
      if (sample % TRAINTESTRATIO == 0) {
        for (feature = 0; feature < IRISFEATURES; feature++) {
          testSet[testsamples][feature] = iris_data[sample][feature];
        }

        testLabels[testsamples] = iris_labels[sample];
        testsamples++;
      } else {
        for (feature = 0; feature < IRISFEATURES; feature++) {
          trainingSet[trainingsamples][feature] = iris_data[sample][feature];
        }

        trainingLabels[trainingsamples] = iris_labels[sample];
        trainingsamples++;
      }
    }
  }

  if (USETEST == 1) {
    for (sample = 0; sample < ABALONE_SET_SIZE; sample++) {
      if (sample % TRAINTESTRATIO == 0) {
        for (feature = 0; feature < ABALONEFEATURES; feature++) {
          testSet[testsamples][feature] = abalone_data[sample][feature];
        }

        testLabels[testsamples] = abalone_labels[sample];
        testsamples++;
      } else {
        for (feature = 0; feature < ABALONEFEATURES; feature++) {
          trainingSet[trainingsamples][feature] = abalone_data[sample][feature];
        }

        trainingLabels[trainingsamples] = abalone_labels[sample];
        trainingsamples++;
      }
    }
  }

  if (USETEST == 2) {
    for (sample = 0; sample < BALANCE_SET_SIZE; sample++) {
      if (sample % TRAINTESTRATIO == 0) {
        for (feature = 0; feature < BALANCEFEATURES; feature++) {
          testSet[testsamples][feature] = balance_data[sample][feature];
        }

        testLabels[testsamples] = balance_labels[sample];
        testsamples++;
      } else {
        for (feature = 0; feature < BALANCEFEATURES; feature++) {
          trainingSet[trainingsamples][feature] = balance_data[sample][feature];
        }

        trainingLabels[trainingsamples] = balance_labels[sample];
        trainingsamples++;
      }
    }
  }

  if (USETEST == 3) {
    for (sample = 0; sample < SEEDS_SET_SIZE; sample++) {
      if (sample % TRAINTESTRATIO == 0) {
        for (feature = 0; feature < SEEDSFEATURES; feature++) {
          testSet[testsamples][feature] = seeds_data[sample][feature];
        }

        testLabels[testsamples] = seeds_labels[sample];
        testsamples++;
      } else {
        for (feature = 0; feature < SEEDSFEATURES; feature++) {
          trainingSet[trainingsamples][feature] = seeds_data[sample][feature];
        }

        trainingLabels[trainingsamples] = seeds_labels[sample];
        trainingsamples++;
      }
    }
  }

  if (USETEST == 4) {
    for (sample = 0; sample < LETTER_SET_SIZE; sample++) {
      if (sample % TRAINTESTRATIO == 0) {
        for (feature = 0; feature < LETTERFEATURES; feature++)
          testSet[testsamples][feature] = letter_data[sample][feature];

        testLabels[testsamples] = letter_labels[sample];
        testsamples++;
      } else {
        for (feature = 0; feature < LETTERFEATURES; feature++) {
          trainingSet[trainingsamples][feature] = letter_data[sample][feature];
        }

        trainingLabels[trainingsamples] = letter_labels[sample];
        trainingsamples++;
      }
    }
  }

  if (USETEST == 5) {
    for (sample = 0; sample < BCANC_SET_SIZE; sample++) {
      if (sample % TRAINTESTRATIO == 0) {
        for (feature = 0; feature < BCANCFEATURES; feature++) {
          testSet[testsamples][feature] = bcanc_data[sample][feature];
        }

        testLabels[testsamples] = bcanc_labels[sample];
        testsamples++;
      } else {
        for (feature = 0; feature < BCANCFEATURES; feature++) {
          trainingSet[trainingsamples][feature] = bcanc_data[sample][feature];
        }

        trainingLabels[trainingsamples] = bcanc_labels[sample];
        trainingsamples++;
      }
    }
  }

  // Call the train function
  ok = train(&trainingSet[0], trainingLabels, trainingsamples, NUM_FEATURES);

  if (ok != 1) {
    printf("there was a problem running the train() function\n");
    returnval = 0;
  } else {
    // Print the results from the training set for information
    printf("On the training set:\n");

    for (sample = 0; sample < trainingsamples; sample++) {
      // Make a copy of the sample ot be classified because it is faster to
      // access a local copy and pass a pointer to it, and also  I don't want
      // to overwrite it by accident
      for (feature = 0; feature < NUM_FEATURES; feature++) {
        mySample[feature] = trainingSet[sample][feature];
      }

      prediction = predictLabel(mySample, NUM_FEATURES);

      if (prediction == trainingLabels[sample]) {
        correct++;
      } else {
        wrong++;
      }
    }

    printf(" %d correct %d incorrect, accuracy = %f %%\n", correct, wrong, 100.0 * (float)correct / (correct + wrong));

    // Now the results that matter - from the test set
    printf("On the test set:\n");

    correct = 0;
    wrong = 0;

    for (sample = 0; sample < testsamples; sample++) {
      // Copy into temp array ot pass into fiunction
      for (feature = 0; feature < NUM_FEATURES; feature++) {
        mySample[feature] = testSet[sample][feature];
      }

      prediction = predictLabel(mySample, NUM_FEATURES);

      if (prediction == testLabels[sample]) {
        correct++;
      } else {
        // printf("actual = %c predicted = %c\n", testLabels[sample], prediction);
        wrong++;
      }
    }

    double accuracy = 100.0 * (float)correct / (correct + wrong);
    printf(" %d correct %d incorrect, accuracy = %f %%\n",
           correct, wrong, accuracy);
  }

  return returnval;
}
