#ifndef MLCoursework_h
#define MLCoursework_h

// Iris
#define NUM_FEATURES 4
#define NUM_SAMPLES 150
#define USETEST 0
#define TRAINTESTRATIO 3

// // Abalone
// #define NUM_FEATURES 8
// #define NUM_SAMPLES 4000
// #define USETEST 1
// #define TRAINTESTRATIO 4

// // Balance
// #define NUM_FEATURES 4
// #define NUM_SAMPLES 625
// #define USETEST 2
// #define TRAINTESTRATIO 5

// // Seeds
// #define NUM_FEATURES 7
// #define NUM_SAMPLES 210
// #define USETEST 3
// #define TRAINTESTRATIO 5

// // Letter
// #define NUM_FEATURES 16
// #define NUM_SAMPLES 2000
// #define USETEST 4
// #define TRAINTESTRATIO 8

// // BreastCancer
// #define NUM_FEATURES 9
// #define NUM_SAMPLES 699
// #define USETEST 5
// #define TRAINTESTRATIO 3

#define NUM_TRAINING_SAMPLES (NUM_SAMPLES - (NUM_SAMPLES / TRAINTESTRATIO))
#define NUM_TEST_SAMPLES (NUM_SAMPLES / TRAINTESTRATIO)

#endif /* MLCoursework_h */
