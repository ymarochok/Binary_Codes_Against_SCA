#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network.h"
#include "data_loader.h"
#include "network_config.h"
#include <time.h>
#define MAX_SAMPLES 2000

#define MAX_LINE_LENGTH 1024
#define NUM_FEATURES 10

typedef struct {
    int8_t features[NUM_FEATURES];
    int8_t label;
} DataPoint;


int main() {
    clock_t start = clock();
    FILE *file = fopen("test.csv", "r");
    if (file == NULL) {
        printf("Error: Could not open file.\n");
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    DataPoint *data = NULL;
    int data_count = 0;

    // Read file line by line
    while (fgets(line, sizeof(line), file) && data_count < MAX_SAMPLES) {
        // Skip empty lines
        if (strlen(line) <= 1) continue;

        // Skip header line
        if (line_count == 0) {
            line_count++;
            continue;
        }

        // Remove newline character
        line[strcspn(line, "\n")] = 0;

        // Allocate memory for new data point
        data = realloc(data, (data_count + 1) * sizeof(DataPoint));
        if (data == NULL) {
            printf("Error: Memory allocation failed.\n");
            return 1;
        }

        printf("%s\n", line);
        // Parse CSV line
        char *token = strtok(line, ",");
        int feature_index = 0;
        
        while (token != NULL && feature_index < NUM_FEATURES) {
            data[data_count].features[feature_index] = atof(token);
            token = strtok(NULL, ",");
            feature_index++;
        }

        // Read label (last column)
        if (token != NULL) {
            data[data_count].label = atof(token);
        }

        data_count++;
        line_count++;
    }

    fclose(file);

    printf("Successfully read %d data points\n\n", data_count);

    // Your evaluation code integrated here
    int correct = 0;
    size_t n = data_count;  // Using the actual number of data points read

    for (size_t i = 0; i < n; i++) {
        int out = network_forward(data[i].features);
        int pred = (out >= 0.0f);   // threshold at 0
        int label = (int)data[i].label;

        printf("Data point %zu: Output=%.6f, Prediction=%d, Actual Label=%d", 
               i, out, pred, label);
        
        if (pred == label) {
            correct++;
            printf(" ✓ CORRECT\n");
        } else {
            printf(" ✗ WRONG\n");
        }
    }

    // Calculate and display accuracy
    float accuracy = (float)correct / n * 100.0f;
    printf("\nResults:\n");
    printf("Correct predictions: %d/%zu\n", correct, n);
    printf("Accuracy: %.2f%%\n", accuracy);

    // Free allocated memory
    free(data);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total execution time: %.4f seconds\n", elapsed);
    return 0;
}
