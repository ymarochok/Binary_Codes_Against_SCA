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
    uint16_t features[NUM_FEATURES];
    uint16_t label;
} DataPoint;

int main() {
    clock_t start = clock();
    FILE *file = fopen("transformed_dataset.csv", "r");
    if (file == NULL) {
        printf("Error: Could not open file.\n");
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    DataPoint *data = malloc(MAX_SAMPLES * sizeof(DataPoint));
    int data_count = 0;

    if (data == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    while (fgets(line, sizeof(line), file) && data_count < MAX_SAMPLES) {
        if (strlen(line) <= 1) continue;
        if (line_count == 0) {
            line_count++;
            continue;
        }

        line[strcspn(line, "\n")] = 0;

        char *token = strtok(line, ",");
        int feature_index = 0;
        
        while (token != NULL && feature_index < NUM_FEATURES) {
            data[data_count].features[feature_index] = (uint16_t)strtol(token, NULL, 0);
            token = strtok(NULL, ",");
            feature_index++;
        }

        if (token != NULL) {
            data[data_count].label = (uint16_t)atoi(token);
        }

        data_count++;
        line_count++;
    }
    fclose(file);

    // execute network forward for each data point
    int correct = 0;
    for (int i = 0; i < data_count; i++) {

        int out = network_forward(data[i].features); 
        int pred = (out >= 0); // threshold
        int label = (int)data[i].label;

        // display results 
        printf("Network Output: %d | Prediction: %d | Actual: %d", out, pred, label);
        
        if (pred == label) {
            correct++;
            printf(" CORRECT\n\n");
        } else {
            printf(" WRONG\n\n");
        }
    }

    float accuracy = (float)correct / data_count * 100.0f;
    printf("Final Results: %d/%d Correct | Accuracy: %.2f%%\n", correct, data_count, accuracy);

    free(data);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total execution time: %.4f seconds\n", elapsed);
    return 0;
}