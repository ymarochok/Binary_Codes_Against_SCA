#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stddef.h>

typedef struct {
    float features[10];
    float label;
} Sample;

size_t load_csv(const char *path, Sample *buffer, size_t max_samples);

#endif
