#include <stdio.h>
#include "data_loader.h"

size_t load_csv(const char *path, Sample *buffer, size_t max_samples)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    size_t count = 0;

    while (count < max_samples && !feof(f)) {
        for (int i = 0; i < 10; i++) {
            fscanf(f, "%f,", &buffer[count].features[i]);
        }
        fscanf(f, "%f", &buffer[count].label);
        count++;
    }

    fclose(f);
    return count;
}
