#include "denoise.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static inline void compare_and_swap(unsigned char *a, unsigned char *b) {
    if (*a > *b) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

static inline unsigned char *PTR(unsigned char *bmap, int width, int i, int j) {
    return (bmap + i * width + j);
}

static inline unsigned char median_of_five(unsigned char v[5]) {
    compare_and_swap(v + 3, v + 4);
    compare_and_swap(v + 2, v + 3);
    compare_and_swap(v + 1, v + 2);
    compare_and_swap(v, v + 1);
    compare_and_swap(v + 3, v + 4);
    compare_and_swap(v + 2, v + 3);
    compare_and_swap(v + 1, v + 2);
    compare_and_swap(v + 3, v + 4);
    compare_and_swap(v + 2, v + 3);
    return v[2];
}

void denoise_cpu(unsigned char *bmap, int width, int height) {
    unsigned char *out = (unsigned char*)malloc(width * height);
    unsigned char v[5];
    assert(out != NULL);

    memcpy(out, bmap, width * height);
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            v[0] = *PTR(bmap, width, i, j);
            v[1] = *PTR(bmap, width, i, j - 1);
            v[2] = *PTR(bmap, width, i, j + 1);
            v[3] = *PTR(bmap, width, i - 1, j);
            v[4] = *PTR(bmap, width, i + 1, j);
            *PTR(out, width, i, j) = median_of_five(v);
        }
    }
    memcpy(bmap, out, width * height);
    free(out);
}