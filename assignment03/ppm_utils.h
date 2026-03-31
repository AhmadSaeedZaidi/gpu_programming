#ifndef PPM_UTILS_H
#define PPM_UTILS_H

#include <stdio.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxcol;  /* Largest color value */
    unsigned char *r, *g, *b; /* Color channels */
} PPM_image;

void read_ppm(FILE *f, PPM_image* img);
void write_ppm(FILE *f, const PPM_image* img, const char *comment);
void free_ppm(PPM_image* img);

#endif // PPM_UTILS_H