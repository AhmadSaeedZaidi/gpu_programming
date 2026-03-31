#include "ppm_utils.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void read_ppm(FILE *f, PPM_image* img) {
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P6\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    
    sscanf(s, "%d %d", &(img->width), &(img->height));
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxcol));
    
    if (img->maxcol > 255) {
        fprintf(stderr, "FATAL: maxcol=%d > 255\n", img->maxcol);
        exit(EXIT_FAILURE);
    }
    
    img->r = (unsigned char*)malloc((img->width) * (img->height));
    assert(img->r != NULL);
    img->g = (unsigned char*)malloc((img->width) * (img->height));
    assert(img->g != NULL);
    img->b = (unsigned char*)malloc((img->width) * (img->height));
    assert(img->b != NULL);
    
    for (int k = 0; k < (img->width) * (img->height); k++) {
        nread = fscanf(f, "%c%c%c", (char*)(img->r + k), (char*)(img->g + k), (char*)(img->b + k));
        if (nread != 3) {
            fprintf(stderr, "FATAL: error reading pixel data\n");
            exit(EXIT_FAILURE);
        }
    }
}

void write_ppm(FILE *f, const PPM_image* img, const char *comment) {
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P6\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxcol);
    for (int k = 0; k < (img->width) * (img->height); k++) {
        fprintf(f, "%c%c%c", img->r[k], img->g[k], img->b[k]);
    }
}

void free_ppm(PPM_image* img) {
    assert(img != NULL);
    free(img->r);
    free(img->g);
    free(img->b);
    img->r = img->g = img->b = NULL;
    img->width = img->height = img->maxcol = -1;
}