#ifndef DENOISE_H
#define DENOISE_H

#ifdef __cplusplus
extern "C" {
#endif

void denoise_cpu(unsigned char *h_bmap, int width, int height);
void denoise_gpu(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // DENOISE_H