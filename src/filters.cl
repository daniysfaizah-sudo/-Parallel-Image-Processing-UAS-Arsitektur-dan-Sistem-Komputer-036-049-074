// Kernel 1 : Grayscale

__kernel void grayscale(
    __global const uchar* input,   // RGB input  [H * W * 3]
    __global       uchar* output,  // RGB output [H * W * 3]
    int width,
    int height)
{
    int x = get_global_id(0); // kolom
    int y = get_global_id(1); // baris

    if (x >= width || y >= height) return; // guard: jangan keluar batas

    int idx = (y * width + x) * 3;

    uchar r = input[idx + 0];
    uchar g = input[idx + 1];
    uchar b = input[idx + 2];

    uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);

    output[idx + 0] = gray;
    output[idx + 1] = gray;
    output[idx + 2] = gray;
}

// Kernel 2 : 

__kernel void sobel(
    __global const uchar* input,   // Grayscale input  [H * W * 3]
    __global       uchar* output,  // Edge output      [H * W * 3]
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Border pixels dibiarkan 0 (tidak diproses)
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            int idx = (y * width + x) * 3;
            output[idx + 0] = 0;
            output[idx + 1] = 0;
            output[idx + 2] = 0;
        }
        return;
    }

    // Kernel Sobel
    int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Ky[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    int gx = 0, gy = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = (int)input[((y + ky) * width + (x + kx)) * 3]; // channel R = gray
            gx += Kx[ky + 1][kx + 1] * px;
            gy += Ky[ky + 1][kx + 1] * px;
        }
    }

    int mag = (int)sqrt((float)(gx*gx + gy*gy));
    if (mag > 255) mag = 255;

    int out_idx = (y * width + x) * 3;
    output[out_idx + 0] = (uchar)mag;
    output[out_idx + 1] = (uchar)mag;
    output[out_idx + 2] = (uchar)mag;
}
