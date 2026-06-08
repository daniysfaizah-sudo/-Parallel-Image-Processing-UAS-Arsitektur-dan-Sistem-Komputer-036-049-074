#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>


struct Image {
    int width, height;
    std::vector<unsigned char> data;
};

bool loadPPM(const std::string& filename, Image& img) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << filename << "\n"; return false; }
    std::string magic;
    f >> magic;
    if (magic != "P6") { std::cerr << "Only binary PPM (P6) supported\n"; return false; }
    int maxval;
    f >> img.width >> img.height >> maxval;
    f.ignore(1); 
    img.data.resize(img.width * img.height * 3);
    f.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    return true;
}

bool savePPM(const std::string& filename, const Image& img) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) { std::cerr << "Cannot write: " << filename << "\n"; return false; }
    f << "P6\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return true;
}

// Filter 1: GRAYSCALE


void grayscale_sequential(const Image& src, Image& dst) {
    dst.width  = src.width;
    dst.height = src.height;
    dst.data.resize(src.width * src.height * 3);

    for (int i = 0; i < src.width * src.height; i++) {
        unsigned char r = src.data[i * 3 + 0];
        unsigned char g = src.data[i * 3 + 1];
        unsigned char b = src.data[i * 3 + 2];
        unsigned char y = static_cast<unsigned char>(0.299f*r + 0.587f*g + 0.114f*b);
        dst.data[i * 3 + 0] = y;
        dst.data[i * 3 + 1] = y;
        dst.data[i * 3 + 2] = y;
    }
}


// Filter 2: SOBEL EDGE DETECTION

void sobel_sequential(const Image& src, Image& dst) {
    int W = src.width, H = src.height;
    dst.width  = W;
    dst.height = H;
    dst.data.resize(W * H * 3, 0);

    // Kernel Sobel
    int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Ky[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = src.data[((y + ky) * W + (x + kx)) * 3]; // channel R = grayscale
                    gx += Kx[ky + 1][kx + 1] * px;
                    gy += Ky[ky + 1][kx + 1] * px;
                }
            }
            int mag = static_cast<int>(std::sqrt((float)(gx*gx + gy*gy)));
            if (mag > 255) mag = 255;
            dst.data[(y * W + x) * 3 + 0] = static_cast<unsigned char>(mag);
            dst.data[(y * W + x) * 3 + 1] = static_cast<unsigned char>(mag);
            dst.data[(y * W + x) * 3 + 2] = static_cast<unsigned char>(mag);
        }
    }
}


// MAIN

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./sequential <input.ppm> <output_gray.ppm> <output_sobel.ppm>\n";
        return 1;
    }

    Image src, gray, sobel;

    if (!loadPPM(argv[1], src)) return 1;
    std::cout << "Image loaded: " << src.width << "x" << src.height << " pixels\n";

    // --- Benchmark Grayscale ---
    auto t0 = std::chrono::high_resolution_clock::now();
    grayscale_sequential(src, gray);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_gray = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- Benchmark Sobel ---
    auto t2 = std::chrono::high_resolution_clock::now();
    sobel_sequential(gray, sobel);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_sobel = std::chrono::duration<double, std::milli>(t3 - t2).count();

    savePPM(argv[2], gray);
    savePPM(argv[3], sobel);

    std::cout << "[SEQUENTIAL] Grayscale : " << ms_gray  << " ms\n";
    std::cout << "[SEQUENTIAL] Sobel     : " << ms_sobel << " ms\n";
    std::cout << "[SEQUENTIAL] Total     : " << (ms_gray + ms_sobel) << " ms\n";

    return 0;
}
