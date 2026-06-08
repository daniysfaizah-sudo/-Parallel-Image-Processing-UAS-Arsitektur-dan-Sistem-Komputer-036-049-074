#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

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
    if (!f) return false;
    f << "P6\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return true;
}

std::string readKernelFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open kernel: " << path << "\n"; return ""; }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

#define CL_CHECK(err, msg) if (err != CL_SUCCESS) { \
    std::cerr << "[OpenCL Error " << err << "] " << msg << "\n"; return 1; }

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./opencl_filter <input.ppm> <output_gray.ppm> <output_sobel.ppm>\n";
        return 1;
    }

    Image src;
    if (!loadPPM(argv[1], src)) return 1;
    int W = src.width, H = src.height;
    size_t imgSize = W * H * 3 * sizeof(unsigned char);
    std::cout << "Image loaded: " << W << "x" << H << " pixels\n";

    cl_int err;
    cl_platform_id platform;
    cl_uint numPlatforms;
    clGetPlatformIDs(1, &platform, &numPlatforms);
    if (numPlatforms == 0) { std::cerr << "No OpenCL platforms found!\n"; return 1; }

    cl_device_id device;
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cout << "No GPU found, using CPU as OpenCL device\n";
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &numDevices);
        CL_CHECK(err, "No OpenCL device found");
    }

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "OpenCL Device: " << deviceName << "\n";

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err, "Create context");
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    CL_CHECK(err, "Create command queue");

    std::string kernelSrc = readKernelFile("filters.cl");
    if (kernelSrc.empty()) return 1;
    const char* src_cstr = kernelSrc.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src_cstr, nullptr, &err);
    CL_CHECK(err, "Create program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build error:\n" << log << "\n";
        return 1;
    }

    cl_kernel kernel_gray  = clCreateKernel(program, "grayscale", &err);
    CL_CHECK(err, "Create grayscale kernel");
    cl_kernel kernel_sobel = clCreateKernel(program, "sobel", &err);
    CL_CHECK(err, "Create sobel kernel");

    cl_mem buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgSize, src.data.data(), &err);
    CL_CHECK(err, "Create input buffer");
    cl_mem buf_gray  = clCreateBuffer(context, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    CL_CHECK(err, "Create gray buffer");
    cl_mem buf_sobel = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);
    CL_CHECK(err, "Create sobel buffer");

    size_t global[2] = { ((size_t)W + 15) / 16 * 16, ((size_t)H + 15) / 16 * 16 };
    size_t local[2]  = { 16, 16 };

    // Grayscale
    clSetKernelArg(kernel_gray, 0, sizeof(cl_mem), &buf_input);
    clSetKernelArg(kernel_gray, 1, sizeof(cl_mem), &buf_gray);
    clSetKernelArg(kernel_gray, 2, sizeof(int),    &W);
    clSetKernelArg(kernel_gray, 3, sizeof(int),    &H);

    auto t0 = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, kernel_gray, 2, nullptr, global, local, 0, nullptr, nullptr);
    CL_CHECK(err, "Enqueue grayscale");
    clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Sobel
    clSetKernelArg(kernel_sobel, 0, sizeof(cl_mem), &buf_gray);
    clSetKernelArg(kernel_sobel, 1, sizeof(cl_mem), &buf_sobel);
    clSetKernelArg(kernel_sobel, 2, sizeof(int),    &W);
    clSetKernelArg(kernel_sobel, 3, sizeof(int),    &H);

    auto t2 = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, kernel_sobel, 2, nullptr, global, local, 0, nullptr, nullptr);
    CL_CHECK(err, "Enqueue sobel");
    clFinish(queue);
    auto t3 = std::chrono::high_resolution_clock::now();

    // Benchmark
    double ms_gray  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_sobel = std::chrono::duration<double, std::milli>(t3 - t2).count();

    Image gray_img, sobel_img;
    gray_img.width = sobel_img.width = W;
    gray_img.height = sobel_img.height = H;
    gray_img.data.resize(imgSize);
    sobel_img.data.resize(imgSize);

    clEnqueueReadBuffer(queue, buf_gray,  CL_TRUE, 0, imgSize, gray_img.data.data(),  0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, buf_sobel, CL_TRUE, 0, imgSize, sobel_img.data.data(), 0, nullptr, nullptr);

    savePPM(argv[2], gray_img);
    savePPM(argv[3], sobel_img);

    std::cout << "[OpenCL] Grayscale : " << ms_gray  << " ms\n";
    std::cout << "[OpenCL] Sobel     : " << ms_sobel << " ms\n";
    std::cout << "[OpenCL] Total     : " << (ms_gray + ms_sobel) << " ms\n";

    clReleaseMemObject(buf_input);
    clReleaseMemObject(buf_gray);
    clReleaseMemObject(buf_sobel);
    clReleaseKernel(kernel_gray);
    clReleaseKernel(kernel_sobel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}