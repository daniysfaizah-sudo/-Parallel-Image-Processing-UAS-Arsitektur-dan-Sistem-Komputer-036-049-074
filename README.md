**PARALLEL IMAGE PROCESSING**

**Sequential – OpenMP – OpenCL**

 

**A.     Nama Penyusun :**

1. Daniys Faizah Athaillah (25032014036)  
2. Nur Faizatul Lailiyah (25032014049)  
3. Marda Azzah Taqiyyah (25032014074)

   

   **B.     Deskripsi Proyek**

    

   Proyek ini mengimplementasikan sistem pemrosesan gambar digital secara paralel menggunakan dua pendekatan komputasi paralel, yaitu OpenMP untuk paralelisme berbasis CPU dan OpenCL untuk paralelisme berbasis GPU. Sistem menerapkan dua filter pemrosesan gambar, yaitu filter Grayscale yang mengkonversi gambar RGB menjadi skala abu-abu menggunakan rumus luminance standar, dan filter Sobel Edge Detection yang mendeteksi tepi objek dalam gambar menggunakan konvolusi kernel 3×3.

    

   Setiap filter diimplementasikan dalam tiga versi berbeda. Versi sequential sebagai baseline yang memproses piksel secara berurutan menggunakan satu core CPU, versi OpenMP yang memparallelkan loop pemrosesan piksel ke beberapa thread CPU menggunakan direktif \#pragma omp parallel for, serta versi OpenCL yang mengeksekusi setiap piksel sebagai work-item secara masif di GPU menggunakan NDRange 2D.

   

   **C. Simulasi Cara Kerja**

   

1. Pastikan sudah menginstall tools berikut

   Python 3 — untuk menyiapkan gambar input dan menjalankan benchmark

   MinGW/G++ — untuk mengkompilasi kode C++

   OpenCL SDK — sesuai GPU yang digunakan (NVIDIA/AMD/Intel)

2. Siapkan 3 file dengan ukuran masing \- masing dalam format ppm  
* bendera\_512.ppm  
* bendera\_1024.ppm  
* bendera\_2048.ppm  
3. Compile code sequential.cpp, openmp.cpp dan opencl.cpp dengan menggunakan command di bawah ini di terminal  
* g++ \-O2 \-o sequential.exe src/sequential.cpp (sequential)  
* g++ \-O2 \-fopenmp \-o openmp\_filter.exe src/openmp.cpp (openmp)  
* g++ \-O2 \-o opencl\_filter.exe src/opencl.cpp \-lOpenCL(opencl)

  Jika berhasil, akan muncul file sequential.exe, openmp\_filter.exe dan opencl\_filter.mp di folder proyek.

4. Jalankan versi sequential terlebih dahulu

   

   .\\sequential.exe bendera\_512.ppm hasil\_gray.ppm hasil\_sobel.ppm

   .\\sequential.exe bendera\_1024.ppm hasil\_gray.ppm hasil\_sobel.ppm

   .\\sequential.exe bendera\_2048.ppm hasil\_gray.ppm hasil\_sobel.ppm

   

   Output terminal akan menampilkan waktu eksekusi Grayscale, Sobel, dan Total dalam milidetik.

   

5. Jalankan versi OpenMP  
* 2 thread

  .\\openmp\_filter.exe bendera\_512.ppm hasil\_gray.ppm hasil\_sobel.ppm 2

  .\\openmp\_filter.exe bendera\_1024.ppm hasil\_gray.ppm hasil\_sobel.ppm 2

  .\\openmp\_filter.exe bendera\_2048.ppm hasil\_gray.ppm hasil\_sobel.ppm 2

* 4 thread

  .\\openmp\_filter.exe bendera\_512.ppm hasil\_gray.ppm hasil\_sobel.ppm 4

  .\\openmp\_filter.exe bendera\_1024.ppm hasil\_gray.ppm hasil\_sobel.ppm 4

  .\\openmp\_filter.exe bendera\_2048.ppm hasil\_gray.ppm hasil\_sobel.ppm 4

6. Jalankan versi OpenCL

   

   .\\opencl\_filter.exe bendera\_512.ppm hasil\_gray.ppm hasil\_sobel.ppm

   .\\opencl\_filter.exe bendera\_1024.ppm hasil\_gray.ppm hasil\_sobel.ppm

   .\\opencl\_filter.exe bendera\_2048.ppm hasil\_gray.ppm hasil\_sobel.ppm

7. Jalankan python benchmark.py

   Script akan otomatis menjalankan semua versi dan menampilkan tabel waktu eksekusi beserta speedup masing-masing implementasi dibandingkan versi sequential.

8. Lihat hasil output

   Output akan berupa hasil\_gray.ppm dan hasil\_sobel.ppm dan dapat dilihat secara visual menggunakan bantuan aplikasi seperti GIMP 

   

**D. Link Video Youtube**

 [Parallel Image Processing | UAS Arsitektur dan Sistem Komputer | 036-049-074](https://youtu.be/OWz1UBBe7jw?si=hboUT_OviWwpy-Az)