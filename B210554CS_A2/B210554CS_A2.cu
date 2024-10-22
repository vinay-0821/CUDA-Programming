#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>

#define IDX2C(i, j, N) (((i) * (N)) + (j))

using namespace std;
using namespace std::chrono;

// Kernel for LU Decomposition using Gaussian Elimination
__global__ void luDecompositionKernel(double *A, double *L, double *U, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        for (int k = 0; k < N; ++k) {
            if (i == 0) {
                // Diagonal and upper triangle elements in U
                for (int j = k; j < N; ++j) {
                    U[IDX2C(k, j, N)] = A[IDX2C(k, j, N)];
                    for (int p = 0; p < k; ++p)
                        U[IDX2C(k, j, N)] -= L[IDX2C(k, p, N)] * U[IDX2C(p, j, N)];
                }

                // Lower triangle elements in L
                for (int i = k + 1; i < N; ++i) {
                    L[IDX2C(i, k, N)] = A[IDX2C(i, k, N)];
                    for (int p = 0; p < k; ++p)
                        L[IDX2C(i, k, N)] -= L[IDX2C(i, p, N)] * U[IDX2C(p, k, N)];
                    L[IDX2C(i, k, N)] /= U[IDX2C(k, k, N)];
                }
                L[IDX2C(k, k, N)] = 1.0; // Diagonal of L is 1
            }
            __syncthreads();  // Synchronize threads
        }
    }
}

// Forward substitution (Sequential, no parallelization)
__global__ void CalculatingYKernel(double *L, double *B, double *Y, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        for (int j = 0; j < N; ++j) {
            Y[j] = B[j];
            for (int k = 0; k < j; ++k) {
                Y[j] -= L[IDX2C(j, k, N)] * Y[k];
            }
        }
    }
}

// Backward substitution (Sequential, no parallelization)
__global__ void CalculateXKernel(double *U, double *Y, double *X, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        for (int j = N - 1; j >= 0; --j) {
            X[j] = Y[j];
            for (int k = j + 1; k < N; ++k) {
                X[j] -= U[IDX2C(j, k, N)] * X[k];
            }
            X[j] /= U[IDX2C(j, j, N)];
        }
    }
}

// Host function to read matrix from file
void readInput(const char *filename, double *A, double *B, int N) {
    ifstream file(filename);

    if (file.is_open()) {
        file >> N;
        for (int i = 0; i < N * N; ++i)
            file >> A[i];
        for (int i = 0; i < N; ++i)
            file >> B[i];
        file.close();
    } else {
        cerr << "Unable to open file!" << endl;
        exit(EXIT_FAILURE);
    }
}

// Host function to write output
  // Include this for setprecision

void writeOutput(const char *filename, double *L, double *U, double *X, int N, double *Y) {
    ofstream file(filename);
    if (file.is_open()) {
        file << fixed << setprecision(16);  // Set precision to 10 decimal places

        file << N << endl;

        // Write L matrix
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                file << L[IDX2C(i, j, N)] << endl;
            }
            //file << endl;
        }

        // Write U matrix
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                file << U[IDX2C(i, j, N)] << endl;
            }
            //file << endl;
        }

        // Write solution vector X
        for (int i = 0; i < N; ++i) {
            file << X[i] << endl;
        }

        file.close();
    } else {
        cerr << "Unable to open output file!" << endl;
    }

    //cout << fixed << setprecision(16);  // Set precision for console output

    // Print L matrix
    // cout << "L matrix (Row-wise):" << endl;
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         cout << L[IDX2C(i, j, N)] << " ";
    //     }
    //     cout << endl;
    // }

    // // Print U matrix
    // cout << "U matrix (Row-wise):" << endl;
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         cout << U[IDX2C(i, j, N)] << " ";
    //     }
    //     cout << endl;
    // }

    // // Print Y vector
    // cout << "Y vector:" << endl;
    // for (int i = 0; i < N; ++i)
    //     cout << Y[i] << endl;

    // // Print solution X
    // cout << "X vector:" << endl;
    // for (int i = 0; i < N; ++i)
    //     cout << X[i] << endl;
}

// void writeTimingReport(const char *filename, double timeRead, double timeL, double timeU, double totalTime) {
//     ofstream file(filename);
//     if (file.is_open()) {
//         file << fixed << setprecision(6);
//         file << "Time to read A and B matrices: " << timeRead << " seconds" << endl;
//         file << "Time to compute L matrix and U matrix: " << timeL << " seconds" << endl;
//         file << "Time to compute Y matrix and X(solution) matrix: " << timeU << " seconds" << endl;
//         file << "Total time to solve system of equations: " << totalTime << " seconds" << endl;
//         file.close();
//     } else {
//         cerr << "Unable to open timing report file!" << endl;
//     }
// }


int main(int argc, char **argv) {
    // if (argc != 4) {
    //     cerr << "Usage: " << argv[0] << " <input file> <output file> <timing report file>" << endl;
    //     return 1;
    // }

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input file> <output file>" << endl;
        return 1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];
    //const char *timingFile = argv[3];

    int N;
    ifstream inFile(inputFile);
    if (inFile.is_open()) {
        inFile >> N;
        inFile.close();
    } else {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    // Allocate host memory
    double *A = new double[N * N];
    double *B = new double[N];
    double *L = new double[N * N]();
    double *U = new double[N * N]();
    double *X = new double[N];
    double *Y = new double[N];

    // Read input matrices A and B
    auto startRead = high_resolution_clock::now();
    readInput(inputFile, A, B, N);
    auto endRead = high_resolution_clock::now();
    double timeRead = duration_cast<duration<double>>(endRead - startRead).count();

    // Allocate device memory
    double *d_A, *d_L, *d_U, *d_B, *d_X, *d_Y;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_L, N * N * sizeof(double));
    cudaMalloc(&d_U, N * N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_X, N * sizeof(double));
    cudaMalloc(&d_Y, N * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);

    // Set up kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Perform LU decomposition
    auto startLU = high_resolution_clock::now();
    luDecompositionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_L, d_U, N);
    cudaDeviceSynchronize();
    auto endLU = high_resolution_clock::now();
    double timeLU = duration_cast<duration<double>>(endLU - startLU).count();

    // Forward substitution
    auto startSolve = high_resolution_clock::now();
    CalculatingYKernel<<<1, 1>>>(d_L, d_B, d_Y, N);
    cudaDeviceSynchronize();

    // Backward substitution
    CalculateXKernel<<<1, 1>>>(d_U, d_Y, d_X, N);
    cudaDeviceSynchronize();
    auto endSolve = high_resolution_clock::now();
    double timeSolve = duration_cast<duration<double>>(endSolve - startSolve).count();

    // Total time for solving the system of equations
    double totalTime = timeRead + timeLU + timeSolve;

    // Copy results back to host
    cudaMemcpy(L, d_L, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(X, d_X, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Write the output
    writeOutput(outputFile, L, U, X, N, Y);
    //writeTimingReport(timingFile, timeRead, timeLU, timeSolve, totalTime);
    // cout << fixed << setprecision(6);
    // cout << "Time to read A and B matrices: " << timeRead << " seconds" << endl;
    // cout << "Time to compute L matrix and U matrix: " << timeLU << " seconds" << endl;
    // cout << "Time to compute Y matrix and X(solution) matrix: " << timeSolve << " seconds" << endl;
    // cout << "Total time to solve system of equations: " << totalTime << " seconds" << endl;

    // Free memory
    delete[] A, B, L, U, X, Y;
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_B);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
