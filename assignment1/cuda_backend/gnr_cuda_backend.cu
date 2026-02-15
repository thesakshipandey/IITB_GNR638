#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                         cudaGetErrorString(err), __FILE__, __LINE__);        \
            return;                                                           \
        }                                                                     \
    } while (0)

__global__ void g16Conv2dForwardKernel(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int batch,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int out_h,
    int out_w,
    int k_h,
    int k_w,
    int stride,
    int padding) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_c * out_h * out_w;
    if (index >= total) {
        return;
    }

    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_c;
    int n = tmp / out_c;

    float acc = b[oc];
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_h; ++kh) {
            int in_y = oh * stride + kh - padding;
            if (in_y < 0 || in_y >= in_h) {
                continue;
            }
            for (int kw = 0; kw < k_w; ++kw) {
                int in_x = ow * stride + kw - padding;
                if (in_x < 0 || in_x >= in_w) {
                    continue;
                }

                int x_pos = ((n * in_c + ic) * in_h + in_y) * in_w + in_x;
                int w_pos = ((oc * in_c + ic) * k_h + kh) * k_w + kw;
                acc += x[x_pos] * w[w_pos];
            }
        }
    }

    out[index] = acc;
}

__global__ void g16Conv2dBackwardKernel(
    const float* dout,
    const float* x,
    const float* w,
    float* dx,
    float* dw,
    float* db,
    int batch,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int out_h,
    int out_w,
    int k_h,
    int k_w,
    int stride,
    int padding) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_c * out_h * out_w;
    if (index >= total) {
        return;
    }

    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_c;
    int n = tmp / out_c;

    float grad = dout[index];
    atomicAdd(&db[oc], grad);

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_h; ++kh) {
            int in_y = oh * stride + kh - padding;
            if (in_y < 0 || in_y >= in_h) {
                continue;
            }
            for (int kw = 0; kw < k_w; ++kw) {
                int in_x = ow * stride + kw - padding;
                if (in_x < 0 || in_x >= in_w) {
                    continue;
                }

                int x_pos = ((n * in_c + ic) * in_h + in_y) * in_w + in_x;
                int w_pos = ((oc * in_c + ic) * k_h + kh) * k_w + kw;

                atomicAdd(&dw[w_pos], x[x_pos] * grad);
                atomicAdd(&dx[x_pos], w[w_pos] * grad);
            }
        }
    }
}

__global__ void g16ReluForwardKernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

__global__ void g16ReluBackwardKernel(
    const float* dout,
    const float* x,
    float* dx,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dx[idx] = x[idx] > 0.0f ? dout[idx] : 0.0f;
    }
}

__global__ void g16Maxpool2dForwardKernel(
    const float* x,
    float* out,
    int* mask,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pool,
    int stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (index >= total) {
        return;
    }

    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int c = tmp % channels;
    int n = tmp / channels;

    float best_val = -1e30f;
    int best_idx = 0;

    for (int ph = 0; ph < pool; ++ph) {
        for (int pw = 0; pw < pool; ++pw) {
            int in_y = oh * stride + ph;
            int in_x = ow * stride + pw;
            int in_pos = ((n * channels + c) * in_h + in_y) * in_w + in_x;
            float val = x[in_pos];
            if (val > best_val) {
                best_val = val;
                best_idx = in_pos;
            }
        }
    }

    out[index] = best_val;
    mask[index] = best_idx;
}

__global__ void g16Maxpool2dBackwardKernel(
    const float* dout,
    const int* mask,
    float* dx,
    int out_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_n) {
        return;
    }
    int input_pos = mask[idx];
    atomicAdd(&dx[input_pos], dout[idx]);
}

__global__ void g16LinearForwardKernel(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int batch,
    int in_features,
    int out_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;
    if (index >= total) {
        return;
    }

    int of = index % out_features;
    int n = index / out_features;

    float acc = b[of];
    for (int inf = 0; inf < in_features; ++inf) {
        int x_pos = n * in_features + inf;
        int w_pos = of * in_features + inf;
        acc += x[x_pos] * w[w_pos];
    }
    out[index] = acc;
}

__global__ void g16LinearBackwardKernel(
    const float* dout,
    const float* x,
    const float* w,
    float* dx,
    float* dw,
    float* db,
    int batch,
    int in_features,
    int out_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;
    if (index >= total) {
        return;
    }

    int of = index % out_features;
    int n = index / out_features;

    float grad = dout[index];
    atomicAdd(&db[of], grad);

    for (int inf = 0; inf < in_features; ++inf) {
        int x_pos = n * in_features + inf;
        int w_pos = of * in_features + inf;
        atomicAdd(&dw[w_pos], grad * x[x_pos]);
        atomicAdd(&dx[x_pos], grad * w[w_pos]);
    }
}

extern "C" {

int g16CudaBackendAvailable() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        return 0;
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return 0;
    }

    return 1;
}

void g16Conv2dForwardCuda(
    const float* h_x,
    const float* h_w,
    const float* h_b,
    float* h_out,
    int batch,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int k_h,
    int k_w,
    int stride,
    int padding) {
    int out_h = (in_h + 2 * padding - k_h) / stride + 1;
    int out_w = (in_w + 2 * padding - k_w) / stride + 1;

    size_t x_bytes = static_cast<size_t>(batch) * in_c * in_h * in_w * sizeof(float);
    size_t w_bytes = static_cast<size_t>(out_c) * in_c * k_h * k_w * sizeof(float);
    size_t b_bytes = static_cast<size_t>(out_c) * sizeof(float);
    size_t out_bytes = static_cast<size_t>(batch) * out_c * out_h * out_w * sizeof(float);

    float *d_x, *d_w, *d_b, *d_out;

    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    int total = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    g16Conv2dForwardKernel<<<blocks, threads>>>(
        d_x,
        d_w,
        d_b,
        d_out,
        batch,
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        k_h,
        k_w,
        stride,
        padding);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

void g16Conv2dBackwardCuda(
    const float* h_dout,
    const float* h_x,
    const float* h_w,
    float* h_dx,
    float* h_dw,
    float* h_db,
    int batch,
    int in_c,
    int in_h,
    int in_w,
    int out_c,
    int k_h,
    int k_w,
    int stride,
    int padding) {
    int out_h = (in_h + 2 * padding - k_h) / stride + 1;
    int out_w = (in_w + 2 * padding - k_w) / stride + 1;

    size_t dout_bytes = static_cast<size_t>(batch) * out_c * out_h * out_w * sizeof(float);
    size_t x_bytes = static_cast<size_t>(batch) * in_c * in_h * in_w * sizeof(float);
    size_t w_bytes = static_cast<size_t>(out_c) * in_c * k_h * k_w * sizeof(float);
    size_t db_bytes = static_cast<size_t>(out_c) * sizeof(float);

    float *d_dout, *d_x, *d_w, *d_dx, *d_dw, *d_db;

    CUDA_CHECK(cudaMalloc(&d_dout, dout_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_dx, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_dw, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_db, db_bytes));

    CUDA_CHECK(cudaMemcpy(d_dout, h_dout, dout_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_dx, 0, x_bytes));
    CUDA_CHECK(cudaMemset(d_dw, 0, w_bytes));
    CUDA_CHECK(cudaMemset(d_db, 0, db_bytes));

    int total = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    g16Conv2dBackwardKernel<<<blocks, threads>>>(
        d_dout,
        d_x,
        d_w,
        d_dx,
        d_dw,
        d_db,
        batch,
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        k_h,
        k_w,
        stride,
        padding);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dx, d_dx, x_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dw, d_dw, w_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db, d_db, db_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_dout);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_dx);
    cudaFree(d_dw);
    cudaFree(d_db);
}

void g16ReluForwardCuda(const float* h_x, float* h_out, int n) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_x, *d_out;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    g16ReluForwardKernel<<<blocks, threads>>>(d_x, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_out);
}

void g16ReluBackwardCuda(
    const float* h_dout,
    const float* h_x,
    float* h_dx,
    int n) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_dout, *d_x, *d_dx;

    CUDA_CHECK(cudaMalloc(&d_dout, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_dx, bytes));

    CUDA_CHECK(cudaMemcpy(d_dout, h_dout, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    g16ReluBackwardKernel<<<blocks, threads>>>(d_dout, d_x, d_dx, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dx, d_dx, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_dout);
    cudaFree(d_x);
    cudaFree(d_dx);
}

void g16Maxpool2dForwardCuda(
    const float* h_x,
    float* h_out,
    int* h_mask,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int pool,
    int stride) {
    int out_h = (in_h - pool) / stride + 1;
    int out_w = (in_w - pool) / stride + 1;

    size_t x_bytes = static_cast<size_t>(batch) * channels * in_h * in_w * sizeof(float);
    size_t out_elems = static_cast<size_t>(batch) * channels * out_h * out_w;
    size_t out_bytes = out_elems * sizeof(float);
    size_t mask_bytes = out_elems * sizeof(int);

    float *d_x, *d_out;
    int* d_mask;

    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, mask_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));

    int total = static_cast<int>(out_elems);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    g16Maxpool2dForwardKernel<<<blocks, threads>>>(
        d_x,
        d_out,
        d_mask,
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        pool,
        stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mask, d_mask, mask_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_mask);
}

void g16Maxpool2dBackwardCuda(
    const float* h_dout,
    const int* h_mask,
    float* h_dx,
    int out_n,
    int in_n) {
    size_t dout_bytes = static_cast<size_t>(out_n) * sizeof(float);
    size_t mask_bytes = static_cast<size_t>(out_n) * sizeof(int);
    size_t dx_bytes = static_cast<size_t>(in_n) * sizeof(float);

    float *d_dout, *d_dx;
    int* d_mask;

    CUDA_CHECK(cudaMalloc(&d_dout, dout_bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, mask_bytes));
    CUDA_CHECK(cudaMalloc(&d_dx, dx_bytes));

    CUDA_CHECK(cudaMemcpy(d_dout, h_dout, dout_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask, mask_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dx, 0, dx_bytes));

    int threads = 256;
    int blocks = (out_n + threads - 1) / threads;
    g16Maxpool2dBackwardKernel<<<blocks, threads>>>(d_dout, d_mask, d_dx, out_n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dx, d_dx, dx_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_dout);
    cudaFree(d_mask);
    cudaFree(d_dx);
}

void g16LinearForwardCuda(
    const float* h_x,
    const float* h_w,
    const float* h_b,
    float* h_out,
    int batch,
    int in_features,
    int out_features) {
    size_t x_bytes = static_cast<size_t>(batch) * in_features * sizeof(float);
    size_t w_bytes = static_cast<size_t>(out_features) * in_features * sizeof(float);
    size_t b_bytes = static_cast<size_t>(out_features) * sizeof(float);
    size_t out_bytes = static_cast<size_t>(batch) * out_features * sizeof(float);

    float *d_x, *d_w, *d_b, *d_out;

    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    int total = batch * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    g16LinearForwardKernel<<<blocks, threads>>>(
        d_x, d_w, d_b, d_out, batch, in_features, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

void g16LinearBackwardCuda(
    const float* h_dout,
    const float* h_x,
    const float* h_w,
    float* h_dx,
    float* h_dw,
    float* h_db,
    int batch,
    int in_features,
    int out_features) {
    size_t dout_bytes = static_cast<size_t>(batch) * out_features * sizeof(float);
    size_t x_bytes = static_cast<size_t>(batch) * in_features * sizeof(float);
    size_t w_bytes = static_cast<size_t>(out_features) * in_features * sizeof(float);
    size_t db_bytes = static_cast<size_t>(out_features) * sizeof(float);

    float *d_dout, *d_x, *d_w, *d_dx, *d_dw, *d_db;

    CUDA_CHECK(cudaMalloc(&d_dout, dout_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_dx, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_dw, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_db, db_bytes));

    CUDA_CHECK(cudaMemcpy(d_dout, h_dout, dout_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_dx, 0, x_bytes));
    CUDA_CHECK(cudaMemset(d_dw, 0, w_bytes));
    CUDA_CHECK(cudaMemset(d_db, 0, db_bytes));

    int total = batch * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    g16LinearBackwardKernel<<<blocks, threads>>>(
        d_dout,
        d_x,
        d_w,
        d_dx,
        d_dw,
        d_db,
        batch,
        in_features,
        out_features);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dx, d_dx, x_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dw, d_dw, w_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_db, d_db, db_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_dout);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_dx);
    cudaFree(d_dw);
    cudaFree(d_db);
}

}  // extern "C"
