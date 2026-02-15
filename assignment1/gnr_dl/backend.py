from __future__ import annotations

import ctypes
import os
from typing import List, Tuple


class Backend:
    def __init__(self) -> None:
        self.g16Lib = None
        self.g16CudaEnabled = False
        self.G16LoadCudaBackend()

    @property
    def G16UsingCuda(self) -> bool:
        return self.g16CudaEnabled

    # Backward-compatible naming aliases
    @property
    def using_cuda(self) -> bool:
        return self.G16UsingCuda

    @property
    def usingCuda(self) -> bool:
        return self.G16UsingCuda

    def G16LoadCudaBackend(self) -> None:
        if os.environ.get("GNR_FORCE_PYTHON", "").strip() == "1":
            return

        env_path = os.environ.get("GNR_CUDA_LIB", "").strip()
        default_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "cuda_backend", "libgnr_cuda_backend.so")
        )
        candidate_paths = [p for p in [env_path, default_path] if p]

        for path in candidate_paths:
            if not os.path.exists(path):
                continue
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            try:
                self.G16ConfigureCudaSignatures(lib)
            except AttributeError:
                continue
            if not self.G16CudaRuntimeAvailable(lib):
                continue
            self.g16Lib = lib
            self.g16CudaEnabled = True
            return

    def G16ConfigureCudaSignatures(self, lib: ctypes.CDLL) -> None:
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_int = ctypes.c_int

        lib.g16Conv2dForwardCuda.argtypes = [
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        lib.g16Conv2dForwardCuda.restype = None

        lib.g16Conv2dBackwardCuda.argtypes = [
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        lib.g16Conv2dBackwardCuda.restype = None

        lib.g16ReluForwardCuda.argtypes = [c_float_p, c_float_p, c_int]
        lib.g16ReluBackwardCuda.argtypes = [c_float_p, c_float_p, c_float_p, c_int]
        lib.g16ReluForwardCuda.restype = None
        lib.g16ReluBackwardCuda.restype = None

        lib.g16Maxpool2dForwardCuda.argtypes = [
            c_float_p,
            c_float_p,
            c_int_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        lib.g16Maxpool2dForwardCuda.restype = None

        lib.g16Maxpool2dBackwardCuda.argtypes = [
            c_float_p,
            c_int_p,
            c_float_p,
            c_int,
            c_int,
        ]
        lib.g16Maxpool2dBackwardCuda.restype = None

        lib.g16LinearForwardCuda.argtypes = [
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_int,
            c_int,
            c_int,
        ]
        lib.g16LinearForwardCuda.restype = None

        lib.g16LinearBackwardCuda.argtypes = [
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_int,
            c_int,
            c_int,
        ]
        lib.g16LinearBackwardCuda.restype = None

    @staticmethod
    def G16CudaRuntimeAvailable(lib: ctypes.CDLL) -> bool:
        if not hasattr(lib, "g16CudaBackendAvailable"):
            return False

        try:
            lib.g16CudaBackendAvailable.argtypes = []
            lib.g16CudaBackendAvailable.restype = ctypes.c_int
            return int(lib.g16CudaBackendAvailable()) == 1
        except Exception:
            return False

    @staticmethod
    def G16ToCFloatArray(values: List[float]):
        return (ctypes.c_float * len(values))(*values)

    @staticmethod
    def G16ToCIntArray(values: List[int]):
        return (ctypes.c_int * len(values))(*values)

    @staticmethod
    def G16FromCFloatArray(values, length: int) -> List[float]:
        return [float(values[i]) for i in range(length)]

    @staticmethod
    def G16FromCIntArray(values, length: int) -> List[int]:
        return [int(values[i]) for i in range(length)]

    def G16Conv2dForward(
        self,
        x: List[float],
        w: List[float],
        b: List[float],
        batch: int,
        in_c: int,
        in_h: int,
        in_w: int,
        out_c: int,
        k_h: int,
        k_w: int,
        stride: int,
        padding: int,
    ) -> List[float]:
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        out_size = batch * out_c * out_h * out_w

        if self.g16CudaEnabled:
            x_c = self.G16ToCFloatArray(x)
            w_c = self.G16ToCFloatArray(w)
            b_c = self.G16ToCFloatArray(b)
            out_c_arr = (ctypes.c_float * out_size)()

            self.g16Lib.g16Conv2dForwardCuda(
                x_c,
                w_c,
                b_c,
                out_c_arr,
                batch,
                in_c,
                in_h,
                in_w,
                out_c,
                k_h,
                k_w,
                stride,
                padding,
            )
            return self.G16FromCFloatArray(out_c_arr, out_size)

        return self.G16Conv2dForwardPy(
            x, w, b, batch, in_c, in_h, in_w, out_c, k_h, k_w, stride, padding
        )

    def G16Conv2dBackward(
        self,
        dout: List[float],
        x: List[float],
        w: List[float],
        batch: int,
        in_c: int,
        in_h: int,
        in_w: int,
        out_c: int,
        k_h: int,
        k_w: int,
        stride: int,
        padding: int,
    ) -> Tuple[List[float], List[float], List[float]]:
        x_size = len(x)
        w_size = len(w)
        b_size = out_c

        if self.g16CudaEnabled:
            dout_c = self.G16ToCFloatArray(dout)
            x_c = self.G16ToCFloatArray(x)
            w_c = self.G16ToCFloatArray(w)
            dx_c = (ctypes.c_float * x_size)()
            dw_c = (ctypes.c_float * w_size)()
            db_c = (ctypes.c_float * b_size)()

            self.g16Lib.g16Conv2dBackwardCuda(
                dout_c,
                x_c,
                w_c,
                dx_c,
                dw_c,
                db_c,
                batch,
                in_c,
                in_h,
                in_w,
                out_c,
                k_h,
                k_w,
                stride,
                padding,
            )
            return (
                self.G16FromCFloatArray(dx_c, x_size),
                self.G16FromCFloatArray(dw_c, w_size),
                self.G16FromCFloatArray(db_c, b_size),
            )

        return self.G16Conv2dBackwardPy(
            dout,
            x,
            w,
            batch,
            in_c,
            in_h,
            in_w,
            out_c,
            k_h,
            k_w,
            stride,
            padding,
        )

    def G16ReluForward(self, x: List[float]) -> List[float]:
        if self.g16CudaEnabled:
            x_c = self.G16ToCFloatArray(x)
            out_c = (ctypes.c_float * len(x))()
            self.g16Lib.g16ReluForwardCuda(x_c, out_c, len(x))
            return self.G16FromCFloatArray(out_c, len(x))
        return [value if value > 0.0 else 0.0 for value in x]

    def G16ReluBackward(self, dout: List[float], x: List[float]) -> List[float]:
        if self.g16CudaEnabled:
            dout_c = self.G16ToCFloatArray(dout)
            x_c = self.G16ToCFloatArray(x)
            dx_c = (ctypes.c_float * len(dout))()
            self.g16Lib.g16ReluBackwardCuda(dout_c, x_c, dx_c, len(dout))
            return self.G16FromCFloatArray(dx_c, len(dout))

        dx = [0.0 for _ in range(len(dout))]
        for idx in range(len(dout)):
            dx[idx] = dout[idx] if x[idx] > 0.0 else 0.0
        return dx

    def G16Maxpool2dForward(
        self,
        x: List[float],
        batch: int,
        channels: int,
        in_h: int,
        in_w: int,
        pool: int,
        stride: int,
    ) -> Tuple[List[float], List[int], int, int]:
        out_h = (in_h - pool) // stride + 1
        out_w = (in_w - pool) // stride + 1
        out_size = batch * channels * out_h * out_w

        if self.g16CudaEnabled:
            x_c = self.G16ToCFloatArray(x)
            out_c = (ctypes.c_float * out_size)()
            mask_c = (ctypes.c_int * out_size)()
            self.g16Lib.g16Maxpool2dForwardCuda(
                x_c, out_c, mask_c, batch, channels, in_h, in_w, pool, stride
            )
            return (
                self.G16FromCFloatArray(out_c, out_size),
                self.G16FromCIntArray(mask_c, out_size),
                out_h,
                out_w,
            )

        return self.G16Maxpool2dForwardPy(x, batch, channels, in_h, in_w, pool, stride)

    def G16Maxpool2dBackward(
        self,
        dout: List[float],
        mask: List[int],
        input_size: int,
    ) -> List[float]:
        if self.g16CudaEnabled:
            dout_c = self.G16ToCFloatArray(dout)
            mask_c = self.G16ToCIntArray(mask)
            dx_c = (ctypes.c_float * input_size)()
            self.g16Lib.g16Maxpool2dBackwardCuda(dout_c, mask_c, dx_c, len(dout), input_size)
            return self.G16FromCFloatArray(dx_c, input_size)

        dx = [0.0 for _ in range(input_size)]
        for idx, grad in enumerate(dout):
            dx[mask[idx]] += grad
        return dx

    def G16LinearForward(
        self,
        x: List[float],
        w: List[float],
        b: List[float],
        batch: int,
        in_features: int,
        out_features: int,
    ) -> List[float]:
        out_size = batch * out_features
        if self.g16CudaEnabled:
            x_c = self.G16ToCFloatArray(x)
            w_c = self.G16ToCFloatArray(w)
            b_c = self.G16ToCFloatArray(b)
            out_c = (ctypes.c_float * out_size)()
            self.g16Lib.g16LinearForwardCuda(
                x_c, w_c, b_c, out_c, batch, in_features, out_features
            )
            return self.G16FromCFloatArray(out_c, out_size)

        return self.G16LinearForwardPy(x, w, b, batch, in_features, out_features)

    def G16LinearBackward(
        self,
        dout: List[float],
        x: List[float],
        w: List[float],
        batch: int,
        in_features: int,
        out_features: int,
    ) -> Tuple[List[float], List[float], List[float]]:
        dx_size = batch * in_features
        dw_size = out_features * in_features
        db_size = out_features

        if self.g16CudaEnabled:
            dout_c = self.G16ToCFloatArray(dout)
            x_c = self.G16ToCFloatArray(x)
            w_c = self.G16ToCFloatArray(w)
            dx_c = (ctypes.c_float * dx_size)()
            dw_c = (ctypes.c_float * dw_size)()
            db_c = (ctypes.c_float * db_size)()
            self.g16Lib.g16LinearBackwardCuda(
                dout_c, x_c, w_c, dx_c, dw_c, db_c, batch, in_features, out_features
            )
            return (
                self.G16FromCFloatArray(dx_c, dx_size),
                self.G16FromCFloatArray(dw_c, dw_size),
                self.G16FromCFloatArray(db_c, db_size),
            )

        return self.G16LinearBackwardPy(dout, x, w, batch, in_features, out_features)

    @staticmethod
    def G16Conv2dForwardPy(
        x: List[float],
        w: List[float],
        b: List[float],
        batch: int,
        in_c: int,
        in_h: int,
        in_w: int,
        out_c: int,
        k_h: int,
        k_w: int,
        stride: int,
        padding: int,
    ) -> List[float]:
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        out = [0.0 for _ in range(batch * out_c * out_h * out_w)]

        for n_idx in range(batch):
            for oc_idx in range(out_c):
                for oh_idx in range(out_h):
                    for ow_idx in range(out_w):
                        acc = b[oc_idx]
                        for ic_idx in range(in_c):
                            for kh_idx in range(k_h):
                                in_y = oh_idx * stride + kh_idx - padding
                                if in_y < 0 or in_y >= in_h:
                                    continue
                                for kw_idx in range(k_w):
                                    in_x = ow_idx * stride + kw_idx - padding
                                    if in_x < 0 or in_x >= in_w:
                                        continue
                                    x_pos = (
                                        ((n_idx * in_c + ic_idx) * in_h + in_y) * in_w
                                        + in_x
                                    )
                                    w_pos = (
                                        ((oc_idx * in_c + ic_idx) * k_h + kh_idx) * k_w
                                        + kw_idx
                                    )
                                    acc += x[x_pos] * w[w_pos]
                        out_pos = (
                            ((n_idx * out_c + oc_idx) * out_h + oh_idx) * out_w + ow_idx
                        )
                        out[out_pos] = acc
        return out

    @staticmethod
    def G16Conv2dBackwardPy(
        dout: List[float],
        x: List[float],
        w: List[float],
        batch: int,
        in_c: int,
        in_h: int,
        in_w: int,
        out_c: int,
        k_h: int,
        k_w: int,
        stride: int,
        padding: int,
    ) -> Tuple[List[float], List[float], List[float]]:
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1

        dx = [0.0 for _ in range(batch * in_c * in_h * in_w)]
        dw = [0.0 for _ in range(out_c * in_c * k_h * k_w)]
        db = [0.0 for _ in range(out_c)]

        for n_idx in range(batch):
            for oc_idx in range(out_c):
                for oh_idx in range(out_h):
                    for ow_idx in range(out_w):
                        dout_pos = (
                            ((n_idx * out_c + oc_idx) * out_h + oh_idx) * out_w + ow_idx
                        )
                        grad_val = dout[dout_pos]
                        db[oc_idx] += grad_val

                        for ic_idx in range(in_c):
                            for kh_idx in range(k_h):
                                in_y = oh_idx * stride + kh_idx - padding
                                if in_y < 0 or in_y >= in_h:
                                    continue
                                for kw_idx in range(k_w):
                                    in_x = ow_idx * stride + kw_idx - padding
                                    if in_x < 0 or in_x >= in_w:
                                        continue

                                    x_pos = (
                                        ((n_idx * in_c + ic_idx) * in_h + in_y) * in_w
                                        + in_x
                                    )
                                    w_pos = (
                                        ((oc_idx * in_c + ic_idx) * k_h + kh_idx) * k_w
                                        + kw_idx
                                    )
                                    dw[w_pos] += x[x_pos] * grad_val
                                    dx[x_pos] += w[w_pos] * grad_val

        return dx, dw, db

    @staticmethod
    def G16Maxpool2dForwardPy(
        x: List[float],
        batch: int,
        channels: int,
        in_h: int,
        in_w: int,
        pool: int,
        stride: int,
    ) -> Tuple[List[float], List[int], int, int]:
        out_h = (in_h - pool) // stride + 1
        out_w = (in_w - pool) // stride + 1

        out = [0.0 for _ in range(batch * channels * out_h * out_w)]
        mask = [0 for _ in range(batch * channels * out_h * out_w)]

        for n_idx in range(batch):
            for c_idx in range(channels):
                for oh_idx in range(out_h):
                    for ow_idx in range(out_w):
                        best_val = float("-inf")
                        best_index = 0
                        for ph_idx in range(pool):
                            for pw_idx in range(pool):
                                in_y = oh_idx * stride + ph_idx
                                in_x = ow_idx * stride + pw_idx
                                in_pos = (
                                    ((n_idx * channels + c_idx) * in_h + in_y) * in_w
                                    + in_x
                                )
                                val = x[in_pos]
                                if val > best_val:
                                    best_val = val
                                    best_index = in_pos

                        out_pos = (
                            ((n_idx * channels + c_idx) * out_h + oh_idx) * out_w
                            + ow_idx
                        )
                        out[out_pos] = best_val
                        mask[out_pos] = best_index

        return out, mask, out_h, out_w

    @staticmethod
    def G16LinearForwardPy(
        x: List[float],
        w: List[float],
        b: List[float],
        batch: int,
        in_features: int,
        out_features: int,
    ) -> List[float]:
        out = [0.0 for _ in range(batch * out_features)]
        for n_idx in range(batch):
            for out_idx in range(out_features):
                acc = b[out_idx]
                for in_idx in range(in_features):
                    x_pos = n_idx * in_features + in_idx
                    w_pos = out_idx * in_features + in_idx
                    acc += x[x_pos] * w[w_pos]
                out[n_idx * out_features + out_idx] = acc
        return out

    @staticmethod
    def G16LinearBackwardPy(
        dout: List[float],
        x: List[float],
        w: List[float],
        batch: int,
        in_features: int,
        out_features: int,
    ) -> Tuple[List[float], List[float], List[float]]:
        dx = [0.0 for _ in range(batch * in_features)]
        dw = [0.0 for _ in range(out_features * in_features)]
        db = [0.0 for _ in range(out_features)]

        for n_idx in range(batch):
            for out_idx in range(out_features):
                grad_val = dout[n_idx * out_features + out_idx]
                db[out_idx] += grad_val
                for in_idx in range(in_features):
                    x_pos = n_idx * in_features + in_idx
                    w_pos = out_idx * in_features + in_idx
                    dx[x_pos] += grad_val * w[w_pos]
                    dw[w_pos] += grad_val * x[x_pos]

        return dx, dw, db


g16Backend = Backend()


def G16GetBackend() -> Backend:
    return g16Backend


get_backend = G16GetBackend
