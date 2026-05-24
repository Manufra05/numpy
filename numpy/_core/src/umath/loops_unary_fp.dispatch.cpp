#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include <cmath>

#include "simd/simd.hpp"
#include <hwy/highway.h>

namespace {
using namespace np::simd;

/*******************************************************************************
 ** 1. SIMD Universal Loops Engines
 ******************************************************************************/
#if NPY_HWY

#define NPY_GENERATE_SIMD_LOOP(NAME, SCALAR_FUNC, SIMD_FUNC)             \
template <typename T>                                                    \
HWY_ATTR SIMD_MSVC_NOINLINE                                              \
static void simd_unary_loop_##NAME(T* op, const T* ip, npy_intp len) {   \
    constexpr int UNROLL = 4;                                            \
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();                          \
    const int wstep = vstep * UNROLL;                                    \
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {       \
        for (int i = 0; i < UNROLL; i++) {                               \
            StoreU(SIMD_FUNC(LoadU(ip + vstep * i)), op + vstep * i);    \
        }                                                                \
    }                                                                    \
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {       \
        StoreU(SIMD_FUNC(LoadU(ip)), op);                                \
    }                                                                    \
    for (; len > 0; len--, ip++, op++) {                                 \
        *op = SCALAR_FUNC(*ip);                                          \
    }                                                                    \
}

// Native Google Highway routines
NPY_GENERATE_SIMD_LOOP(absolute,  std::abs,   hn::Abs)
NPY_GENERATE_SIMD_LOOP(square,    [](auto v) { return v*v; }, [](auto v) { return hn::Mul(v,v); })
NPY_GENERATE_SIMD_LOOP(reciprocal,[](auto v) { return 1.0/v; }, [](auto v) { return hn::Div(hn::Set(hn::DFromV<decltype(v)>(), 1.0), v); })
NPY_GENERATE_SIMD_LOOP(sqrt,      std::sqrt,  hn::Sqrt)
NPY_GENERATE_SIMD_LOOP(floor,     std::floor, hn::Floor)
NPY_GENERATE_SIMD_LOOP(ceil,      std::ceil,  hn::Ceil)
NPY_GENERATE_SIMD_LOOP(trunc,     std::trunc, hn::Trunc)
NPY_GENERATE_SIMD_LOOP(rint,      std::rint,  hn::Round)

#endif

/*******************************************************************************
 ** 2. Strides Manager
 ******************************************************************************/
template <typename T, void (*SIMD_FUNC)(T*, const T*, npy_intp), T (*SCALAR_FUNC)(T)>
static void execute_unary_ufunc(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    char *ip_c = args[0];
    char *op_c = args[1];
    npy_intp is = steps[0];
    npy_intp os = steps[1];
    npy_intp n = dimensions[0];

#if NPY_HWY
    if (is == sizeof(T) && os == sizeof(T) && !is_mem_overlap(ip_c, is, op_c, os, n)) {
        SIMD_FUNC(reinterpret_cast<T*>(op_c), reinterpret_cast<T*>(ip_c), n);
        return;
    }
#endif
    T *ip = reinterpret_cast<T*>(ip_c);
    T *op = reinterpret_cast<T*>(op_c);
    for (npy_intp i = 0; i < n; i++, ip = reinterpret_cast<T*>(reinterpret_cast<char*>(ip) + is), 
                                  op = reinterpret_cast<T*>(reinterpret_cast<char*>(op) + os)) {
        *op = SCALAR_FUNC(*ip);
    }
}

template <typename T> static HWY_INLINE T scalar_abs(T v)  { return std::abs(v); }
template <typename T> static HWY_INLINE T scalar_sq(T v)   { return v * v; }
template <typename T> static HWY_INLINE T scalar_rec(T v)  { return static_cast<T>(1.0) / v; }
template <typename T> static HWY_INLINE T scalar_sqrt(T v) { return std::sqrt(v); }
template <typename T> static HWY_INLINE T scalar_flr(T v)  { return std::floor(v); }
template <typename T> static HWY_INLINE T scalar_cel(T v)  { return std::ceil(v); }
template <typename T> static HWY_INLINE T scalar_trn(T v)  { return std::trunc(v); }
template <typename T> static HWY_INLINE T scalar_rnt(T v)  { return std::rint(v); }

} // namespace anonymous

/*******************************************************************************
 ** 3. Dinammic expansion of Numpy C-API dispatch
 ******************************************************************************/
#if NPY_HWY
  #define EXPAND_SIMD_FUNC(kind) simd_unary_loop_##kind
#else
  #define EXPAND_SIMD_FUNC(kind) nullptr
#endif

#define NPY_CREATE_UNARY_UFUNC(TYPE, ctype, kind, SCALAR, IS_ABS)                                  \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##kind)(                                          \
    char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))        \
{                                                                                                  \
    execute_unary_ufunc<ctype, EXPAND_SIMD_FUNC(kind), SCALAR>(args, dimensions, steps);           \
    if (sizeof(ctype) == sizeof(float) && IS_ABS) {                                                \
        npy_clear_floatstatus_barrier(reinterpret_cast<char*>(const_cast<npy_intp*>(dimensions))); \
    }                                                                                              \
}

#define NPY_EXPAND_FP_LOOPS(TYPE, ctype)                                   \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, absolute,   scalar_abs,  1)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, square,     scalar_sq,   0)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, reciprocal, scalar_rec,  0)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, sqrt,       scalar_sqrt, 0)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, floor,      scalar_flr,  0)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, ceil,       scalar_cel,  0)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, trunc,      scalar_trn,  0)        \
    NPY_CREATE_UNARY_UFUNC(TYPE, ctype, rint,       scalar_rnt,  0)

NPY_EXPAND_FP_LOOPS(FLOAT, float)
NPY_EXPAND_FP_LOOPS(DOUBLE, double)

#undef NPY_EXPAND_FP_LOOPS
#undef NPY_CREATE_UNARY_UFUNC