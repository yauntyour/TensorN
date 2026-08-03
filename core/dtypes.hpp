#pragma once
#ifndef __TENSORN_DTYPES_HPP__
#define __TENSORN_DTYPES_HPP__

// ============================================================================
// TensorN low-precision data types for accelerated compute.
//
// Provides self-contained (host + device) implementations of:
//   * half          - IEEE 754 binary16  (FP16),   2 bytes
//   * bfloat16      - Google bfloat16,             2 bytes
//   * tf32          - TF32 storage type (10-bit mantissa float), 4 bytes
//   * fp8_e4m3      - NVIDIA FP8 (4 exp / 3 mantissa, bias 7),    1 byte
//   * fp8_e5m2      - NVIDIA FP8 (5 exp / 2 mantissa, bias 15),   1 byte
//
// All types are trivially copyable and bit-layout-compatible with the CUDA
// types __half / __nv_bfloat16 / __nv_fp8_e4m3 / __nv_fp8_e5m2, so device
// buffers can be passed to cuBLAS via reinterpret_cast.
//
// Arithmetic of all types is performed in float and rounded once per
// operation (round-to-nearest-even), which is safe to use in CUDA kernels.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <ostream>
#include <type_traits>

#ifndef __host__
#define __host__
#define __device__
#define __forceinline__ inline
#endif

namespace TensorN
{
    // ================================================================
    // Bit-level conversion helpers (host + device compatible)
    // ================================================================
    namespace fp
    {
        __host__ __device__ __forceinline__ uint32_t float_to_bits(float f)
        {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        }

        __host__ __device__ __forceinline__ float bits_to_float(uint32_t u)
        {
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }

        // -------- IEEE binary16 (FP16) --------

        // float -> FP16 bits, round-to-nearest-even
        __host__ __device__ __forceinline__ uint16_t float_to_half_bits(float f)
        {
            const uint32_t x = float_to_bits(f);
            const uint32_t sign = (x >> 16) & 0x8000u;
            const uint32_t exp = (x >> 23) & 0xFFu;
            uint32_t mant = x & 0x7FFFFFu;

            if (exp == 0xFFu) // NaN / Inf
                return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));

            const int32_t e = static_cast<int32_t>(exp) - 127 + 15;
            if (e >= 31) // overflow -> Inf
                return static_cast<uint16_t>(sign | 0x7C00u);

            if (e <= 0)
            {
                if (e < -10) // underflow -> 0
                    return static_cast<uint16_t>(sign);
                mant |= 0x800000u; // implicit leading bit
                const uint32_t shift = static_cast<uint32_t>(14 - e);
                uint32_t h = mant >> shift;
                const uint32_t rem = mant & ((1u << shift) - 1u);
                const uint32_t halfway = 1u << (shift - 1);
                if (rem > halfway || (rem == halfway && (h & 1u)))
                    ++h;
                return static_cast<uint16_t>(sign | h);
            }

            uint32_t h = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
            const uint32_t rem = mant & 0x1FFFu;
            if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
                ++h;
            return static_cast<uint16_t>(sign | h);
        }

        // FP16 bits -> float
        __host__ __device__ __forceinline__ float half_bits_to_float(uint16_t h)
        {
            const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
            const uint32_t exp = (h >> 10) & 0x1Fu;
            const uint32_t mant = h & 0x3FFu;
            uint32_t f;
            if (exp == 0u)
            {
                if (mant == 0u)
                    f = sign;
                else
                {
                    uint32_t e = 0u;
                    uint32_t m = mant;
                    while (!(m & 0x400u)) { ++e; m <<= 1; }
                    f = sign | ((113u - e) << 23) | ((m & 0x3FFu) << 13);
                }
            }
            else if (exp == 31u)
                f = sign | 0x7F800000u | (mant << 13);
            else
                f = sign | ((exp + 112u) << 23) | (mant << 13);
            return bits_to_float(f);
        }

        // -------- bfloat16 --------

        // float -> bfloat16 bits, round-to-nearest-even (standard BF16 formula)
        __host__ __device__ __forceinline__ uint16_t float_to_bfloat16_bits(float f)
        {
            uint32_t x = float_to_bits(f);
            const uint32_t rounding_bias = ((x >> 16) & 1u) + 0x7FFFu;
            x += rounding_bias;
            return static_cast<uint16_t>(x >> 16);
        }

        // bfloat16 bits -> float
        __host__ __device__ __forceinline__ float bfloat16_bits_to_float(uint16_t b)
        {
            return bits_to_float(static_cast<uint32_t>(b) << 16);
        }

        // -------- TF32 (storage: float, mantissa truncated to 10 bits) --------

        __host__ __device__ __forceinline__ float tf32_round(float f)
        {
            uint32_t x = float_to_bits(f);
            if ((x & 0x7F800000u) == 0x7F800000u) // preserve Inf / NaN
                return f;
            const uint32_t rounding_bias = ((x >> 13) & 1u) + 0xFFFu;
            x += rounding_bias;
            x &= 0xFFFFE000u; // keep 10 mantissa bits
            return bits_to_float(x);
        }

        // -------- FP8 E4M3 (1 sign, 4 exp, 3 mantissa, bias 7, no subnormals) --------

        // float -> FP8 e4m3 bits, round-to-nearest-even, clamp to [-448, 448]
        __host__ __device__ __forceinline__ uint8_t float_to_fp8_e4m3_bits(float f)
        {
            const uint32_t x = float_to_bits(f);
            const uint8_t sign = static_cast<uint8_t>((x >> 24) & 0x80u);
            const uint32_t exp = (x >> 23) & 0xFFu;
            const uint32_t mant = x & 0x7FFFFFu;

            if (exp == 0xFFu) // NaN / Inf -> NaN
                return 0x7Fu;
            if (x == 0u)
                return 0u;

            const int32_t e = static_cast<int32_t>(exp) - 127 + 7;
            if (e >= 15) // overflow -> clamp to max 448 (0x7E, NaN encoding excluded)
                return static_cast<uint8_t>(sign | 0x7Eu);
            if (e <= 0)
            {
                // no subnormals: round to min normal 2^-6 (0x04) or zero
                if (e == 0 && mant >= 0x400000u)
                    return static_cast<uint8_t>(sign | 0x04u);
                return sign;
            }

            uint32_t h = (static_cast<uint32_t>(e) << 3) | (mant >> 20);
            const uint32_t rem = mant & 0xFFFFFu;
            if (rem > 0x80000u || (rem == 0x80000u && (h & 1u)))
                ++h;
            if (h >= 0x7Fu) // rounding overflow into NaN encoding -> clamp
                return static_cast<uint8_t>(sign | 0x7Eu);
            return static_cast<uint8_t>(sign | h);
        }

        // FP8 e4m3 bits -> float
        __host__ __device__ __forceinline__ float fp8_e4m3_bits_to_float(uint8_t b)
        {
            const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
            const uint32_t exp = (b >> 3) & 0xFu;
            const uint32_t mant = b & 0x7u;
            if (exp == 0xFu) // NaN
                return bits_to_float(sign | 0x7FC00000u);
            if (exp == 0u) // only zero (no subnormals)
                return bits_to_float(sign);
            return bits_to_float(sign | ((exp + 120u) << 23) | (mant << 20));
        }

        // -------- FP8 E5M2 (1 sign, 5 exp, 2 mantissa, bias 15, subnormals) --------

        // float -> FP8 e5m2 bits, round-to-nearest-even, clamp to [-57344, 57344]
        __host__ __device__ __forceinline__ uint8_t float_to_fp8_e5m2_bits(float f)
        {
            const uint32_t x = float_to_bits(f);
            const uint8_t sign = static_cast<uint8_t>((x >> 24) & 0x80u);
            const uint32_t exp = (x >> 23) & 0xFFu;
            uint32_t mant = x & 0x7FFFFFu;

            if (exp == 0xFFu)
            {
                if (mant == 0u) // Inf
                    return static_cast<uint8_t>(sign | 0x7Cu);
                return static_cast<uint8_t>(sign | 0x7Eu); // NaN
            }
            if (x == 0u)
                return 0u;

            const int32_t e = static_cast<int32_t>(exp) - 127 + 15;
            if (e >= 31) // overflow -> Inf
                return static_cast<uint8_t>(sign | 0x7Cu);

            if (e >= 1)
            {
                uint32_t h = (static_cast<uint32_t>(e) << 2) | (mant >> 21);
                const uint32_t rem = mant & 0x1FFFFFu;
                if (rem > 0x100000u || (rem == 0x100000u && (h & 1u)))
                    ++h;
                if (h >= 0x7Cu) // rounding overflow -> Inf
                    return static_cast<uint8_t>(sign | 0x7Cu);
                return static_cast<uint8_t>(sign | h);
            }

            // subnormal / zero: value = (0x800000|mant) * 2^(e-38)
            if (e >= -1)
            {
                mant |= 0x800000u;
                const uint32_t shift = static_cast<uint32_t>(22 - e);
                uint32_t h = mant >> shift;
                const uint32_t rem = mant & ((1u << shift) - 1u);
                const uint32_t halfway = 1u << (shift - 1);
                if (rem > halfway || (rem == halfway && (h & 1u)))
                    ++h;
                return static_cast<uint8_t>(sign | h);
            }
            // round to min subnormal 2^-16 (0x01) or zero
            if (mant >= 0x800000u)
                return static_cast<uint8_t>(sign | 0x01u);
            return sign;
        }

        // FP8 e5m2 bits -> float
        __host__ __device__ __forceinline__ float fp8_e5m2_bits_to_float(uint8_t b)
        {
            const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
            const uint32_t exp = (b >> 2) & 0x1Fu;
            const uint32_t mant = b & 0x3u;
            if (exp == 0x1Fu)
                return bits_to_float(sign | 0x7F800000u | (mant << 21));
            if (exp == 0u)
            {
                if (mant == 0u)
                    return bits_to_float(sign);
                uint32_t lead = 0u;
                uint32_t m = mant;
                while (m > 1u) { m >>= 1; ++lead; }
                const uint32_t frac = (mant - (1u << lead)) << (23u - lead);
                const uint32_t e2 = 127u + lead - 16u;
                return bits_to_float(sign | (e2 << 23) | frac);
            }
            return bits_to_float(sign | ((exp + 112u) << 23) | (mant << 21));
        }
    } // namespace fp

    // ================================================================
    // Common arithmetic / comparison operator set for low-precision types
    // ================================================================

#define TENSORN_LOWP_ARITH(TYPE)                                                        \
    __host__ __device__ TYPE& operator+=(const TYPE& r) { *this = TYPE(float(*this) + float(r)); return *this; } \
    __host__ __device__ TYPE& operator-=(const TYPE& r) { *this = TYPE(float(*this) - float(r)); return *this; } \
    __host__ __device__ TYPE& operator*=(const TYPE& r) { *this = TYPE(float(*this) * float(r)); return *this; } \
    __host__ __device__ TYPE& operator/=(const TYPE& r) { *this = TYPE(float(*this) / float(r)); return *this; } \
    __host__ __device__ TYPE& operator+=(float r) { *this = TYPE(float(*this) + r); return *this; } \
    __host__ __device__ TYPE& operator-=(float r) { *this = TYPE(float(*this) - r); return *this; } \
    __host__ __device__ TYPE& operator*=(float r) { *this = TYPE(float(*this) * r); return *this; } \
    __host__ __device__ TYPE& operator/=(float r) { *this = TYPE(float(*this) / r); return *this; } \
    __host__ __device__ TYPE operator-() const { return TYPE(-float(*this)); } \
    __host__ __device__ friend TYPE operator+(const TYPE& a, const TYPE& b) { return TYPE(float(a) + float(b)); } \
    __host__ __device__ friend TYPE operator-(const TYPE& a, const TYPE& b) { return TYPE(float(a) - float(b)); } \
    __host__ __device__ friend TYPE operator*(const TYPE& a, const TYPE& b) { return TYPE(float(a) * float(b)); } \
    __host__ __device__ friend TYPE operator/(const TYPE& a, const TYPE& b) { return TYPE(float(a) / float(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator+(const TYPE& a, U b) { return TYPE(float(a) + static_cast<float>(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator-(const TYPE& a, U b) { return TYPE(float(a) - static_cast<float>(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator*(const TYPE& a, U b) { return TYPE(float(a) * static_cast<float>(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator/(const TYPE& a, U b) { return TYPE(float(a) / static_cast<float>(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator+(U a, const TYPE& b) { return TYPE(static_cast<float>(a) + float(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator-(U a, const TYPE& b) { return TYPE(static_cast<float>(a) - float(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator*(U a, const TYPE& b) { return TYPE(static_cast<float>(a) * float(b)); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend TYPE operator/(U a, const TYPE& b) { return TYPE(static_cast<float>(a) / float(b)); } \
    __host__ __device__ friend bool operator==(const TYPE& a, const TYPE& b) { return float(a) == float(b); } \
    __host__ __device__ friend bool operator!=(const TYPE& a, const TYPE& b) { return float(a) != float(b); } \
    __host__ __device__ friend bool operator< (const TYPE& a, const TYPE& b) { return float(a) <  float(b); } \
    __host__ __device__ friend bool operator<=(const TYPE& a, const TYPE& b) { return float(a) <= float(b); } \
    __host__ __device__ friend bool operator> (const TYPE& a, const TYPE& b) { return float(a) >  float(b); } \
    __host__ __device__ friend bool operator>=(const TYPE& a, const TYPE& b) { return float(a) >= float(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator==(const TYPE& a, U b) { return float(a) == static_cast<float>(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator==(U a, const TYPE& b) { return static_cast<float>(a) == float(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator!=(const TYPE& a, U b) { return float(a) != static_cast<float>(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator!=(U a, const TYPE& b) { return static_cast<float>(a) != float(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator< (const TYPE& a, U b) { return float(a) <  static_cast<float>(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator< (U a, const TYPE& b) { return static_cast<float>(a) <  float(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator<=(const TYPE& a, U b) { return float(a) <= static_cast<float>(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator<=(U a, const TYPE& b) { return static_cast<float>(a) <= float(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator> (const TYPE& a, U b) { return float(a) >  static_cast<float>(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator> (U a, const TYPE& b) { return static_cast<float>(a) >  float(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator>=(const TYPE& a, U b) { return float(a) >= static_cast<float>(b); } \
    template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0> \
    __host__ __device__ friend bool operator>=(U a, const TYPE& b) { return static_cast<float>(a) >= float(b); } \
    friend std::ostream& operator<<(std::ostream& os, const TYPE& v) { return os << float(v); }

    // ================================================================
    // half — IEEE 754 binary16
    // ================================================================

    class half
    {
    private:
        uint16_t _v = 0;

    public:
        half() = default;
        __host__ __device__ half(float v) : _v(fp::float_to_half_bits(v)) {}
        __host__ __device__ half(double v) : _v(fp::float_to_half_bits(static_cast<float>(v))) {}
        template <typename U, std::enable_if_t<std::is_integral_v<U>, int> = 0>
        __host__ __device__ half(U v) : _v(fp::float_to_half_bits(static_cast<float>(v))) {}

        __host__ __device__ operator float() const { return fp::half_bits_to_float(_v); }
        __host__ __device__ explicit operator double() const { return static_cast<double>(fp::half_bits_to_float(_v)); }

        __host__ __device__ static constexpr half from_bits(uint16_t b) { half h; h._v = b; return h; }
        __host__ __device__ constexpr uint16_t bits() const { return _v; }

        TENSORN_LOWP_ARITH(half)
    };

    // ================================================================
    // bfloat16
    // ================================================================

    class bfloat16
    {
    private:
        uint16_t _v = 0;

    public:
        bfloat16() = default;
        __host__ __device__ bfloat16(float v) : _v(fp::float_to_bfloat16_bits(v)) {}
        __host__ __device__ bfloat16(double v) : _v(fp::float_to_bfloat16_bits(static_cast<float>(v))) {}
        template <typename U, std::enable_if_t<std::is_integral_v<U>, int> = 0>
        __host__ __device__ bfloat16(U v) : _v(fp::float_to_bfloat16_bits(static_cast<float>(v))) {}

        __host__ __device__ operator float() const { return fp::bfloat16_bits_to_float(_v); }
        __host__ __device__ explicit operator double() const { return static_cast<double>(fp::bfloat16_bits_to_float(_v)); }

        __host__ __device__ static constexpr bfloat16 from_bits(uint16_t b) { bfloat16 v; v._v = b; return v; }
        __host__ __device__ constexpr uint16_t bits() const { return _v; }

        TENSORN_LOWP_ARITH(bfloat16)
    };

    // ================================================================
    // tf32 — TF32 storage type (float with 10-bit mantissa)
    // ================================================================

    class tf32
    {
    private:
        float _v = 0.0f;

        struct raw_tag {};

        __host__ __device__ constexpr tf32(raw_tag, float v) : _v(v) {}

    public:
        tf32() = default;
        __host__ __device__ tf32(float v) : _v(fp::tf32_round(v)) {}
        __host__ __device__ tf32(double v) : _v(fp::tf32_round(static_cast<float>(v))) {}
        template <typename U, std::enable_if_t<std::is_integral_v<U>, int> = 0>
        __host__ __device__ tf32(U v) : _v(fp::tf32_round(static_cast<float>(v))) {}

        __host__ __device__ operator float() const { return _v; }
        __host__ __device__ explicit operator double() const { return static_cast<double>(_v); }

        __host__ __device__ static constexpr tf32 from_raw(float v) { return tf32(raw_tag{}, v); }
        __host__ __device__ constexpr float raw() const { return _v; }

        TENSORN_LOWP_ARITH(tf32)
    };

    // ================================================================
    // fp8_e4m3 — NVIDIA FP8 (4 exp / 3 mantissa, bias 7)
    // ================================================================

    class fp8_e4m3
    {
    private:
        uint8_t _v = 0;

    public:
        fp8_e4m3() = default;
        __host__ __device__ fp8_e4m3(float v) : _v(fp::float_to_fp8_e4m3_bits(v)) {}
        __host__ __device__ fp8_e4m3(double v) : _v(fp::float_to_fp8_e4m3_bits(static_cast<float>(v))) {}
        template <typename U, std::enable_if_t<std::is_integral_v<U>, int> = 0>
        __host__ __device__ fp8_e4m3(U v) : _v(fp::float_to_fp8_e4m3_bits(static_cast<float>(v))) {}

        __host__ __device__ operator float() const { return fp::fp8_e4m3_bits_to_float(_v); }
        __host__ __device__ explicit operator double() const { return static_cast<double>(fp::fp8_e4m3_bits_to_float(_v)); }

        __host__ __device__ static constexpr fp8_e4m3 from_bits(uint8_t b) { fp8_e4m3 v; v._v = b; return v; }
        __host__ __device__ constexpr uint8_t bits() const { return _v; }

        TENSORN_LOWP_ARITH(fp8_e4m3)
    };

    // ================================================================
    // fp8_e5m2 — NVIDIA FP8 (5 exp / 2 mantissa, bias 15)
    // ================================================================

    class fp8_e5m2
    {
    private:
        uint8_t _v = 0;

    public:
        fp8_e5m2() = default;
        __host__ __device__ fp8_e5m2(float v) : _v(fp::float_to_fp8_e5m2_bits(v)) {}
        __host__ __device__ fp8_e5m2(double v) : _v(fp::float_to_fp8_e5m2_bits(static_cast<float>(v))) {}
        template <typename U, std::enable_if_t<std::is_integral_v<U>, int> = 0>
        __host__ __device__ fp8_e5m2(U v) : _v(fp::float_to_fp8_e5m2_bits(static_cast<float>(v))) {}

        __host__ __device__ operator float() const { return fp::fp8_e5m2_bits_to_float(_v); }
        __host__ __device__ explicit operator double() const { return static_cast<double>(fp::fp8_e5m2_bits_to_float(_v)); }

        __host__ __device__ static constexpr fp8_e5m2 from_bits(uint8_t b) { fp8_e5m2 v; v._v = b; return v; }
        __host__ __device__ constexpr uint8_t bits() const { return _v; }

        TENSORN_LOWP_ARITH(fp8_e5m2)
    };

#undef TENSORN_LOWP_ARITH

    // Short aliases. Prefer these unqualified names in TU scopes that also
    // pull in CUDA (global `half`) or OpenBLAS (global `bfloat16`) headers,
    // where `TensorN::half` / `TensorN::bfloat16` must be fully qualified.
    using fp16 = half;
    using bf16 = bfloat16;

    // ================================================================
    // Traits & introspection
    // ================================================================

    enum class dtype_t : uint8_t
    {
        Unknown  = 0,
        Float16  = 1,
        BFloat16 = 2,
        TF32     = 3,
        FP8_E4M3 = 4,
        FP8_E5M2 = 5,
        Float32  = 6,
        Float64  = 7,
        Int8     = 8,
        Int16    = 9,
        Int32    = 10,
        Int64    = 11,
        UInt8    = 12,
    };

    template <typename T>
    struct dtype_trait
    {
        static constexpr dtype_t value = dtype_t::Unknown;
    };

    template <> struct dtype_trait<float>      { static constexpr dtype_t value = dtype_t::Float32; };
    template <> struct dtype_trait<double>     { static constexpr dtype_t value = dtype_t::Float64; };
    template <> struct dtype_trait<int8_t>     { static constexpr dtype_t value = dtype_t::Int8; };
    template <> struct dtype_trait<int16_t>    { static constexpr dtype_t value = dtype_t::Int16; };
    template <> struct dtype_trait<int32_t>    { static constexpr dtype_t value = dtype_t::Int32; };
    template <> struct dtype_trait<int64_t>    { static constexpr dtype_t value = dtype_t::Int64; };
    template <> struct dtype_trait<uint8_t>    { static constexpr dtype_t value = dtype_t::UInt8; };
    template <> struct dtype_trait<half>       { static constexpr dtype_t value = dtype_t::Float16; };
    template <> struct dtype_trait<bfloat16>   { static constexpr dtype_t value = dtype_t::BFloat16; };
    template <> struct dtype_trait<tf32>       { static constexpr dtype_t value = dtype_t::TF32; };
    template <> struct dtype_trait<fp8_e4m3>   { static constexpr dtype_t value = dtype_t::FP8_E4M3; };
    template <> struct dtype_trait<fp8_e5m2>   { static constexpr dtype_t value = dtype_t::FP8_E5M2; };

    template <typename T>
    constexpr dtype_t dtype_of() { return dtype_trait<T>::value; }

    template <typename T>
    inline const char* name_of()
    {
        if constexpr (std::is_same_v<T, float>)      return "float32";
        else if constexpr (std::is_same_v<T, double>) return "float64";
        else if constexpr (std::is_same_v<T, half>)   return "float16";
        else if constexpr (std::is_same_v<T, bfloat16>) return "bfloat16";
        else if constexpr (std::is_same_v<T, tf32>)   return "tf32";
        else if constexpr (std::is_same_v<T, fp8_e4m3>) return "fp8_e4m3";
        else if constexpr (std::is_same_v<T, fp8_e5m2>) return "fp8_e5m2";
        else if constexpr (std::is_same_v<T, int8_t>) return "int8";
        else if constexpr (std::is_same_v<T, int16_t>) return "int16";
        else if constexpr (std::is_same_v<T, int32_t>) return "int32";
        else if constexpr (std::is_same_v<T, int64_t>) return "int64";
        else if constexpr (std::is_same_v<T, uint8_t>) return "uint8";
        else return "unknown";
    }

    template <typename T>
    struct is_lowprecision : std::bool_constant<
        std::is_same_v<T, half> || std::is_same_v<T, bfloat16> ||
        std::is_same_v<T, tf32> || std::is_same_v<T, fp8_e4m3> ||
        std::is_same_v<T, fp8_e5m2>> {};

    template <typename T>
    inline constexpr bool is_lowprecision_v = is_lowprecision<T>::value;
} // namespace TensorN

// ================================================================
// std::numeric_limits specializations
// ================================================================

namespace std
{
    template <>
    class numeric_limits<TensorN::half>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = false;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -13;
        static constexpr int max_exponent = 16;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent10 = 4;
        static constexpr bool traps = false;
        static constexpr bool is_tiny = true;

        static constexpr TensorN::half min()            { return TensorN::half::from_bits(0x0400); } // 2^-14
        static constexpr TensorN::half lowest()         { return TensorN::half::from_bits(0xFBFF); } // -65504
        static constexpr TensorN::half max()            { return TensorN::half::from_bits(0x7BFF); } // 65504
        static constexpr TensorN::half epsilon()        { return TensorN::half::from_bits(0x1400); } // 2^-10
        static constexpr TensorN::half round_error()    { return TensorN::half::from_bits(0x3800); } // 0.5
        static constexpr TensorN::half infinity()       { return TensorN::half::from_bits(0x7C00); }
        static constexpr TensorN::half quiet_NaN()      { return TensorN::half::from_bits(0x7E00); }
        static constexpr TensorN::half signaling_NaN()  { return TensorN::half::from_bits(0x7E00); }
        static constexpr TensorN::half denorm_min()     { return TensorN::half::from_bits(0x0001); } // 2^-24
    };

    template <>
    class numeric_limits<TensorN::bfloat16>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = false;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 8;
        static constexpr int digits10 = 2;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -125;
        static constexpr int max_exponent = 129;
        static constexpr int min_exponent10 = -37;
        static constexpr int max_exponent10 = 38;
        static constexpr bool traps = false;
        static constexpr bool is_tiny = true;

        static constexpr TensorN::bfloat16 min()            { return TensorN::bfloat16::from_bits(0x0080); } // 2^-126
        static constexpr TensorN::bfloat16 lowest()         { return TensorN::bfloat16::from_bits(0xFF7F); } // -max
        static constexpr TensorN::bfloat16 max()            { return TensorN::bfloat16::from_bits(0x7F7F); } // ~3.39e38
        static constexpr TensorN::bfloat16 epsilon()        { return TensorN::bfloat16::from_bits(0x7800); } // 2^-7
        static constexpr TensorN::bfloat16 round_error()    { return TensorN::bfloat16::from_bits(0x3F00); } // 0.5
        static constexpr TensorN::bfloat16 infinity()       { return TensorN::bfloat16::from_bits(0x7F80); }
        static constexpr TensorN::bfloat16 quiet_NaN()      { return TensorN::bfloat16::from_bits(0x7FC0); }
        static constexpr TensorN::bfloat16 signaling_NaN()  { return TensorN::bfloat16::from_bits(0x7FC0); }
        static constexpr TensorN::bfloat16 denorm_min()     { return TensorN::bfloat16::from_bits(0x0001); } // 2^-133
    };

    template <>
    class numeric_limits<TensorN::tf32>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = false;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -125;
        static constexpr int max_exponent = 129;
        static constexpr int min_exponent10 = -37;
        static constexpr int max_exponent10 = 38;
        static constexpr bool traps = false;
        static constexpr bool is_tiny = true;

        static constexpr TensorN::tf32 min()            { return TensorN::tf32::from_raw(1.17549435e-38f); } // 2^-126
        static constexpr TensorN::tf32 lowest()         { return TensorN::tf32::from_raw(-3.40282347e+38f); }
        static constexpr TensorN::tf32 max()            { return TensorN::tf32::from_raw(3.40282347e+38f); }
        static constexpr TensorN::tf32 epsilon()        { return TensorN::tf32::from_raw(0.0009765625f); } // 2^-10
        static constexpr TensorN::tf32 round_error()    { return TensorN::tf32::from_raw(0.5f); }
        static constexpr TensorN::tf32 infinity()       { return TensorN::tf32::from_raw(std::numeric_limits<float>::infinity()); }
        static constexpr TensorN::tf32 quiet_NaN()      { return TensorN::tf32::from_raw(std::numeric_limits<float>::quiet_NaN()); }
        static constexpr TensorN::tf32 signaling_NaN()  { return TensorN::tf32::from_raw(std::numeric_limits<float>::quiet_NaN()); }
        static constexpr TensorN::tf32 denorm_min()     { return TensorN::tf32::from_raw(1.40129846e-45f); }
    };

    template <>
    class numeric_limits<TensorN::fp8_e4m3>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = false;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = false;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 4;
        static constexpr int digits10 = 1;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -5;
        static constexpr int max_exponent = 8;
        static constexpr int min_exponent10 = -2;
        static constexpr int max_exponent10 = 2;
        static constexpr bool traps = false;
        static constexpr bool is_tiny = false;

        static constexpr TensorN::fp8_e4m3 min()            { return TensorN::fp8_e4m3::from_bits(0x04); } // 2^-6
        static constexpr TensorN::fp8_e4m3 lowest()         { return TensorN::fp8_e4m3::from_bits(0xFE); } // -448
        static constexpr TensorN::fp8_e4m3 max()            { return TensorN::fp8_e4m3::from_bits(0x7E); } // 448
        static constexpr TensorN::fp8_e4m3 epsilon()        { return TensorN::fp8_e4m3::from_bits(0x20); } // 2^-3
        static constexpr TensorN::fp8_e4m3 round_error()    { return TensorN::fp8_e4m3::from_bits(0x30); } // 0.5
        static constexpr TensorN::fp8_e4m3 infinity()       { return TensorN::fp8_e4m3::from_bits(0x7F); } // NaN (no Inf)
        static constexpr TensorN::fp8_e4m3 quiet_NaN()      { return TensorN::fp8_e4m3::from_bits(0x7F); }
        static constexpr TensorN::fp8_e4m3 signaling_NaN()  { return TensorN::fp8_e4m3::from_bits(0x7F); }
        static constexpr TensorN::fp8_e4m3 denorm_min()     { return TensorN::fp8_e4m3::from_bits(0x04); } // no subnormals
    };

    template <>
    class numeric_limits<TensorN::fp8_e5m2>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = false;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 3;
        static constexpr int digits10 = 0;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -13;
        static constexpr int max_exponent = 16;
        static constexpr int min_exponent10 = -5;
        static constexpr int max_exponent10 = 4;
        static constexpr bool traps = false;
        static constexpr bool is_tiny = true;

        static constexpr TensorN::fp8_e5m2 min()            { return TensorN::fp8_e5m2::from_bits(0x04); } // 2^-14
        static constexpr TensorN::fp8_e5m2 lowest()         { return TensorN::fp8_e5m2::from_bits(0xFB); } // -57344
        static constexpr TensorN::fp8_e5m2 max()            { return TensorN::fp8_e5m2::from_bits(0x7B); } // 57344
        static constexpr TensorN::fp8_e5m2 epsilon()        { return TensorN::fp8_e5m2::from_bits(0x34); } // 2^-2
        static constexpr TensorN::fp8_e5m2 round_error()    { return TensorN::fp8_e5m2::from_bits(0x38); } // 0.5
        static constexpr TensorN::fp8_e5m2 infinity()       { return TensorN::fp8_e5m2::from_bits(0x7C); }
        static constexpr TensorN::fp8_e5m2 quiet_NaN()      { return TensorN::fp8_e5m2::from_bits(0x7E); }
        static constexpr TensorN::fp8_e5m2 signaling_NaN()  { return TensorN::fp8_e5m2::from_bits(0x7E); }
        static constexpr TensorN::fp8_e5m2 denorm_min()     { return TensorN::fp8_e5m2::from_bits(0x01); } // 2^-16
    };
} // namespace std

#endif // __TENSORN_DTYPES_HPP__
