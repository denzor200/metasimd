#include <immintrin.h>
#include <stdio.h>
#include <functional>

template<typename T>
class TD;

void mulScalar(float* out, const float* in, const size_t size, const float scalar)
{
    for (size_t i = 0; i < size; ++i)
    {
        out[i] = scalar * in[i];
    }
}

void mulScalarSIMD(float* out, const float* in, const float* in2, const size_t size, const float scalar)
{
    for (size_t i = 0; i < size; i+=8)
    {
        __m256 acc = _mm256_load_ps(in + i);
        __m256 s = _mm256_load_ps(in2 + i);
        acc = _mm256_mul_ps(s, acc);
        s = _mm256_broadcast_ss(&scalar);
        acc = _mm256_mul_ps(s, acc);
        _mm256_store_ps(out + i, acc);
    }
}

namespace simd
{
    namespace detail
    {
        struct op_times_token_t {};

        template<typename TT, typename TO>
        struct operation_t
        {
            TT operator_;
            TO operand_;
        };

        template<typename T, typename ...Args>
        struct expression_t
        {
            T first;
            std::tuple<Args...> rest; // each Arg is operation
        };

        template<typename L, typename R>
        struct assign_expression_t
        {
            L l;
            R r;
        };

        template<typename T>
        using operation_times_t = operation_t<op_times_token_t, T>;
        using operation_times_float_t = operation_t<op_times_token_t, float>;
    }

    namespace detail
    {
        template<typename I, typename T>
        struct get_t
        {
            T t;

            template<typename T2, typename ...Args>
            auto operator= (expression_t<T2, Args...> r) const noexcept -> assign_expression_t<T, expression_t<T2, Args...>>
            {
                //TD<T> td;
                return { t, r };
            }
        };
    }

    namespace detail
    {
        template<typename T, typename ...Args>
        auto operator* (expression_t<T, Args...> v, float s) noexcept -> expression_t<T, Args..., operation_times_float_t>
        {
            using operation_t = operation_times_float_t;
            operation_t operation = { op_times_token_t{}, s };
            return { v.first, std::tuple_cat(v.rest, std::tuple(operation)) };
        }

        template<typename I, typename T>
        auto operator* (get_t<I, T> v, float s) noexcept -> expression_t<get_t<I, T>, operation_times_float_t>
        {
            using operation_t = operation_times_float_t;
            operation_t operation = { op_times_token_t{}, s };
            return { v, std::make_tuple(operation) };
        }

        template<typename I, typename T, typename I2, typename T2>
        auto operator* (get_t<I, T> lhs, get_t<I2, T2> rhs) noexcept -> expression_t<get_t<I, T>, operation_times_t<get_t<I2, T2>>>
        {
            using operation_t = operation_times_t<get_t<I2, T2>>;
            operation_t operation = { op_times_token_t{}, rhs };
            return { lhs, std::make_tuple(operation) };
        }
    }

    namespace detail
    {
        struct apply_t
        {
            size_t i;

            template<typename I, typename T>
            __m256 operator() (__m256 acc, operation_times_t<get_t<I, T>> v) noexcept
            {
                __m256 s = _mm256_load_ps(v.operand_.t + i);
                acc = _mm256_mul_ps(s, acc);
                return acc;
            }

            __m256 operator() (__m256 acc, operation_times_float_t v) noexcept
            {
                __m256 s = _mm256_broadcast_ss(&v.operand_);
                acc = _mm256_mul_ps(s, acc);
                return acc;
            }
        };
    }

    namespace placeholders
    {
        struct _1 {};
    }

    template<typename I, typename T>
    auto get(T t) noexcept -> detail::get_t<I, T>
    {
        return { t };
    }

    template<typename It, typename F>
    void for_each(It b, It e, F f) noexcept
    {
        for (size_t i = b; i < e; i += 8)
        {
            detail::apply_t apply{ i };
            __m256 acc = _mm256_load_ps(f.r.first.t + i);

            std::apply
            (
                [&](auto const&... tupleArgs)
                {
                    ((acc = apply(acc, tupleArgs)), ...);
                }, f.r.rest
            );

            _mm256_store_ps(f.l + i, acc);
        }
    }

}



using TOP = simd::detail::operation_t<simd::detail::op_times_token_t, float>;
using TEX = simd::detail::expression_t<float, TOP, TOP>;

TEX ex = { 10, std::make_tuple(simd::detail::operation_t<simd::detail::op_times_token_t, float>{{}, 20.f}, simd::detail::operation_t<simd::detail::op_times_token_t, float>{simd::detail::op_times_token_t{}, 30.f}) };





#if 0
namespace simd
{
    namespace detail
    {

        template<typename T>
        struct mul_scalar_expression_t
        {
            float scalar;
            T val;
        };

        template<typename L, typename R>
        struct assign_expression_t
        {
            L l;
            R r;
        };

        template<typename I, typename T>
        struct get_t
        {
            T t;

            template<typename T2>
            auto operator= (mul_scalar_expression_t<T2> r) const noexcept -> assign_expression_t<T, mul_scalar_expression_t<T2>>
            {
                return { t, r };
            }
        };

        template<typename I, typename T>
        auto operator* (get_t<I, T> v, float s) noexcept -> mul_scalar_expression_t<get_t<I, T>>
        {
            return {s, v};
        }

        
    }


    namespace placeholders
    {
        struct _1 {};
    }

    template<typename I, typename T>
    auto get(T t) noexcept -> detail::get_t<I, T>
    {
        return { t };

    }

    template<typename It, typename F>
    void for_each(It b, It e, F f) noexcept
    {
        __m256 s = _mm256_broadcast_ss(&f.r.scalar);
        for (size_t i = b; i < e; i += 8)
        {
            __m256 v = _mm256_load_ps(f.r.val.t + i);
            __m256 r = _mm256_mul_ps(s, v);
            _mm256_store_ps(f.l + i, r);
        }
    }

}
#endif // 0

void mulScalarBetter(float* out, const float* in, const float* in2, const size_t size, const float scalar)
{
    using namespace simd;
    using namespace simd::placeholders;

    for_each((size_t)0, size, 
        get<_1>(out) = get<_1>(in) * get<_1>(in2) * scalar
    );
}


int main() {
    float a[16] = {
        1.20708f, 4.27732f, 4.4424f, 1.5752f, 2.01787f, 4.41157f, 1.16588f, 1.21913f,
        1.22544f, 2.39342f, 3.55537f, 3.29319f, 1.3999f, 3.46275f, 3.78729f, 3.17829f
    };
    float a2[16] = {
        1.20708f, 4.27732f, 4.4424f, 1.5752f, 2.01787f, 4.41157f, 1.16588f, 1.21913f,
        1.22544f, 2.39342f, 3.55537f, 3.29319f, 1.3999f, 3.46275f, 3.78729f, 3.17829f
    };
    float b[16] ;
    mulScalarBetter(b, a, a2, 16, 2.f);

    for (size_t i = 0; i < 8; ++i)
        printf("%f ", b[i]);

    printf("\n");

    for (size_t i = 0; i < 8; ++i)
        printf("%f ", b[i+8]);

    return 0;
}
