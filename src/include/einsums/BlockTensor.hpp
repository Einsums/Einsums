#pragma once

#include "einsums/_Common.hpp"
#include "einsums/Tensor.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums)

template<typename T, size_t Rank>
struct BlockTensorView;

template<typename T, size_t Rank>
struct BlockTensor : public detail::TensorBase<T, Rank> {
  private:
    std::string  _name{"(Unnamed)"};
    Dim<Rank>    _dims;

    std::vector<Tensor<T, Rank>> _blocks;
    std::vector<Range> _ranges;

    template <typename T_, size_t Rank_>
    friend struct BlockTensorView;

    template <typename T_, size_t OtherRank>
    friend struct BlockTensor;

  public:
    BlockTensor() = default;
    
    BlockTensor(const BlockTensor &) = default;

    ~BlockTensor() = default;

    template<typename... Dims>
    BlockTensor(std::string name, Dims... dims);

    BlockTensor(const BlockTensorView<T, Rank> &other);

    void zero();

    void set_all(T value);

    std::vector<Tensor<T, Rank>> &blocks();
    const std::vector<Tensor<T, Rank>> &blocks() const;

    const Tensor<T, Rank> &block(int i) const;
    Tensor<T, Rank> &block(int i);

    template<typename TOther>
    bool is_compatible(const BlockTensor<TOther, Rank> &other) const;

    template<typename TOther>
    bool is_compatible(const BlockTensorView<TOther, Rank> &other) const;

    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    T *data(MultiIndex... index);

    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> const T &;

    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> T &;

    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    auto operator()(MultiIndex... index) -> BlockTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>;

    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> BlockTensorView<T, Rank>;

    auto operator=(const BlockTensor<T, Rank> &other) -> BlockTensor<T, Rank> &;

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const BlockTensor<TOther, Rank> &other) -> BlockTensor<T, Rank> &;

    template <typename TOther>
    auto operator=(const BlockTensorView<TOther, Rank> &other) -> BlockTensor<T, Rank> &;

    auto operator=(const T &fill_value) -> BlockTensor &;

    #define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b) -> Tensor<T, Rank> & {                                                                                    \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto begin     = _data.begin() + chunksize * tid;                                                                              \
            auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;                                         \
            EINSUMS_OMP_SIMD for (auto i = begin; i < end; i++) {                                                                          \
                (*i) OP b;                                                                                                                 \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    auto operator OP(const Tensor<T, Rank> &b) -> Tensor<T, Rank> & {                                                                      \
        if (size() != b.size()) {                                                                                                          \
            throw std::runtime_error(fmt::format("operator" EINSUMS_STRINGIFY(OP) " : tensors differ in size : {} {}", size(), b.size())); \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto abegin    = _data.begin() + chunksize * tid;                                                                              \
            auto bbegin    = b._data.begin() + chunksize * tid;                                                                            \
            auto aend      = (tid == omp_get_num_threads() - 1) ? _data.end() : abegin + chunksize;                                        \
            auto j         = bbegin;                                                                                                       \
            EINSUMS_OMP_SIMD for (auto i = abegin; i < aend; i++) {                                                                        \
                (*i) OP(*j++);                                                                                                             \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    [[nodiscard]] auto dim(int d) const -> size_t;

    auto dims() const -> Dim<Rank> { return _dims; }

    ALIAS_TEMPLATE_FUNCTION(shape, dims);

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void               set_name(const std::string &name) { _name = name; }

    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }
};

END_EINSUMS_NAMESPACE_HPP(einsums)