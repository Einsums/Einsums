#pragma once

#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#include <type_traits>

#include "Einsums/Concepts/Tensor.hpp"

namespace einsums {

// forward declaration.
template <typename T, size_t Rank, size_t BaseRank>
struct FunctionTensorView;

#ifdef __HIP__
template <typename T, size_t Rank, size_t BaseRank>
struct DeviceFunctionTensorView;
#endif

/**
 * @struct FunctionTensor
 *
 * @brief Base class for function tensors.
 *
 * A function tensor is one which takes advantage of the use of the operator() to index tensors to provide a way to
 * call a function. An example of this might be the Kronecker delta, which has such simple structure that creating
 * a whole new tensor object for it would be wasteful.
 */
namespace tensor_base {
template <typename T, size_t Rank>
struct FunctionTensor : public virtual Tensor<T, Rank>, virtual FunctionTensorNoExtra, virtual CoreTensor {
  protected:
    Dim<Rank>   _dims;
    std::string _name{"(unnamed)"};
    size_t      _size;

    /**
     * @brief Checks for negative indices and makes them positive, then performs range checking.
     */
    virtual void fix_indices(std::array<int, Rank> *inds) const {
        for (int i = 0; i < Rank; i++) {
            int orig = inds->at(i);
            if (inds->at(i) < 0) {
                inds->at(i) += _dims[i];
            }
            if (inds->at(i) >= _dims[i] || inds->at(i) < 0) {
                std::string message = fmt::format("Function tensor index out of range! Index at rank {} ", i);
                if (orig != inds->at(i)) {
                    message = fmt::format("{}({} -> {}) ", message, orig, inds->at(i));
                } else {
                    message = fmt::format("{}({}) ", message, inds->at(i));
                }
                throw EINSUMSEXCEPTION(fmt::format("{}is too far less than zero or is greater than {}", message, _dims[i]));
            }
        }
    }

  public:
    FunctionTensor()                       = default;
    FunctionTensor(FunctionTensor const &) = default;

    template <typename... Args>
        requires requires {
            requires(!std::is_same_v<Args, Dim<Rank>> || ...);
            requires sizeof...(Args) == Rank;
        }
    FunctionTensor(std::string name, Args... dims) : _dims{dims...}, _name{name} {
        _size = 1;

        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    FunctionTensor(std::string name, Dim<Rank> dims) : _dims(dims), _name{name} {
        _size = 1;

        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    FunctionTensor(Dim<Rank> dims) : _dims(dims) {
        _size = 1;

        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    virtual ~FunctionTensor() = default;

    /**
     * @brief Call the function.
     *
     * This is the method that should be overloaded in child classes to perform the actual
     * function call. Due to the inability to override methods with variable parameters,
     * this method with a set input type is the workaround.
     */
    virtual T call(std::array<int, Rank> const &inds) const = 0;

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v<MultiIndex> && ...);
        }
    T operator()(MultiIndex... inds) const {
        auto new_inds = std::array<int, Rank>{static_cast<int>(inds)...};

        fix_indices(&new_inds);

        return this->call(new_inds);
    }

    template <typename Storage>
        requires requires {
            requires !std::is_integral_v<Storage>;
            requires !std::is_same_v<Storage, AllT>;
            requires !std::is_same_v<Storage, Range>;
            requires !std::is_same_v<Storage, std::array<int, Rank>>;
        }
    T operator()(Storage const &inds) const {
        auto new_inds = std::array<int, Rank>();

        for (int i = 0; i < Rank; i++) {
            new_inds[i] = (int)inds.at(i);
        }

        fix_indices(&new_inds);

        return this->call(new_inds);
    }

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(AtLeastOneOfType<AllT, MultiIndex...>);
            requires(NoneOfType<std::array<int, Rank>, MultiIndex...>);
        }
    auto operator()(MultiIndex... inds) const
        -> FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank> {
        auto const &indices = std::forward_as_tuple(inds...);

        std::array<int, Rank> index_template;

        Offset<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> offsets;
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
                auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                index_template[i] = tmp;
            } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                dims[counter]     = _dims[i];
                offsets[counter]  = 0;
                index_template[i] = -1;
                counter++;

            } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                auto range       = std::get<i>(indices);
                offsets[counter] = range[0];
                if (range[1] < 0) {
                    auto temp = _dims[i] + range[1];
                    range[1]  = temp;
                }
                dims[counter]     = range[1] - range[0];
                index_template[i] = -1;
                counter++;
            }
        });

        return FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(
            this, offsets, dims, index_template);
    }

    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> FunctionTensorView<T, Rank, Rank> {
        Dim<Rank>             dims{};
        Offset<Rank>          offset{};
        std::array<int, Rank> index_template;

        auto ranges = get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; r < Rank; r++) {
            auto range = ranges[r];
            offset[r]  = range[0];
            if (range[1] < 0) {
                auto temp = _dims[r] + range[1];
                range[1]  = temp;
            }
            dims[r]           = range[1] - range[0];
            index_template[r] = -1;
        }

        return FunctionTensorView<T, Rank, Rank>{this, std::move(offset), std::move(dims), index_template};
    }

    virtual Dim<Rank> dims() const override { return _dims; }

    virtual auto dim(int d) const -> size_t override { return _dims[d]; }

    virtual std::string const &name() const override { return _name; }

    virtual void set_name(std::string const &str) override { _name = str; }

    operator Tensor<T, Rank>() const {
        Tensor<T, Rank> out(dims());
        out.set_name(name());

        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view        = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for default(none) shared(view, out)
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(out, *target_combination);
            target    = std::apply(*this, *target_combination);
        }

        return out;
    }
};

} // namespace tensor_base

template <typename T, size_t Rank>
struct FuncPointerTensor : public virtual tensor_base::FunctionTensor<T, Rank>, virtual tensor_base::CoreTensor {
  protected:
    T (*_func_ptr)(std::array<int, Rank> const &);

  public:
    template <typename... Args>
    FuncPointerTensor(std::string name, T (*func_ptr)(std::array<int, Rank> const &), Args... dims)
        : tensor_base::FunctionTensor<T, Rank>(name, dims...), _func_ptr(func_ptr) {}

    FuncPointerTensor(FuncPointerTensor<T, Rank> const &copy) : tensor_base::FunctionTensor<T, Rank>(copy) { _func_ptr = copy._func_ptr; }

    virtual ~FuncPointerTensor() = default;

    virtual T call(std::array<int, Rank> const &inds) const override { return _func_ptr(inds); }

    size_t dim(int d) const override { return tensor_base::FunctionTensor<T, Rank>::dim(d); }
};

template <typename T, size_t Rank, size_t UnderlyingRank>
struct FunctionTensorView : public virtual tensor_base::FunctionTensor<T, Rank>,
                            virtual tensor_base::TensorView<tensor_base::FunctionTensor<T, UnderlyingRank>> {
  protected:
    tensor_base::FunctionTensor<T, UnderlyingRank> const *_func_tensor;
    Offset<Rank>                                          _offsets;
    std::array<int, UnderlyingRank>                       _index_template;
    bool                                                  _full_view{true};

    virtual std::array<int, UnderlyingRank> apply_view(std::array<int, Rank> const &inds) const {
        std::array<int, UnderlyingRank> out{_index_template};
        int                             curr_rank = 0;
        for (int i = 0; i < Rank && curr_rank < UnderlyingRank; i++) {
            while (out.at(curr_rank) >= 0) {
                curr_rank++;
            }
            out.at(curr_rank) = inds.at(i);
            if (out.at(curr_rank) < 0) {
                out.at(curr_rank) += this->_dims[i];
            }
            if (out.at(curr_rank) >= this->_dims[i] || out.at(curr_rank) < 0) {
                throw EINSUMSEXCEPTION(
                    fmt::format("Function tensor view index out of range! Index of rank {} is {}, which is < 0 or >= {}.", i, inds.at(i),
                                this->_dims[i]));
            }

            out.at(curr_rank) += _offsets[i];
            curr_rank++;
        }

        return out;
    }

  public:
    FunctionTensorView() = default;
    FunctionTensorView(std::string name, tensor_base::FunctionTensor<T, UnderlyingRank> *func_tens, Offset<Rank> const &offsets,
                       Dim<Rank> const &dims, std::array<int, UnderlyingRank> const &index_template)
        : _offsets{offsets}, _func_tensor(func_tens), _index_template{index_template}, tensor_base::FunctionTensor<T, Rank>(name, dims) {
        if constexpr (Rank != UnderlyingRank) {
            _full_view = false;
        } else {
            for (int i = 0; i < UnderlyingRank; i++) {
                if (index_template.at(i) >= 0) {
                    _full_view = false;
                    break;
                }
                if (dims[i] != func_tens->dim(i)) {
                    _full_view = false;
                    break;
                }
                if (_offsets.at(i) != 0) {
                    _full_view = false;
                    break;
                }
            }
        }
    }

    FunctionTensorView(tensor_base::FunctionTensor<T, UnderlyingRank> const *func_tens, Offset<Rank> const &offsets, Dim<Rank> const &dims,
                       std::array<int, UnderlyingRank> const &index_template)
        : _offsets{offsets}, _func_tensor(func_tens), _index_template{index_template}, tensor_base::FunctionTensor<T, Rank>(dims) {}

    FunctionTensorView(FunctionTensorView const &copy) : tensor_base::FunctionTensor<T, Rank>(copy) {
        _func_tensor    = copy._func_tensor;
        _offsets        = copy._offsets;
        _index_template = copy._index_template;
        _full_view      = copy._full_view;
    }

    virtual T call(std::array<int, Rank> const &inds) const override {
        auto fixed_inds = apply_view(inds);
        return _func_tensor->call(fixed_inds);
    }

    bool full_view_of_underlying() const override { return _full_view; }
};

/**
 * @struct KroneckerDelta
 *
 * @brief This function tensor represents the Kronecker delta, and can be used in einsum calls.
 *
 * This function tensor can act as an example of what can be done with the more general function tensors.
 */
template <typename T>
struct KroneckerDelta : public virtual tensor_base::FunctionTensor<T, 2>, virtual tensor_base::CoreTensor {
  public:
    KroneckerDelta(size_t dim) : tensor_base::FunctionTensor<T, 2>("Kronecker Delta", dim, dim) {}

    virtual T call(std::array<int, 2> const &inds) const override { return (inds[0] == inds[1]) ? T{1.0} : T{0.0}; }
};

} // namespace einsums