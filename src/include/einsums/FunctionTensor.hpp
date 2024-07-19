#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <stdexcept>
#include <type_traits>

namespace einsums {

// forward declaration.
template <typename T, size_t Rank>
struct FunctionTensorView;

template <typename T, size_t Rank>
struct BaseFunctionTensor : einsums::detail::TensorBase<T, Rank> {
  protected:
    Dim<Rank>   _dims;
    std::string _name;
    size_t      _size;

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
                throw std::out_of_range(fmt::format("{}is too far less than zero or is greater than {}", message, _dims[i]));
            }
        }
    }

  public:
    template <typename... Args>
    BaseFunctionTensor(std::string name, Args... dims) : _dims{dims...}, _name{name} {
        _size = 1;

#pragma omp for simd reduction(* : _size)
        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    template <typename MultiIndex...>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v(MultiIndex) && ...);
        }
    T operator()(MultiIndex... inds) const {
        auto new_inds = std::array<int, Rank>{static_cast<int>(inds)...};

        fix_indices(&new_inds);

        return (*this)(new_inds);
    }

    template <typename Storage>
        requires requires {
            requires !std::is_integral_v(Storage);
            requires !std::is_same_v<Storage, AllT>;
            requires !std::is_same_v<Storage, Range>;
            requires !std::is_same_v<Storage, std::array<int, Rank>>;
        }
    T operator()(const Storage &inds) const {
        auto new_inds = std::array<int, Rank>();

        for (int i = 0; i < Rank; i++) {
            new_inds[i] = (int)inds.at(i);
        }

        fix_indices(&new_inds);

        return (*this)(new_inds);
    }

    [[nodiscard]] virtual T operator()(const std::array<int, Rank> &inds) const = 0;

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(AtLeastOneOfType<AllT, MultiIndex...>);
        }
    auto operator()(MultiIndex... inds) const
        -> FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        const auto &indices = std::forward_as_tuple(inds...);

        std::vector<int> index_template(Rank);

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

        return FunctionTensorView(this, offsets, dims, index_template);
    }
};

template <typename T, size_t Rank>
struct FunctionTensorView : einsums::detail::TensorBase<T, Rank> {
  protected:
    const BaseFunctionTensor<T, Rank> *_func_tensor;
    Dim<Rank>                          _dims;
    std::string                        _name;
    Offset<Rank>                       _offsets;
    size_t                             size;
    size_t                             _true_rank;
    std::vector<int>                _index_template;

    virtual std::vector<int> fix_indices(const std::array<int, Rank> &inds) const {
        std::vector<int> out(_index_template);
        int                 curr_rank = 0;
        for (int i = 0; i < Rank && curr_rank < _true_rank; i++) {
            while (out.at(curr_rank) >= 0) {
                curr_rank++;
            }
            out.at(curr_rank) = inds.at(i);
            if (out.at(curr_rank) < 0) {
                out.at(curr_rank) += _dims[i];
            }
            if (out.at(curr_rank) >= _dims[i] || out.at(curr_rank) < 0) {
                throw std::out_of_range(fmt::format(
                    "Function tensor view index out of range! Index of rank {} is {}, which is < 0 or >= {}.", i, inds.at(i), _dims[i]));
            }

            out.at(curr_rank) += _offsets[i];
            curr_rank++;
        }
    }

  public:
    FunctionTensorView(std::string name, const BaseFunctionTensor<T, Rank> *func_tens, const Offset<Rank> &offsets, const Dim<Rank> &dims, std::vector<size_t> index_template))
        : _dims{dims}, _name{name}, _offsets{offsets}, _func_tensor(func_tens), _index_template(index_template), _true_rank{index_template.size()} {
        _size = 1;

#pragma omp for simd reduction(* : _size)
        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    FunctionTensorView(const BaseFunctionTensor<T, Rank> *func_tens, const Offset<Rank> &offsets, const Dim<Rank> &dims, std::vector<size_t> index_template))
        : _dims{dims}, _name{"(unnamed)"}, _offsets{offsets}, _func_tensor(func_tens), _index_template(index_template), _true_rank{index_template.size()} {
        _size = 1;

#pragma omp for simd reduction(* : _size)
        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    template <typename MultiIndex...>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v(MultiIndex) && ...);
        }
    T operator()(MultiIndex... inds) const {
        auto new_inds = std::array<int, Rank>{static_cast<int>(inds)...};

        auto fixed_inds = fix_indices(&new_inds);

        return (*_func_tensor)(fixed_inds);
    }

    template <typename Storage>
        requires requires {
            requires !std::is_integral_v(Storage);
            requires !std::is_same_v<Storage, AllT>;
            requires !std::is_same_v<Storage, Range>;
            requires !std::is_same_v<Storage, std::array<int, Rank>>;
        }
    T operator()(const Storage &inds) const {
        auto new_inds = std::array<int, Rank>();

        for (int i = 0; i < Rank; i++) {
            new_inds[i] = (int)inds.at(i);
        }

        fix_indices(&new_inds);

        return (*_func_tensor)(new_inds);
    }
}

} // namespace einsums