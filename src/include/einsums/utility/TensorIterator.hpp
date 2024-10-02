#pragma once

#include "einsums/Tensor.hpp"

namespace einsums {
/**
 * @struct TensorIterator
 *
 * @brief Iterator class to make iterating over tensors easier.
 */
template <typename T, size_t Rank>
struct TensorIterator : public std::iterator<std::input_iterator_tag, T> {
  private:
    TensorView<T, Rank>       _tensor;
    std::array<ssize_t, Rank> _curr_index;

  public:
    TensorIterator(Tensor<T, Rank> &tensor) : _tensor{tensor}, _curr_index{0} {}
    TensorIterator(TensorView<T, Rank> &tensor) : _tensor{tensor}, _curr_index{0} {}

    TensorIterator(const TensorIterator &copy) : _tensor{copy._tensor}, _curr_index{copy._curr_index} {}

    TensorIterator &operator++() {
        _curr_index[Rank - 1] += 1;

        for (ssize_t i = Rank - 1; i > 0 && _curr_index[i] >= _tensor.dim(i); i--) {
            _curr_index[i] = 0;
            _curr_index[i - 1] += 1;
        }

        return *this;
    }

    TensorIterator &operator++(int) {
        auto retval = *this;
        _curr_index[Rank - 1] += 1;

        for (ssize_t i = Rank - 1; i > 0 && _curr_index[i] >= _tensor.dim(i); i--) {
            _curr_index[i] = 0;
            _curr_index[i - 1] += 1;
        }

        return retval;
    }

    TensorIterator &operator--() {
        _curr_index[Rank - 1] -= 1;

        for (ssize_t i = Rank - 1; i > 0 && _curr_index[i] < 0; i--) {
            _curr_index[i] = _tensor.dims(i);
            _curr_index[i - 1] -= 1;
        }

        return *this;
    }

    TensorIterator &operator--(int) {
        auto retval = *this;
        _curr_index[Rank - 1] -= 1;

        for (ssize_t i = Rank - 1; i > 0 && _curr_index[i] < 0; i--) {
            _curr_index[i] = 0;
            _curr_index[i - 1] -= 1;
        }

        return retval;
    }

    T &operator*() { return _tensor(_curr_index); }
};

} // namespace einsums