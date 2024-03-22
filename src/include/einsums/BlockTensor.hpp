#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <stdexcept>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums)

template <typename T, size_t Rank>
struct BlockTensor;

template <typename T, size_t Rank>
struct BlockTensor : public detail::TensorBase<T, Rank> {
  private:
    std::string _name{"(Unnamed)"};
    size_t      _dim; // Only allowing square tensors.

    std::vector<Tensor<T, Rank>> _blocks;
    std::vector<Range>           _ranges;

    template <typename T_, size_t OtherRank>
    friend struct BlockTensor;

    template <typename T_, size_t Rank_>
    friend struct BlockTensorView;

  public:
    using datatype = T;
    using Vector   = std::vector<T, AlignedAllocator<T, 64>>;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    BlockTensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    BlockTensor(const BlockTensor &) = default;

    /**
     * @brief Destroy the Tensor object.
     */
    ~BlockTensor() = default;

    /**
     * @brief Construct a new Tensor object with the given name and dimensions.
     *
     * Constructs a new Tensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit BlockTensor(std::string name, Dims... block_dims)
        : _name{std::move(name)}, _dim{(static_cast<size_t>(block_dims) + ...)}, _blocks(), _ranges(sizeof...(Dims)) {
        auto dim_array  = Dim<sizeof...(Dims)>{block_dims...};
        auto _block_dims = Dim<Rank>();

        size_t sum = 0;
        for (int i = 0; i < sizeof...(Dims); i++) {
            _ranges[i] = Range{sum, sum + dim_array[i]};
            sum += dim_array[i];

            _block_dims.fill(dim_array[i]);

            _blocks.emplace_back(_block_dims);
        }
    }

    template<typename ArrayArg>
    explicit BlockTensor(std::string name, const ArrayArg &block_dims)
        : _name{std::move(name)}, _dim{0}, _blocks() {

        auto _block_dims = Dim<Rank>();

        size_t sum = 0;
        for (int i = 0; i < block_dims.size(); i++) {
            _ranges[i] = Range{sum, sum + block_dims[i]};
            sum += block_dims[i];

            _block_dims.fill(block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        _dim = sum;
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockTensor(Dim<Dims> block_dims) : _blocks(), _ranges(Dims) {
        auto _block_dims = Dim<Rank>();

        size_t sum = 0;
        for (int i = 0; i < Dims; i++) {
            _ranges[i] = Range{sum, sum + _block_dims[i]};
            sum += _block_dims[i];

            _block_dims.fill(_block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        _dim = sum;
    }

    int block_of(size_t index) const {
        for(int i = 0; i < _ranges.size(); i++) {
            if(_ranges[i][0] <= i && _ranges[i][1] > i) {
                return i;
            }
        }

        throw std::out_of_range("Index out of range.");
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].zero();
        }
    }

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].set_all(value);
        }
    }

    const Tensor<T, Rank> &block(int id) const { return _blocks.at(id); }

    Tensor<T, Rank> &block(int id) { return _blocks.at(id); }

    /**
     * Add a block to the end of the list of blocks.
     */
    void push_block(Tensor<T, Rank> &&value) {
        for (int i = 0; i < Rank; i++) {
            if (value.block_dim(i) != value.block_dim(0)) {
                throw std::runtime_error(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        _ranges.emplace_back({_dim, _dim + value.block_dim(0)});
        _dim += value.block_dim(0);
    }

    /**
     * Add a bloc to the specified position in the
     */
    void insert_block(int pos, Tensor<T, Rank> &&value) {
        for (int i = 0; i < Rank; i++) {
            if (value.block_dim(i) != value.block_dim(0)) {
                throw std::runtime_error(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        // Add the new ranges.
        if (pos == 0) {
            _ranges.emplace(_ranges.begin(), {0, value.block_dim(0)});
        } else {
            _ranges.emplace(std::next(_ranges.begin(), pos), {_ranges[pos - 1][1], _ranges[pos - 1][1] + value.block_dim(0)});
        }

        for (int i = pos + 1; i < _ranges.size(); i++) {
            _ranges[i][0] += value.block_dim(0);
            _ranges[i][1] += value.block_dim(0);
        }

        // Add the new dimension.
        _dim += value.block_dim(0);
    }

    /**
     * Add a block to the end of the list of blocks.
     */
    void push_block(const Tensor<T, Rank> &value) {
        for (int i = 0; i < Rank; i++) {
            if (value.block_dim(i) != value.block_dim(0)) {
                throw std::runtime_error(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        _ranges.emplace_back({_dim, _dim + value.block_dim(0)});
        _dim += value.block_dim(0);
    }

    /**
     * Add a bloc to the specified position in the
     */
    void insert_block(int pos, const Tensor<T, Rank> &value) {
        for (int i = 0; i < Rank; i++) {
            if (value.block_dim(i) != value.block_dim(0)) {
                throw std::runtime_error(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        // Add the new ranges.
        if (pos == 0) {
            _ranges.emplace(_ranges.begin(), {0, value.block_dim(0)});
        } else {
            _ranges.emplace(std::next(_ranges.begin(), pos), {_ranges[pos - 1][1], _ranges[pos - 1][1] + value.block_dim(0)});
        }

        for (int i = pos + 1; i < _ranges.size(); i++) {
            _ranges[i][0] += value.block_dim(0);
            _ranges[i][1] += value.block_dim(0);
        }

        // Add the new dimension.
        _dim += value.block_dim(0);
    }


    /**
     * Returns a pointer into the tensor at the given location.
     *
     * @code
     * auto A = Tensor("A", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     *
     * double* A_pointer = A.data(1, 2, 3); // Returns the pointer to element (1, 2, 3) in A.
     * @endcode
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> T * {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
        assert(sizeof...(MultiIndex) <= Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        int  block      = -1;

        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            // Find the block.
            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return nullptr; // The indices point outside of all the blocks.
            }
        }

        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks[block].strides().begin(), size_t{0});
        return &(_blocks[block].data()[ordinal]);
#endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> const T & {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return 0; // The indices point outside of all the blocks.
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks[block].strides().begin(), size_t{0});
        return _blocks[block].data()[ordinal];
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> T & {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return *new T(0);
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks[block].strides().begin(), size_t{0});
        return _blocks[block].data()[ordinal];
    }

    /**
     * @brief Return the block with the given index.
     */
    const Tensor<T, Rank> &operator[](size_t index) const { return _blocks.at(index); }

    /**
     * @brief Return the block with the given index.
     */
    Tensor<T, Rank> &operator[](size_t index) { return _blocks.at(index); }

    /**
     * @brief Return the block with the given name.
     */
    const Tensor<T, Rank> operator[](const std::string &name) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens;
            }
        }
        throw std::out_of_range("Could not find block with the name " + name);
    }

    /**
     * @brief Return the block with the given name.
     */
    Tensor<T, Rank> operator[](const std::string &name) {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens;
            }
        }
        throw std::out_of_range("Could not find block with the name " + name);
    }

    auto operator=(const BlockTensor<T, Rank> &other) -> BlockTensor<T, Rank> & {

        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
            _ranges.resize(other._ranges.size());
        }

        _dim = other._dim;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
            _ranges[i] = other._ranges[i];
        }

        return *this;
    }

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const BlockTensor<TOther, Rank> &other) -> BlockTensor<T, Rank> & {
        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
            _ranges.resize(other._ranges.size());
        }

        _dim = other._dim;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
            _ranges[i] = other._ranges[i];
        }

        return *this;
    }

#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b) -> BlockTensor<T, Rank> & {                                                                               \
        for (auto tens : _blocks) {                                                                                                        \
            tens OP b;                                                                                                                     \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    auto operator OP(const BlockTensor<T, Rank> &b) -> BlockTensor<T, Rank> & {                                                            \
        if (_blocks.size() != b._blocks.size()) {                                                                                          \
            throw std::runtime_error(fmt::format("operator" EINSUMS_STRINGIFY(OP) " : tensors differ in number of blocks : {} {}",         \
                                                 _blocks.size(), b._blocks.size()));                                                       \
        }                                                                                                                                  \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (_blocks[i].size() != b._blocks[i].size()) {                                                                                \
                throw std::runtime_error(fmt::format("operator" EINSUMS_STRINGIFY(OP) " : tensor blocks differ in size : {} {}",           \
                                                     _blocks[i].size(), b._blocks[i].size()));                                             \
            }                                                                                                                              \
        }                                                                                                                                  \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            _blocks[i] OP b._blocks[i];                                                                                                    \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    /**
     * @brief Convert block tensor into a normal tensor.
     */
    explicit operator Tensor<T, Rank>() {
        Dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        Tensor<T, Rank> out(_name, block_dims);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            std::array<Range, Rank> ranges;
            ranges.fill(_ranges[i]);
            out(ranges) = _blocks[i];
        }

        return out;
    }

    size_t num_blocks() const {
        return _blocks.size();
    }

    [[nodiscard]] auto block_dim() const -> size_t { return _dim; }

    Range block_range(int i) const {
        return _ranges.at(i);
    }

    Dim<Rank> block_dims(size_t block) const { return _blocks.at(block).dims(); }

    Dim<Rank> block_dim(size_t block, int ind) const { return _blocks.at(block).dim(ind); }

    Dim<Rank> block_dims(const std::string &name) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dims();
            }
        }

        throw std::out_of_range("Could not find block with the name " + name);
    }

    Dim<Rank> block_dim(const std::string &name, int ind) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dim(ind);
            }
        }

        throw std::out_of_range("Could not find block with the name " + name);
    }

    Dim<Rank> dims() const {
        Dim<Rank> out;
        out.fill(_dim);
        return out;
    }

    size_t dim(int dim = 0) const {
        return _dim;
    }

    auto vector_data() const -> const std::vector<Tensor<T, Rank>> & { return _blocks; }
    auto vector_data() -> std::vector<Tensor<T, Rank>> & { return _blocks; }

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void               set_name(const std::string &name) { _name = name; }

    auto strides(int i) const noexcept -> const auto & { return _blocks[i].strides(); }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const {
        size_t sum = 0;
        for (auto tens : _blocks) {
            sum += tens.size();
        }

        return sum;
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }
};

END_EINSUMS_NAMESPACE_HPP(einsums)