#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#ifdef __HIP__
#    include "einsums/DeviceTensor.hpp"
#endif
#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorBases.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <stdexcept>

namespace einsums {

namespace tensor_props {

/**
 * @struct BlockTensorBase
 *
 * Represents a block-diagonal tensor.
 *
 * @tparam T The data type stored in this tensor.
 * @tparam Rank The rank of the tensor
 * @tparam TensorType The underlying type for the tensors.
 */
template <typename T, size_t Rank, typename TensorType>
struct BlockTensorBase : public virtual CollectedTensorBase<T, Rank, TensorType>,
                         virtual TensorBase<T, Rank>,
                         virtual BlockTensorBaseNoExtra,
                         virtual LockableTensorBase,
                         virtual AlgebraOptimizedTensor {
  protected:
    std::string _name{"(Unnamed)"};
    size_t      _dim{0}; // Only allowing square tensors.

    std::vector<TensorType> _blocks{};
    std::vector<Range>      _ranges{};
    std::vector<size_t>     _dims;

    template <typename T_, size_t OtherRank, typename OtherTensorType>
    friend struct BlockTensorBase;

    T _zero_value{0.0};

    // template <typename T_, size_t Rank_>
    // friend struct BlockTensorViewBase;

    void update_dims() {
        if (_dims.size() != _blocks.size()) {
            _dims.resize(_blocks.size());
        }
        if (_ranges.size() != _blocks.size()) {
            _ranges.resize(_blocks.size());
        }

        size_t sum = 0;

        for (int i = 0; i < _blocks.size(); i++) {
            _dims[i]   = _blocks[i].dim(0);
            _ranges[i] = Range{sum, sum + _dims[i]};
            sum += _dims[i];
        }

        _dim = sum;
    }

  public:
    /**
     * @brief Construct a new BlockTensor object. Default constructor.
     */
    BlockTensorBase() = default;

    /**
     * @brief Construct a new BlockTensor object. Default copy constructor
     */
    BlockTensorBase(const BlockTensorBase &other) : _ranges{other._ranges}, _dims{other._dims}, _blocks{}, _dim{other._dim} {
        for (int i = 0; i < other._blocks.size(); i++) {
            _blocks.emplace_back((const TensorType &)other._blocks[i]);
        }

        update_dims();
    }

    /**
     * @brief Destroy the BlockTensor object.
     */
    virtual ~BlockTensorBase() = default;

    /**
     * @brief Construct a new BlockTensor object with the given name and blockdimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
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
    explicit BlockTensorBase(std::string name, Dims... block_dims)
        : _name{std::move(name)}, _dim{(static_cast<size_t>(block_dims) + ... + 0)}, _blocks(), _ranges(), _dims(sizeof...(Dims)) {
        auto dim_array   = Dim<sizeof...(Dims)>{block_dims...};
        auto _block_dims = Dim<Rank>();

        for (int i = 0; i < sizeof...(Dims); i++) {
            _block_dims.fill(dim_array[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * @brief Construct a new BlockTensor object with the given name and blockdimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
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
    template <typename ArrayArg>
    explicit BlockTensorBase(std::string name, const ArrayArg &block_dims)
        : _name{std::move(name)}, _dim{0}, _blocks(), _ranges(), _dims(block_dims.cbegin(), block_dims.cend()) {

        auto _block_dims = Dim<Rank>();

        for (int i = 0; i < block_dims.size(); i++) {
            _block_dims.fill(block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimension of each block.
     */
    template <size_t Dims>
    explicit BlockTensorBase(Dim<Dims> block_dims) : _blocks(), _ranges(), _dims(block_dims) {
        auto _block_dims = Dim<Rank>();

        for (int i = 0; i < Dims; i++) {
            _block_dims.fill(_block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * @brief Find the block which can be indexed by the given index.
     *
     * This finds the block for which the index is within its range.
     */
    int block_of(size_t index) const {
        for (int i = 0; i < _ranges.size(); i++) {
            if (_ranges[i][0] <= index && _ranges[i][1] > index) {
                return i;
            }
        }

        throw EINSUMSEXCEPTION("Index out of range!");
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
     * @brief Set the all entries in the blocks to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].set_all(value);
        }
    }

    /**
     * @brief Return the selected block with an integer ID.
     */
    const TensorType &block(int id) const { return _blocks.at(id); }

    /**
     * @brief Return the selected block with an integer ID.
     */
    TensorType &block(int id) { return _blocks.at(id); }

    /**
     * @brief Return the selected block with an integer ID.
     */
    const TensorType &block(const std::string &name) const {
        for (int i = 0; i < _blocks.size(); i++) {
            if (_blocks[i].name() == name) {
                return _blocks[i];
            }
        }
        if (_blocks.size() == 0) {
            throw EINSUMSEXCEPTION("Could not find block with the name " + name + ": no blocks in tensor.");
        }
        throw EINSUMSEXCEPTION("Could not find block with the name " + name + ": no blocks with given name.");
    }

    /**
     * @brief Return the selected block with an integer ID.
     */
    TensorType &block(const std::string &name) {
        for (int i = 0; i < _blocks.size(); i++) {
            if (_blocks[i].name() == name) {
                return _blocks[i];
            }
        }
        if (_blocks.size() == 0) {
            throw EINSUMSEXCEPTION("Could not find block with the name " + name + ": no blocks in tensor.");
        }
        throw EINSUMSEXCEPTION("Could not find block with the name " + name + ": no blocks with given name.");
    }

    /**
     * @brief Add a block to the end of the list of blocks.
     */
    void push_block(TensorType &&value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                throw EINSUMSEXCEPTION(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        update_dims();
    }

    /**
     * @brief Add a block to the specified position in the
     */
    void insert_block(int pos, TensorType &&value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                throw EINSUMSEXCEPTION(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        update_dims();
    }

    /**
     * @brief Add a block to the end of the list of blocks.
     */
    void push_block(const TensorType &value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                throw EINSUMSEXCEPTION(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        update_dims();
    }

    /**
     * @brief Add a block to the specified position in the list of blocks.
     */
    void insert_block(int pos, const TensorType &value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                throw EINSUMSEXCEPTION(
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        update_dims();
    }

    /**
     * Returns a pointer into the tensor at the given location.
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
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks.at(block).strides().begin(), size_t{0});
        return _blocks.at(block).data()[ordinal];
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

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return std::apply(_blocks.at(block), index_list);
    }

    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    T &operator()(const Container &index) {
        if (index.size() < Rank) {
            [[unlikely]] throw EINSUMSEXCEPTION("Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) {
            [[unlikely]] throw EINSUMSEXCEPTION("Too many indices passed to Tensor!");
        }

        std::array<std::int64_t, Rank> index_list{};

        for(int i = 0; i < Rank; i++) {
            index_list[i] = index[i];
        }

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

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return std::apply(_blocks.at(block), index_list);
    }

    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    const T &operator()(const Container &index) const {
        if (index.size() < Rank) {
            [[unlikely]] throw EINSUMSEXCEPTION("Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) {
            [[unlikely]] throw EINSUMSEXCEPTION("Too many indices passed to Tensor!");
        }

        std::array<std::int64_t, Rank> index_list{};

        for(int i = 0; i < Rank; i++) {
            index_list[i] = index[i];
        }

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

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return std::apply(_blocks.at(block), index_list);
    }

    /**
     * @brief Return the block with the given index.
     */
    const TensorType &operator[](size_t index) const { return this->block(index); }

    /**
     * @brief Return the block with the given index.
     */
    TensorType &operator[](size_t index) { return this->block(index); }

    /**
     * @brief Return the block with the given name.
     */
    const TensorType &operator[](const std::string &name) const { return this->block(name); }

    /**
     * @brief Return the block with the given name.
     */
    TensorType &operator[](const std::string &name) { return this->block(name); }

    /**
     * @brief Copy assignment.
     */
    auto operator=(const BlockTensorBase<T, Rank, TensorType> &other) -> BlockTensorBase<T, Rank, TensorType> & {

        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
        }

        _dims = other._dims;

        _dim = other._dim;

        _ranges = other._ranges;

        assert(_dims.size() > 0 || _dim == 0);
        assert(_dims.size() == _ranges.size());
        assert(_dims.size() == _blocks.size());

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
        }

        update_dims();

        return *this;
    }

    /**
     * @brief Copy assignment with a cast.
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const BlockTensorBase<TOther, Rank, TensorType> &other) -> BlockTensorBase<T, Rank, TensorType> & {
        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
        }

        _dims = other._dims;

        _dim = other._dim;

        _ranges = other._ranges;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
        }

        update_dims();

        return *this;
    }

#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b)->BlockTensorBase<T, Rank, TensorType> & {                                                                 \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (block_dim(i) == 0) {                                                                                                       \
                continue;                                                                                                                  \
            }                                                                                                                              \
            _blocks[i] OP b;                                                                                                               \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    auto operator OP(const BlockTensorBase<T, Rank, TensorType> &b)->BlockTensorBase<T, Rank, TensorType> & {                              \
        if (_blocks.size() != b._blocks.size()) {                                                                                          \
            throw EINSUMSEXCEPTION(fmt::format("tensors differ in number of blocks : {} {}", _blocks.size(), b._blocks.size()));           \
        }                                                                                                                                  \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (_blocks[i].size() != b._blocks[i].size()) {                                                                                \
                throw EINSUMSEXCEPTION(fmt::format("tensor blocks differ in size : {} {}", _blocks[i].size(), b._blocks[i].size()));       \
            }                                                                                                                              \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (block_dim(i) == 0) {                                                                                                       \
                continue;                                                                                                                  \
            }                                                                                                                              \
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
    explicit operator TensorType() const {
        Dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        TensorType out(block_dims);

        out.set_name(_name);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            if (this->block_dim(i) == 0) {
                continue;
            }
            std::array<Range, Rank> ranges;
            ranges.fill(_ranges[i]);
            std::apply(out, ranges) = _blocks[i];
        }

        return out;
    }

    explicit operator TensorType() {
        update_dims();
        Dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        TensorType out(block_dims);

        out.set_name(_name);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            if (this->block_dim(i) == 0) {
                continue;
            }
            std::array<Range, Rank> ranges;
            ranges.fill(_ranges[i]);
            std::apply(out, ranges) = _blocks[i];
        }

        return out;
    }

    /**
     * @brief Return the number of blocks.
     */
    size_t num_blocks() const { return _blocks.size(); }

    /**
     * @brief Return the dimensions of each block.
     */
    [[nodiscard]] auto block_dims() const -> const std::vector<size_t> { return _dims; }

    /**
     * @brief Return a list containing the ranges for each block.
     */
    std::vector<Range> ranges() const { return _ranges; }

    /**
     * @brief Return the range for a given block.
     */
    Range block_range(int i) const { return _ranges.at(i); }

    /**
     * @brief Return the dimensions of the given block.
     */
    Dim<Rank> block_dims(size_t block) const { return _blocks.at(block).dims(); }

    /**
     * @brief Return the dimension of a block on a given axis.
     *
     * Because the tensors are assumed to be square, changing the second parameter should not affect the output.
     * The second parameter is not ignored.
     */
    size_t block_dim(size_t block, int ind = 0) const { return _blocks.at(block).dim(ind); }

    /**
     * @brief Return the dimensions of a given block.
     */
    Dim<Rank> block_dims(const std::string &name) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dims();
            }
        }

        throw EINSUMSEXCEPTION("Could not find block with the name " + name);
    }

    /**
     * @brief Return the dimension of a block on a given axis.
     *
     * Because the tensors are assumed to be square, changing the second parameter should not affect the output.
     * The second parameter is not ignored.
     */
    size_t block_dim(const std::string &name, int ind = 0) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dim(ind);
            }
        }

        throw EINSUMSEXCEPTION("Could not find block with the name " + name);
    }

    /**
     * @brief Return the dimensions of this tensor.
     */
    virtual Dim<Rank> dims() const override {
        Dim<Rank> out;
        out.fill(_dim);
        return out;
    }

    /**
     * @brief Return the dimension of this tensor along an axis.
     *
     * Because the tensor is square, the argument is ignored.
     */
    virtual size_t dim(int dim) const override { return _dim; }

    virtual size_t dim() const { return _dim; }

    /**
     * @brief Return the dimensions of each of the blocks.
     */
    std::vector<size_t> vector_dims() const {
        std::vector<size_t> out(num_blocks());

        for (int i = 0; i < out.size(); i++) {
            out[i] = _blocks[i].dim(0);
        }

        return out;
    }

    /**
     * @brief Returns the list of tensors.
     */
    auto vector_data() const -> const std::vector<TensorType> & { return _blocks; }

    /**
     * @brief Returns the list of tensors.
     */
    auto vector_data() -> std::vector<TensorType> & { return _blocks; }

    /**
     * @brief Gets the name of the tensor.
     */
    [[nodiscard]] auto name() const -> const std::string & override { return _name; }

    /**
     * @brief Sets the name of the tensor.
     */
    void set_name(const std::string &name) override { _name = name; }

    /**
     * @brief Gets the name of a block.
     */
    [[nodiscard]] auto name(int i) const -> const std::string & { return _blocks[i].name(); }

    /**
     * @brief Sets the name of a block.
     */
    void set_name(int i, const std::string &name) { _blocks[i].set_name(name); }

    /**
     * @brief Returns the strides of a given block.
     */
    auto strides(int i) const noexcept -> const auto & { return _blocks[i].strides(); }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const {
        size_t sum = 0;
        for (auto tens : _blocks) {
            sum += tens.size();
        }

        return sum;
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual void lock() const override { LockableTensorBase::lock(); }

    virtual void unlock() const override { LockableTensorBase::unlock(); }

    virtual bool try_lock() const override { return LockableTensorBase::try_lock(); }

    /**
     * Lock the specific block.
     */
    virtual void lock(int block) const {
        if constexpr (einsums::detail::IsLockableTensorV<TensorType>) {
            _blocks.at(block).lock();
        }
    }

    /**
     * Try to lock the specific block.
     */
    virtual bool try_lock(int block) const {
        if constexpr (einsums::detail::IsLockableTensorV<TensorType>) {
            return _blocks.at(block).try_lock();
        } else {
            return true;
        }
    }

    /**
     * Unlock the specific block.
     */
    virtual void unlock(int block) const {
        if constexpr (einsums::detail::IsLockableTensorV<TensorType>) {
            _blocks.at(block).unlock();
        }
    }
};
} // namespace tensor_props

/**
 * @struct BlockTensor
 *
 * Represents a block-diagonal tensor in core memory.
 *
 * @tparam T The type of data stored in the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct BlockTensor : public virtual tensor_props::BlockTensorBase<T, Rank, Tensor<T, Rank>>, virtual tensor_props::CoreTensorBase {
  public:
    /**
     * @brief Construct a new BlockTensor object. Default constructor.
     */
    BlockTensor() = default;

    /**
     * @brief Construct a new BlockTensor object. Default copy constructor
     */
    BlockTensor(const BlockTensor &) = default;

    /**
     * @brief Destroy the BlockTensor object.
     */
    ~BlockTensor() = default;

    /**
     * @brief Construct a new BlockTensor object with the given name and dimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed BlockTensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit BlockTensor(std::string name, Dims... block_dims)
        : tensor_props::BlockTensorBase<T, Rank, Tensor<T, Rank>>(name, block_dims...) {}

    /**
     * @brief Construct a new BlockTensor object with the given name and dimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed BlockTensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockTensor(std::string name, const ArrayArg &block_dims)
        : tensor_props::BlockTensorBase<T, Rank, Tensor<T, Rank>>(name, block_dims) {}

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockTensor(Dim<Dims> block_dims) : tensor_props::BlockTensorBase<T, Rank, Tensor<T, Rank>>(block_dims) {}

    // size_t dim(int d) const override { return detail::BlockTensorBase<T, Rank, Tensor>::dim(d); }
};

#ifdef __HIP__
/**
 * @struct BlockDeviceTensor
 *
 * Represents a block-diagonal tensor stored on the device.
 *
 * @tparam T The type of data stored. Automatic conversion of complex types.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct BlockDeviceTensor : public virtual tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>,
                           virtual tensor_props::DeviceTensorBase,
                           virtual tensor_props::DevTypedTensorBase<T> {
  public:
    using host_datatype = typename tensor_props::DevTypedTensorBase<T>::host_datatype;
    using dev_datatype  = typename tensor_props::DevTypedTensorBase<T>::dev_datatype;

    /**
     * @brief Construct a new BlockDeviceTensor object. Default constructor.
     */
    BlockDeviceTensor() = default;

    /**
     * @brief Construct a new BlockDeviceTensor object. Default copy constructor
     */
    BlockDeviceTensor(const BlockDeviceTensor &) = default;

    /**
     * @brief Destroy the BlockDeviceTensor object.
     */
    ~BlockDeviceTensor() = default;

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
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
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit BlockDeviceTensor(std::string name, detail::HostToDeviceMode mode, Dims... block_dims)
        : tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>(name) {

        auto dims = std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(block_dims)...};

        for (int i = 0; i < sizeof...(Dims); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(dims[i]);

            tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
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
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockDeviceTensor(std::string name, detail::HostToDeviceMode mode, const ArrayArg &block_dims)
        : tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>(name) {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param mode The storage mode.
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockDeviceTensor(detail::HostToDeviceMode mode, Dim<Dims> block_dims)
        : tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>() {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            this->push_block(DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
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
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
        requires(NoneOfType<detail::HostToDeviceMode, Dims...>)
    explicit BlockDeviceTensor(std::string name, Dims... block_dims) : tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>(name) {

        auto dims = std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(block_dims)...};

        for (int i = 0; i < sizeof...(Dims); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(dims[i]);

            tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
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
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockDeviceTensor(std::string name, const ArrayArg &block_dims)
        : tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>(name) {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockDeviceTensor(Dim<Dims> block_dims) : tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>() {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            this->push_block(DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * Returns a pointer into the tensor at the given location.
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
    auto gpu_data(MultiIndex... index) -> T * {
#    if !defined(DOXYGEN_SHOULD_SKIP_THIS)
        assert(sizeof...(MultiIndex) <= Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        int  block      = -1;

        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = this->_dim + _index;
            }

            // Find the block.
            if (block == -1) {
                for (int j = 0; j < this->_ranges.size(); j++) {
                    if (this->_ranges[j][0] <= _index && _index < this->_ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (this->_ranges[block][0] <= _index && _index < this->_ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= this->_ranges[block][0];
            } else {
                return nullptr; // The indices point outside of all the blocks.
            }
        }

        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), this->_blocks[block].strides().begin(), size_t{0});
        return &(this->_blocks[block].gpu_data()[ordinal]);
#    endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This is different from the normal subscript since there needs to be a way to
     * access data on the device. C++ references do not handle data synchronization
     * in this way, so it needs to be done differently.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> HostDevReference<T> {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = this->_dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < this->_ranges.size(); j++) {
                    if (this->_ranges[j][0] <= _index && _index < this->_ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (this->_ranges.at(block)[0] <= _index && _index < this->_ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= this->_ranges.at(block)[0];
            } else {
                return HostDevReference<T>();
            }
        }

        return std::apply(this->_blocks.at(block), index_list);
    }

    size_t dim(int d) const override { return tensor_props::BlockTensorBase<T, Rank, DeviceTensor<T, Rank>>::dim(d); }
};
#endif
} // namespace einsums

template <einsums::BlockTensorConcept AType>
void println(const AType &A, TensorPrintOptions options = {}) {
    println("Name: {}", A.name());
    {
        print::Indent const indent{};
        println("Block Tensor");
        println("Data Type: {}", type_name<typename AType::data_type>());

        for (int i = 0; i < A.num_blocks(); i++) {
            println(A[i], options);
        }
    }
}

template <einsums::BlockTensorConcept AType, typename... Args>
void fprintln(FILE *fp, const AType &A, TensorPrintOptions options = {}) {
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Block Tensor");
        fprintln(fp, "Data Type: {}", type_name<typename AType::data_type>());

        for (int i = 0; i < A.num_blocks(); i++) {
            fprintln(fp, A[i], options);
        }
    }
}

template <einsums::BlockTensorConcept AType, typename... Args>
void fprintln(std::ostream &os, const AType &A, TensorPrintOptions options = {}) {
    fprintln(os, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(os, "Block Tensor");
        fprintln(os, "Data Type: {}", type_name<typename AType::data_type>());

        for (int i = 0; i < A.num_blocks(); i++) {
            fprintln(os, A[i], options);
        }
    }
}