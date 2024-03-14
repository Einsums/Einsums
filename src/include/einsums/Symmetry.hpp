#pragma once

#include "einsums/_Common.hpp"

#include <initializer_list>
#include <type_traits>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::symm)

namespace detail {
enum OpType { Identity, PropRot, Inversion, ImpRot, Mirror };

enum Axis { Primary, Dihedral, Vertical, Horizontal };

struct Character;

struct CharTerm {
  private:
    int          _coef;
    unsigned int _m, _n; // cos (m/n pi)

  public:
    CharTerm() : _coef(0), _m(0), _n(1) {}

    CharTerm(int coef, int m = 0, int n = 1);

    virtual ~CharTerm() = default;

    Character &operator+(const CharTerm &other) const;
    Character &operator-(const CharTerm &other) const;
    Character &operator*(const CharTerm &other) const;
    Character &operator+(const Character &other) const;
    Character &operator-(const Character &other) const;
    Character &operator*(const Character &other) const;
    Character &operator+(int other) const;
    Character &operator-(int other) const;
    CharTerm  &operator*(int other) const;

    friend Character &operator+(int other, const CharTerm &term);
    friend Character &operator-(int other, const CharTerm &term);
    friend CharTerm  &operator*(int other, const CharTerm &term);

    CharTerm &operator-() const;

    bool operator==(const CharTerm &other) const;
    bool operator==(const Character &other) const;

    template <typename T>
    auto operator==(T other) const -> std::enable_if_t<std::is_arithmetic_v<T>, double> {
        return other == this->eval();
    }

    double operator<=>(const CharTerm &other) const;
    double operator<=>(const Character &other) const;

    CharTerm &operator=(const CharTerm &other);
    CharTerm &operator=(int other);

    template <typename T>
    auto operator<=>(T other) const -> std::enable_if_t<std::is_arithmetic_v<T>, double> {
        return other - this->eval();
    }

    double operator()() const;

    double eval() const;

    bool is_compatible(const CharTerm &other) const;

    int get_coef() const;
    int get_m() const;
    int get_n() const;

    void reduce();

    void add_to_coef(int val);
    void add_to_coef(const CharTerm &val);
};

Character &operator+(int other, const CharTerm &term);
Character &operator-(int other, const CharTerm &term);
CharTerm  &operator*(int other, const CharTerm &term);

struct Character {
  private:
    std::vector<CharTerm> _terms;

  public:
    Character() = default;

    Character(int term) : _terms{(CharTerm)term} {}

    Character(const CharTerm &term) : _terms{term} {}

    Character(const Character &copy) : _terms{copy._terms} {}

    ~Character() = default;

    Character &operator+=(const CharTerm &other);
    Character &operator-=(const CharTerm &other);
    Character &operator*=(const CharTerm &other);
    Character &operator+=(const Character &other);
    Character &operator-=(const Character &other);
    Character &operator*=(const Character &other);
    Character &operator+=(int other);
    Character &operator-=(int other);
    Character &operator*=(int other);

    Character &operator+(const CharTerm &other) const;
    Character &operator-(const CharTerm &other) const;
    Character &operator*(const CharTerm &other) const;
    Character &operator+(const Character &other) const;
    Character &operator-(const Character &other) const;
    Character &operator*(const Character &other) const;
    Character &operator+(int other) const;
    Character &operator-(int other) const;
    Character &operator*(int other) const;

    friend Character &operator+(int other, const Character &term);
    friend Character &operator-(int other, const Character &term);
    friend Character &operator*(int other, const Character &term);

    Character &operator-() const;

    bool operator==(const CharTerm &other) const;
    bool operator==(const Character &other) const;

    template <typename T>
    auto operator==(T other) const -> std::enable_if_t<std::is_arithmetic_v<T>, double> {
        return other == this->eval();
    }

    double operator<=>(const CharTerm &other) const;
    double operator<=>(const Character &other) const;

    template <typename T>
    auto operator<=>(T other) const -> std::enable_if_t<std::is_arithmetic_v<T>, double> {
        return other - this->eval();
    }

    double operator()() const;

    double eval() const;

    void reduce();
};

Character &operator+(int other, const Character &term);
Character &operator-(int other, const Character &term);
Character &operator*(int other, const Character &term);
} // namespace detail

struct Operation {
  protected:
    std::string    _name;
    detail::OpType _type;
    detail::Axis   _axis;

    unsigned int _order, _power;

  public:
    Operation() = default;

    Operation(std::string name, detail::OpType type, unsigned int order = 1, unsigned int power = 1, detail::Axis axis = detail::Primary)
        : _name(name), _type(type), _order(order), _power(power), _axis(axis) {}

    virtual ~Operation() = default;

    bool operator==(const Operation &other) const;

    std::string get_name() const;

    detail::OpType get_type() const;

    detail::Axis get_axis() const;

    unsigned int get_order() const;

    unsigned int get_power() const;

    detail::Character get_char(int L, bool gerade) const;
};

struct Representation {
  protected:
    std::vector<detail::Character> _characters;

  public:
    Representation() = default;

    Representation(const std::vector<detail::Character> &characters) : _characters(characters) {}
    Representation(std::initializer_list<detail::Character> &characters) : _characters(characters) {}
    Representation(const Representation &copy) : _characters(copy._characters) {}

    virtual ~Representation() = default;

    Representation operator*(const Representation &other) const;
    Representation operator*(int value) const;
    Representation operator+(const Representation &other) const;
    Representation operator-(const Representation &other) const;
    Representation operator-() const;

    Representation &operator*=(const Representation &other);
    Representation &operator*=(int value);
    Representation &operator+=(const Representation &other);
    Representation &operator-=(const Representation &other);

    bool operator==(const Representation &other) const;

    const detail::Character &operator[](int i) const;
    detail::Character       &operator[](int i);

    int size() const;

    friend Representation operator*(int value, const Representation &other);
};

Representation operator*(int value, const Representation &other);

struct Irrep : public Representation {
  protected:
    std::string _name;

  public:
    Irrep() = default;

    Irrep(std::string name, const std::vector<detail::Character> &characters) : _name(name), Representation(characters) {}
    Irrep(std::string name, std::initializer_list<detail::Character> &characters) : _name(name), Representation(characters) {}

    virtual ~Irrep() = default;

    std::string get_name() const;
};

struct PointGroup {
  protected:
    std::string            _name;
    size_t                 _order;
    std::vector<Irrep>     _irreps;
    std::vector<Operation> _ops;
    std::vector<int>       _mults;

  public:
    PointGroup(std::string name, const std::vector<Operation> &ops, const std::vector<int> _mults, const std::vector<Irrep> &irreps)
        : _name(name), _order(irreps.size()), _irreps(irreps) {}
    PointGroup(std::string name, const std::initializer_list<Operation> &ops, const std::initializer_list<int> _mults,
               const std::initializer_list<Irrep> &irreps)
        : _name(name), _order(irreps.size()), _irreps(irreps) {}

    virtual ~PointGroup() = default;

    int order() const;

    std::string name() const;

    std::vector<Irrep>     get_irreps() const;
    std::vector<Operation> get_ops() const;
    std::vector<int>       get_mults() const;

    Irrep     irrep(int i) const;
    Operation operation(int i) const;
    int       mult(int i) const;

    Representation get_ang_rep(unsigned int L, bool gerade) const;

    std::vector<Irrep> reduce_rep(const Representation &rep) const;
};

int dot(const Representation &first, const Representation &second, const PointGroup &pg);

PointGroup get_point_group(std::string name);

END_EINSUMS_NAMESPACE_HPP(einsums::symm)