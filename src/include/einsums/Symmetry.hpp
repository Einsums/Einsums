#pragma once

#include "einsums/_Common.hpp"
#include <initializer_list>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::symm)

struct Representation {
  protected:
    std::vector<int> _characters;

  public:

    Representation() = default;

    Representation(const std::vector<int> &characters) : _characters(characters) { }
    Representation(std::initializer_list<int> &characters) : _characters(characters) { }
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

    int operator[](int i) const;
    int &operator[](int i);

    int size() const;

    friend Representation operator*(int value, const Representation &other);
};

Representation operator*(int value, const Representation &other);

int dot(const Representation &first, const Representation &second);

struct Irrep : public Representation {
    protected:
    std::string _name;

    public:

    Irrep() = default;

    Irrep(std::string name, const std::vector<int> &characters) : _name(name), Representation(characters) {}
    Irrep(std::string name, std::initializer_list<int> &characters) : _name(name), Representation(characters) {}

    virtual ~Irrep() = default;

    std::string get_name() const;
};

struct PointGroup {
  protected:
    std::string _name;
    size_t _order;
    std::vector<Irrep> _irreps;
public:
    PointGroup(std::string name, const std::vector<Irrep> &irreps) : _name(name), _order(irreps.size()), _irreps(irreps) {}
    PointGroup(std::string name, const std::initializer_list<Irrep> &irreps) : _name(name), _order(irreps.size()), _irreps(irreps) {}

    virtual ~PointGroup() = default;

    int order() const;

    std::string name() const;

    std::vector<Irrep> get_irreps() const;
};

PointGroup get_point_group(std::string name);

END_EINSUMS_NAMESPACE_HPP(einsums::symm)