#include "einsums/Symmetry.hpp"
#include "einsums/_Common.hpp"
#include <algorithm>
#include <cctype>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::symm)

Representation &Representation::operator*=(const Representation &other) {
    for(int i = 0; i < size(); i++) {
        _characters[i] *= other[i];
    }

    return *this;
}

Representation &Representation::operator*=(int other) {
    for(int i = 0; i < size(); i++) {
        _characters[i] *= other;
    }

    return *this;
}

Representation &Representation::operator+=(const Representation &other) {
    for(int i = 0; i < size(); i++) {
        _characters[i] += other[i];
    }

    return *this;
}

Representation &Representation::operator-=(const Representation &other) {
    for(int i = 0; i < size(); i++) {
        _characters[i] -= other[i];
    }

    return *this;
}

Representation Representation::operator*(const Representation &other) const {
    Representation out(*this);

    return out *= other;
}

Representation Representation::operator*(int other) const {
    Representation out(*this);

    return out *= other;
}

Representation Representation::operator+(const Representation &other) const {
    Representation out(*this);

    return out += other;
}

Representation Representation::operator-(const Representation &other) const {
    Representation out(*this);

    return out -= other;
}

Representation Representation::operator-() const {
    Representation out(*this);

    for(int i = 0; i < size(); i++) {
        out[i] = -_characters[i];
    }

    return out;
}

bool Representation::operator==(const Representation &other) const {
    for(int i = 0; i < size(); i++) {
        if(std::abs(_characters[i] - other[i]) > 1e-8) {
            return false;
        }
    }
    return true;
}

int Representation::operator[](int i) const {
    return _characters[i];
}

int &Representation::operator[](int i) {
    return _characters[i];
}

int Representation::size() const {
    return _characters.size();
}

Representation operator*(int value, const Representation &other) {
    return other * value;
}

int dot(const Representation &first, const Representation &second) {
    int out = 0;

    for(int i = 0; i < first.size(); i++) {
        out += first[i] * second[i];
    }

    return out;
}

std::string Irrep::get_name() const {
    return _name;
}

int PointGroup::order() const {
    return _order;
}

std::string PointGroup::name() const {
    return _name;
}

std::vector<Irrep> PointGroup::get_irreps() const {
    return _irreps;
}


PointGroup get_point_group(std::string name) {
    std::string lower = name;

    std::transform(name.cbegin(), name.cend(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    if(lower == "c1") {
        return PointGroup("C1", {Irrep("A", {1})});
    } else if(lower == "ci") {
        return PointGroup("Ci", {Irrep("Ag", {1, 1}),
        Irrep("Au", {1, -1})});
    } else if(lower == "cs") {
        return PointGroup("Cs", {Irrep("A'", {1, 1}),
        Irrep("A\"", {1, -1})});
    } else if(lower == "c2") {
        return PointGroup("C2", {Irrep("A", {1, 1}),
        Irrep("B", {1, -1})});
    } else if(lower == "c2v") {
        return PointGroup("C2v", {Irrep("A1", {1, 1, 1, 1}),
        Irrep("A2", {1, 1, -1, -1}),
        Irrep("B1", {1, -1, 1, -1}),
        Irrep("B2", {1, -1, -1, 1})});
    } else if(lower == "c2h") {
        return PointGroup("C2h", {Irrep("Ag", {1, 1, 1, 1}),
        Irrep("Bg", {1, -1, 1, -1}),
        Irrep("Au", {1, 1, -1, -1}),
        Irrep("Bu", {1, -1, -1, 1})});
    } else if(lower == "d2") {
        return PointGroup("D2", {Irrep("A", {1, 1, 1, 1}),
        Irrep("B1", {1, 1, -1, -1}),
        Irrep("B2", {1, -1, 1, -1}),
        Irrep("B3", {1, -1, -1, 1})});
    } else if(lower == "d2d") {
        return PointGroup("D2h", {Irrep("Ag", {1, 1, 1, 1, 1, 1, 1, 1}),
        Irrep("B1g", {1, 1, -1, -1, 1, 1, -1, -1}),
        Irrep("B2g", {1, -1, 1, -1, 1, -1, 1, -1}),
        Irrep("B3g", {1, -1, -1, 1, 1, -1, -1, 1}),
        Irrep("Au", {1, 1, 1, 1, -1, -1, -1, -1}),
        Irrep("B1u", {1, 1, -1, -1, -1, -1, 1, 1}),
        Irrep("B2u", {1, -1, 1, -1, -1, 1, -1, 1}),
        Irrep("B3u", {1, -1, -1, 1, -1, 1, 1, -1})});
    } else {
        return PointGroup("C1", {Irrep("A", {1})});
    }
}

END_EINSUMS_NAMESPACE_CPP(einsums::symm)