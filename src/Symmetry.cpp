#include "einsums/Symmetry.hpp"

#include "einsums/_Common.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::symm)

namespace detail {

void CharTerm::reduce() {
    // Reduce _m to fit within one cycle.
    _m = _m % _n;

    // fit even better by using cosine symmetry.

    // Fit the latter half onto the first half, negating the sign.
    if ((double)_m / _n > 0.5) {
        _coef = -_coef;

        if (_n & 1) {
            _m = 2 * _m - _n;
            _n *= 2; // Should not overflow, since signed is cast to unsigned.
        } else {
            _m += _n / 2;
        }
    }

    // Flip the second quarter onto the first quarter.
    if ((double)_m / _n > 0.25) {
        _coef = -_coef;

        if (_n & 1) {
            _m = _n - 2 * _m;
            _n *= 2;
        } else {
            _m = _n / 2 - _m;
        }
    }

    // Find the gcd.
    unsigned int a = _m, b = _n, factor = 1;

    // Should happen at most 31 times. Worst case would be the gcd of 2^31-1 and 1.
    while (a != 0 && b != 0) {
        if (a > b) {
            std::swap(a, b);
        }

        // If both are even, then 2 is a common factor.
        if (a % 2 == 0 && b % 2 == 0) {
            factor *= 2;
            a /= 2;
            b /= 2;
        } else if (a % 2 == 0) { // If one is even, then 2 is not a common factor.
            a /= 2;
        } else if (b % 2 == 0) {
            b /= 2;
        } else { // Both are odd.
            b -= a;

            // a is still odd, but b is now even. Divide by 2.
            b /= 2;
        }
    }

    if (a == 0 && b == 0) {
        factor = 1;
    } else if (a == 0) {
        factor *= b;
    } else if (b == 0) {
        factor *= a;
    }

    // Reduce.
    _m /= factor;
    _n /= factor;

    // Check for fractions.
    if (_coef % 2 == 0 && _m == 1 && _n == 6) { // cos gives 1/2
        _m = 0;
        _n = 1;
        _coef /= 2;
    } else if (_m == 1 && _n == 2) { // cos gives 0.
        _coef = 0;
        _m    = 0;
        _n    = 1;
    }
}

CharTerm::CharTerm(int coef, int m, int n) : _coef(coef) {
    // reduce the fractions.

    // Make the values positive.
    if (m < 0) {
        _m = -m;
    } else {
        _m = m;
    }
    if (n < 0) {
        _n = -n;
    } else {
        _n = n;
    }

    reduce();
}

Character &CharTerm::operator+(const CharTerm &other) const {
    Character *out = new Character(*this);

    *out += other;
    return *out;
}

Character &CharTerm::operator-(const CharTerm &other) const {
    Character *out = new Character(*this);

    *out += other;
    return *out;
}

Character &CharTerm::operator*(const CharTerm &other) const {
    if (_coef == 0 || other._coef == 0) {
        return *new Character(); // empty is just zero.
    } else if (_m == 0) {
        CharTerm term(_coef * other._coef, other._m, other._n);

        Character *out = new Character(term);
        return *out;
    } else if (other._m == 0) {
        CharTerm term(_coef * other._coef, _m, _n);

        Character *out = new Character(term);
        return *out;
    } else if (_m == other._m && _n == other._n) {
        CharTerm first((_coef * other._coef) / 2, 2 * _m, _n), second((_coef * other._coef));

        Character *out = new Character(first);

        return *out += second;
    } else {
        CharTerm first((_coef * other._coef) / 2, _m * other._n + other._m * _n, _n * other._n),
            second((_coef * other._coef) / 2, _m * other._n - other._m * _n, _n * other._n);

        Character *out = new Character(first);

        return *out += second;
    }
}

Character &CharTerm::operator+(const Character &other) const {
    Character *out = new Character(*this);

    return *out += other;
}

Character &CharTerm::operator-(const Character &other) const {
    Character *out = new Character(*this);

    return *out -= other;
}

Character &CharTerm::operator*(const Character &other) const {
    Character *out = new Character(*this);

    return *out *= other;
}

Character &CharTerm::operator+(int other) const {
    Character *out = new Character(*this);

    return *out += other;
}

Character &CharTerm::operator-(int other) const {
    Character *out = new Character(*this);

    return *out -= other;
}

CharTerm &CharTerm::operator*(int other) const {
    CharTerm *out = new CharTerm(*this);

    out->_coef *= other;

    if (other == 0) {
        out->_m = 0;
        out->_n = 1;
    }
    return *out;
}

CharTerm &CharTerm::operator-() const {
    CharTerm *out = new CharTerm(*this);

    out->_coef = -out->_coef;
    return *out;
}

CharTerm &CharTerm::operator=(const CharTerm &other) {
    _coef = other._coef;
    _m    = other._m;
    _n    = other._n;
    return *this;
}

CharTerm &CharTerm::operator=(int other) {
    _coef = other;
    _m    = 0;
    _n    = 1;

    return *this;
}

bool CharTerm::operator==(const CharTerm &other) const {
    return eval() == other();
}

bool CharTerm::operator==(const Character &other) const {
    return eval() == other();
}

double CharTerm::operator<=>(const CharTerm &other) const {
    return other() - this->eval();
}

double CharTerm::operator<=>(const Character &other) const {
    return other() - this->eval();
}

double CharTerm::operator()() const {
    return this->eval();
}

double CharTerm::eval() const {
    if (_n != 0) {
        return _coef * std::cos(2 * _m * M_PI / _n);
    } else {
        return (double)_coef;
    }
}

bool CharTerm::is_compatible(const CharTerm &other) const {
    // Don't let everything be compatible if this term is zero, but always let this term be compatible with zero.
    return (_m == other._m && _n == other._n) || other._coef == 0;
}

int CharTerm::get_coef() const {
    return _coef;
}

int CharTerm::get_m() const {
    return _m;
}

int CharTerm::get_n() const {
    return _n;
}

void CharTerm::add_to_coef(int val) {
    _coef += val;

    if (_coef == 0) {
        _m = 0;
        _n = 1;
    }
}

void CharTerm::add_to_coef(const CharTerm &val) {
    _coef += val._coef;

    if (_coef == 0) {
        _m = 0;
        _n = 1;
    }
}

Character &operator+(int other, const CharTerm &term) {
    return term + other;
}
Character &operator-(int other, const CharTerm &term) {
    return -term + other;
}
CharTerm &operator*(int other, const CharTerm &term) {
    return term * other;
}

void Character::reduce() {
    // Remove zeros.
    for (int i = 0; i < _terms.size(); i++) {
        if (_terms[i].get_coef() == 0) {
            _terms.erase(std::next(_terms.begin(), i));
        }
    }
}

Character &Character::operator+=(const CharTerm &other) {
    for (auto it : _terms) {
        if (it.is_compatible(other)) {
            it.add_to_coef(other);

            reduce();

            return *this;
        }
    }

    _terms.push_back(other);

    reduce();
    return *this;
}

Character &Character::operator-=(const CharTerm &other) {
    for (auto it : _terms) {
        if (it.is_compatible(other)) {
            it.add_to_coef(-other);

            reduce();

            return *this;
        }
    }

    _terms.push_back(-other);

    reduce();
    return *this;
}

Character &Character::operator*=(const CharTerm &other) {
    Character temp;

    for (auto it : _terms) {
        temp += it * other;
    }

    _terms.clear();
    _terms = temp._terms;

    return *this;
}

Character &Character::operator+=(const Character &other) {
    for (auto it : other._terms) {
        *this += it;
    }

    return *this;
}

Character &Character::operator-=(const Character &other) {
    for (auto it : other._terms) {
        *this -= it;
    }

    return *this;
}

Character &Character::operator*=(const Character &other) {
    for (auto it : other._terms) {
        *this *= it;
    }

    return *this;
}

Character &Character::operator+=(int other) {
    for (auto it : _terms) {
        if (it.get_m() == 0) {
            it.add_to_coef(other);

            reduce();

            return *this;
        }
    }

    _terms.push_back(other);

    reduce();
    return *this;
}

Character &Character::operator-=(int other) {
    for (auto it : _terms) {
        if (it.get_m() == 0) {
            it.add_to_coef(-other);

            reduce();

            return *this;
        }
    }

    _terms.push_back(-other);

    reduce();
    return *this;
}

Character &Character::operator*=(int other) {
    for (auto it : _terms) {
        it = it * other;
    }

    reduce();
    return *this;
}

Character &Character::operator+(const CharTerm &other) const {
    Character *out = new Character(*this);

    return *out += other;
}

Character &Character::operator-(const CharTerm &other) const {
    Character *out = new Character(*this);

    return *out -= other;
}

Character &Character::operator*(const CharTerm &other) const {
    Character *out = new Character(*this);

    return *out *= other;
}

Character &Character::operator+(const Character &other) const {
    Character *out = new Character(*this);

    return *out += other;
}

Character &Character::operator-(const Character &other) const {
    Character *out = new Character(*this);

    return *out -= other;
}

Character &Character::operator*(const Character &other) const {
    Character *out = new Character(*this);

    return *out *= other;
}

Character &Character::operator+(int other) const {
    Character *out = new Character(*this);

    return *out += other;
}

Character &Character::operator-(int other) const {
    Character *out = new Character(*this);

    return *out -= other;
}

Character &Character::operator*(int other) const {
    Character *out = new Character(*this);

    return *out *= other;
}

Character &Character::operator-() const {
    return *this * -1;
}

bool Character::operator==(const CharTerm &other) const {
    return eval() == other();
}

bool Character::operator==(const Character &other) const {
    return eval() == other();
}

double Character::operator<=>(const CharTerm &other) const {
    return other() - this->eval();
}

double Character::operator<=>(const Character &other) const {
    return other() - this->eval();
}

double Character::operator()() const {
    return this->eval();
}

double Character::eval() const {
    double sum = 0;

    for (auto it : _terms) {
        sum += it.eval();
    }
    return sum;
}

Character &operator+(int other, const Character &term) {
    return term + other;
}

Character &operator-(int other, const Character &term) {
    return -term + other;
}

Character &operator*(int other, const Character &term) {
    return term * other;
}

} // namespace detail

bool Operation::operator==(const Operation &other) const {
    return _name == other._name && _type == other._type && _axis == other._axis && _order == other._order &&
           (_power == other._power || _power == other._order - other._power);
}

std::string Operation::get_name() const {
    return _name;
}

detail::OpType Operation::get_type() const {
    return _type;
}

detail::Axis Operation::get_axis() const {
    return _axis;
}

unsigned int Operation::get_order() const {
    return _order;
}

unsigned int Operation::get_power() const {
    return _power;
}

static detail::Character chebychev(int n, const detail::CharTerm &cos_term) {
    if (n == 0) {
        return 1;
    } else if (n == 1) {
        return 2 * cos_term;
    } else {
        return 2 * cos_term * chebychev(n - 1, cos_term) - chebychev(n - 2, cos_term);
    }
}

detail::Character Operation::get_char(int L, bool gerade) const {
    switch (_type) {
    case detail::Identity:
        return 2 * L + 1;
    case detail::PropRot:
        return chebychev(2 * L, detail::CharTerm(1, _power, _order));
    case detail::Inversion:
        return (gerade) ? 2 * L + 1 : -(2 * L + 1);
    case detail::ImpRot:
        return ((gerade) ? 1 : -1) * chebychev(2 * L, detail::CharTerm(1, 2 * _power + _order, 2 * _order));
    case detail::Mirror:
        return ((gerade) ? 1 : -1) * ((L % 2) ? 1 : -1);
    }
    return 1;
}

Representation &Representation::operator*=(const Representation &other) {
    for (int i = 0; i < size(); i++) {
        _characters[i] *= other[i];
    }

    return *this;
}

Representation &Representation::operator*=(int other) {
    for (int i = 0; i < size(); i++) {
        _characters[i] *= other;
    }

    return *this;
}

Representation &Representation::operator+=(const Representation &other) {
    for (int i = 0; i < size(); i++) {
        _characters[i] += other[i];
    }

    return *this;
}

Representation &Representation::operator-=(const Representation &other) {
    for (int i = 0; i < size(); i++) {
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

    for (int i = 0; i < size(); i++) {
        out[i] = -_characters[i];
    }

    return out;
}

bool Representation::operator==(const Representation &other) const {
    for (int i = 0; i < size(); i++) {
        if (_characters[i] != other[i]) {
            return false;
        }
    }
    return true;
}

const detail::Character &Representation::operator[](int i) const {
    return _characters[i];
}

detail::Character &Representation::operator[](int i) {
    return _characters[i];
}

int Representation::size() const {
    return _characters.size();
}

Representation operator*(int value, const Representation &other) {
    return other * value;
}

int dot(const Representation &first, const Representation &second, const PointGroup &pg) {
    detail::Character out;

    for (int i = 0; i < first.size(); i++) {
        out += first[i] * second[i] * pg.mult(i);
    }

    return (int)out();
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

std::vector<Operation> PointGroup::get_ops() const {
    return _ops;
}
std::vector<int> PointGroup::get_mults() const {
    return _mults;
}

Irrep PointGroup::irrep(int i) const {
    return _irreps[i];
}

Operation PointGroup::operation(int i) const {
    return _ops[i];
}

int PointGroup::mult(int i) const {
    return _mults[i];
}

PointGroup get_point_group(std::string name) {
    std::string lower = name;

    std::transform(name.cbegin(), name.cend(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    if (lower == "c1") {
        return PointGroup("C1", {Operation("E", detail::Identity)}, {1}, {Irrep("A", {1})});
    } else if (lower == "ci") {
        return PointGroup("Ci", {Operation("E", detail::Identity), Operation("i", detail::Inversion)}, {1, 1},
                          {Irrep("Ag", {1, 1}), Irrep("Au", {1, -1})});
    } else if (lower == "cs") {
        return PointGroup("Cs", {Operation("E", detail::Identity), Operation("s", detail::Mirror)}, {1, 1},
                          {Irrep("A'", {1, 1}), Irrep("A\"", {1, -1})});
    } else if (lower == "c2") {
        return PointGroup("C2", {Operation("E", detail::Identity), Operation("C2", detail::PropRot, 2)}, {1, 1},
                          {Irrep("A", {1, 1}), Irrep("B", {1, -1})});
    } else if (lower == "c2v") {
        return PointGroup(
            "C2v",
            {Operation("E", detail::Identity), Operation("C2", detail::PropRot, 2),
             Operation("s(xz)", detail::Mirror, 1, 1, detail::Vertical), Operation("s(yz)", detail::Mirror, 1, 1, detail::Vertical)},
            {1, 1, 1, 1},
            {Irrep("A1", {1, 1, 1, 1}), Irrep("A2", {1, 1, -1, -1}), Irrep("B1", {1, -1, 1, -1}), Irrep("B2", {1, -1, -1, 1})});
    } else if (lower == "c2h") {
        return PointGroup(
            "C2h",
            {Operation("E", detail::Identity), Operation("C2", detail::PropRot, 2), Operation("i", detail::Inversion),
             Operation("sh", detail::Mirror, 1, 1, detail::Horizontal)},
            {1, 1, 1, 1},
            {Irrep("Ag", {1, 1, 1, 1}), Irrep("Bg", {1, -1, 1, -1}), Irrep("Au", {1, 1, -1, -1}), Irrep("Bu", {1, -1, -1, 1})});
    } else if (lower == "d2") {
        return PointGroup(
            "D2",
            {Operation("E", detail::Identity), Operation("C2(z)", detail::PropRot, 2, 1, detail::Primary),
             Operation("C2(y)", detail::PropRot, 2, 1, detail::Horizontal),
             Operation("C2(x)", detail::PropRot, 2, 1, detail::Horizontal)},
            {1, 1, 1, 1},
            {Irrep("A", {1, 1, 1, 1}), Irrep("B1", {1, 1, -1, -1}), Irrep("B2", {1, -1, 1, -1}), Irrep("B3", {1, -1, -1, 1})});
    } else if (lower == "d2h") {
        return PointGroup("D2h",
                          {Operation("E", detail::Identity), Operation("C2(z)", detail::PropRot, 2, 1, detail::Primary),
                           Operation("C2(y)", detail::PropRot, 2, 1, detail::Horizontal),
                           Operation("C2(x)", detail::PropRot, 2, 1, detail::Horizontal), Operation("i", detail::Inversion),
                           Operation("s(xy)", detail::Mirror, 1, 1, detail::Horizontal),
                           Operation("s(xz)", detail::Mirror, 1, 1, detail::Vertical),
                           Operation("s(yz)", detail::Mirror, 1, 1, detail::Vertical)},
                          {1, 1, 1, 1, 1, 1, 1, 1},
                          {Irrep("Ag", {1, 1, 1, 1, 1, 1, 1, 1}), Irrep("B1g", {1, 1, -1, -1, 1, 1, -1, -1}),
                           Irrep("B2g", {1, -1, 1, -1, 1, -1, 1, -1}), Irrep("B3g", {1, -1, -1, 1, 1, -1, -1, 1}),
                           Irrep("Au", {1, 1, 1, 1, -1, -1, -1, -1}), Irrep("B1u", {1, 1, -1, -1, -1, -1, 1, 1}),
                           Irrep("B2u", {1, -1, 1, -1, -1, 1, -1, 1}), Irrep("B3u", {1, -1, -1, 1, -1, 1, 1, -1})});
    } else {
        return PointGroup("C1", {Operation("E", detail::Identity)}, {1}, {Irrep("A", {1})});
    }
}

Representation PointGroup::get_ang_rep(unsigned int L, bool gerade) const {
    std::vector<detail::Character> chars;

    for (int i = 0; i < _order; i++) {
        chars.push_back(_ops[i].get_char(L, gerade));
    }

    return Representation(chars);
}

std::vector<Irrep> PointGroup::reduce_rep(const Representation &rep) const {
    std::vector<Irrep> out;

    for(auto irrep : _irreps) {
        int prod = dot(irrep, rep, *this);

        if(prod % _order != 0) {
            throw std::invalid_argument("Representation is not evenly reducible in the given point group!");
        }

        prod /= _order;

        for(int i = 0; i < prod; i++) {
            out.push_back(irrep);
        }
    }

    return out;
}

END_EINSUMS_NAMESPACE_CPP(einsums::symm)