//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy_contraction.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#if HWY_ONCE

namespace einsums2 {
// -----------------------------------------------------------------------------
// TensorView + AxisGroup: represent logical axes, physical origins and strides
// -----------------------------------------------------------------------------

// TensorView describes a tensor *logical* axes and how those logical axes map to
// *physical* storage axes (origins). This enables representing "use D_i as D_{ii}"
// by creating a logical view with two logical axes that both map to origin 0.
struct TensorView {
    double                *data = nullptr; // base pointer (elements)
    std::vector<size_t>    logical_dims;   // per-logical-axis length
    std::vector<int>       origin_id;      // origin_id[logical_axis] -> physical axis id
    std::vector<ptrdiff_t> phys_strides;   // physical strides (in elements) indexed by origin id

    // OPTIONAL: per-logical-axis label ids; same id -> same symbol across operands
    // Canonical K ordering helpers use these to align K across operands
    std::vector<int> label_id;

    size_t logical_rank() const { return logical_dims.size(); }
    size_t physical_rank() const { return phys_strides.size(); }
};

// AxisGroup is a helper that flattens a "group" of logical axes and supports
// computing a flat size and an offset-to-physical-index calculations that
// respects repeated origins and projection/broadcast semantics
struct AxisGroup {
    // dims and origin ids are aligned: dims[i] belongs to logical axis i in this group,
    // origin[i] gives the origin id (physical axis) that logical axis maps to.
    std::vector<size_t> dims;   // per-logical-axis length
    std::vector<int>    origin; // origin id per logical axis (maps into phys_strides)
    std::vector<size_t> radix;  // flattening multipliers for dims
    size_t              flat_size = 0;

    AxisGroup() = default;

    // Build AxisGroup from the full TensorView + a list of logical axis indices
    // `axes`: indices into the view.logical_dims / view.origin_id vectors
    AxisGroup(TensorView const &v, std::vector<int> const &axes) {
        dims.reserve(axes.size());
        origin.reserve(axes.size());
        for (int a : axes) {
            dims.push_back(v.logical_dims[a]);
            origin.push_back(v.origin_id[a]);
        }
        // compute radix multipliers for flattening
        radix.resize(dims.size());
        size_t r = 1;
        for (size_t i = dims.size(); i-- > 0;) {
            radix[i] = r;
            r *= dims[i];
        }
    }

    // offset_from_flat: given a flat logical index `t` in [0,flat_size), compute:
    //  - returns pair(valid, offset) where valid==false means the logical index
    //    maps to an *inconsistent* combination of origin-equalities and thus
    //    semantically is zero (this arises when two logical axes share the same
    //    origin but their logical indices differ—e.g., off-diagonal of a projected diag)
    //  - offset is the physical offset (in elements) relative to base_ptr for the
    //    chosen combination of logical indices (sum over unique origins of phys_stride*idx)
    //
    // `phys_strides` must be a vector indexed by origin id giving the stride in elements.
    std::pair<bool, ptrdiff_t> offset_from_flat(size_t t, std::vector<ptrdiff_t> const &phys_strides) const {
        // expand t -> per-logical-aixs indices
        std::vector<size_t> idx(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            idx[i] = (t / radix[i]) % dims[i];
        }

        // chosen value per origin; -1 means not chosen yet
        int max_origin = -1;
        for (int o : origin)
            if (o > max_origin)
                max_origin = 0;
        if (max_origin < 0)
            return {true, 0}; // no axes

        std::vector<int64_t> chosen((size_t)max_origin + 1, -1);
        for (size_t p = 0; p < origin.size(); ++p) {
            int     orig = origin[p];
            int64_t val  = (int64_t)idx[p];
            if (chosen[orig] == -1)
                chosen[orig] = val;
            else {
                if (chosen[orig] != val) {
                    // Mismatch of indices for the same physical origin -> semantically zero
                    return {false, 0};
                }
            }
        }

        // Compute physical offset (sum over unique origin ids)
        ptrdiff_t off = 0;
        for (size_t orig = 0; orig < chosen.size(); ++orig) {
            if (chosen[orig] != -1) {
                // bounds check
                EINSUMS_ASSERT((size_t)orig < phys_strides.size());
                off += phys_strides[orig] * chosen[orig];
            }
        }

        return {true, off};
    }
};

// -----------------------------------------------------------------------------
// Helper: compute OperandSizes (M, K, N) from a TensorView and grouping axes
// - This is the glue you call before constructing EinsumPlanner.
// - Repeated logical axes and origins are handled: when multiple logical axes
//   map to the same origin, they contribute *only once* to the flattened product
//   for *that origin* (consistent with projection semantics).
// -----------------------------------------------------------------------------
struct OperandSizes {
    size_t M         = 1; // left free product (for left operand)
    size_t K         = 1; // contraction product
    size_t N         = 1; // right free product (for right operand)
    size_t elem_size = sizeof(double);
};

// Utility: compute the product over the set of unique origins for a given axis list.
// axes: logical axis indices into view.logical_dims
inline size_t product_unique_origins(TensorView const &view, std::vector<int> const &axes) {
    // map origin_id -> dimension (ensure consistent)
    std::unordered_map<int, size_t> origin_dim;
    for (int a : axes) {
        int    origin = view.origin_id[a];
        size_t dim    = view.logical_dims[a];
        auto   it     = origin_dim.find(origin);
        if (it == origin_dim.end())
            origin_dim[origin] = dim;
        else {
            // If the same origin appears multiple times, the logical dimensions must agree
            // (e.g., using D_i as D_{ii} is allowed because both logical dims equal the physical dim).
            if (it->second != dim) {
                std::ostringstream os;
                os << "Inconsistent dims for origin " << origin << ": " << it->second << " vs " << dim;
                throw std::runtime_error(os.str());
            }
        }
    }
    size_t prod = 1;
    for (auto &kv : origin_dim)
        prod *= kv.second;
    return prod;
}

// MakeOperandSizesFromView:
// - For a left operand: free_axes -> contributes to M, k_axes -> contributes to K.
// - For a right operand: k_axes -> contributes to K, free_axes -> contributes to N.
// The caller should supply axes in the logical order they'd like the flattening to use.
// Repeated logical axes mapping to the same origin contribute once (projection semantics).
inline OperandSizes MakeOperandSizesFromView(TensorView const &view, std::vector<int> const &free_axes, std::vector<int> const &k_axes,
                                             bool is_left_operand) {
    OperandSizes out;
    // Compute unique-origin product for free and contract axes
    if (is_left_operand) {
        out.M = product_unique_origins(view, free_axes);
        out.K = product_unique_origins(view, k_axes);
        out.N = 1; // left operand doesn't provide N
    } else {
        // right operand: free -> N, k_axes -> K
        out.M = 1;
        out.K = product_unique_origins(view, k_axes);
        out.N = product_unique_origins(view, free_axes);
    }
    out.elem_size = sizeof(double);
    return out;
}

// ============================================================================
// Canonical K ordering across operands (by label ids)
// ============================================================================

inline std::vector<int> DetermineCanonicalKLabels(std::vector<TensorView *> const     &views,
                                                  std::vector<std::vector<int>> const &k_axes_per_operand) {
    std::vector<int> labels;
    for (size_t op = 0; op < views.size(); ++op) {
        TensorView const *v = views[op];
        if (!v || v->label_id.empty())
            continue;
        for (int ax : k_axes_per_operand[op]) {
            labels.push_back(v->label_id[ax]);
        }
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    return labels;
}

inline std::vector<std::vector<int>> ReorderKAxesToCanonical(std::vector<TensorView *> const     &views,
                                                             std::vector<std::vector<int>> const &k_axes_per_operand,
                                                             std::vector<int> const              &canonical_k_labels) {
    std::vector<std::vector<int>> out(views.size());
    for (size_t op = 0; op < views.size(); ++op) {
        auto const                  *v      = views[op];
        auto const                  &k_axes = k_axes_per_operand[op];
        std::unordered_map<int, int> label_to_axis;
        if (v && !v->label_id.empty()) {
            for (int ax : k_axes)
                label_to_axis[v->label_id[ax]] = ax;
        }
        std::vector<int> ordered;
        ordered.reserve(k_axes.size());
        for (int lab : canonical_k_labels) {
            auto it = label_to_axis.find(lab);
            if (it != label_to_axis.end())
                ordered.push_back(it->second);
        }
        if (ordered.size() != k_axes.size()) {
            for (int ax : k_axes) {
                bool seen = false;
                if (v && !v->label_id.empty()) {
                    int lab = v->label_id[ax];
                    seen    = std::binary_search(canonical_k_labels.begin(), canonical_k_labels.end(), lab);
                }
                if (!seen)
                    ordered.push_back(ax);
            }
        }
        out[op] = std::move(ordered);
    }
    return out;
}

inline OperandSizes MakeOperandSizesFromViewCanonical(TensorView const &view, std::vector<int> const &free_axes,
                                                      std::vector<int> const &k_axes_canonical, bool is_left_operand) {
    return MakeOperandSizesFromView(view, free_axes, k_axes_canonical, is_left_operand);
}

// ============================================================================
// Packing utilities
// ============================================================================

// Pack 1D axis-group into contiguous buffer; fills zeros for invalid (off-diagonal) elements.
inline void PackAxisGroupContiguous(TensorView const &view, AxisGroup const &group, double *__restrict dst /* size == group.flat_size */) {
    for (size_t t = 0; t < group.flat_size; ++t) {
        auto [ok, off] = group.offset_from_flat(t, view.phys_strides);
        dst[t]         = ok ? view.data[off] : 0.0;
    }
}

// Strided 2D pack assuming rows/cols groups map to disjoint physical origins.
// If they can share origins, use PackAxisGroup2D_Exact instead.
inline void PackAxisGroupStrided_DisjointOrigins(TensorView const &view, AxisGroup const &rows, AxisGroup const &cols,
                                                 double *__restrict dst, size_t ld_dst) {
    for (size_t c = 0; c < cols.flat_size; ++c) {
        auto [okc, offc] = cols.offset_from_flat(c, view.phys_strides);
        for (size_t r = 0; r < rows.flat_size; ++r) {
            auto [okr, offr]    = rows.offset_from_flat(r, view.phys_strides);
            dst[r + c * ld_dst] = (okc && okr) ? view.data[offr + offc] : 0.0;
        }
    }
}

// Exact 2D pack that handles shared origins (e.g., D_{ii}).
inline void PackAxisGroup2D_Exact(TensorView const &view, AxisGroup const &rows, AxisGroup const &cols, double *__restrict dst,
                                  size_t ld_dst) {
    AxisGroup merged;
    merged.dims.reserve(rows.dims.size() + cols.dims.size());
    merged.origin.reserve(rows.origin.size() + cols.origin.size());
    merged.dims.insert(merged.dims.end(), rows.dims.begin(), rows.dims.end());
    merged.dims.insert(merged.dims.end(), cols.dims.begin(), cols.dims.end());
    merged.origin.insert(merged.origin.end(), rows.origin.begin(), rows.origin.end());
    merged.origin.insert(merged.origin.end(), cols.origin.begin(), cols.origin.end());

    merged.radix.resize(merged.dims.size());
    size_t rmult = 1;
    for (ptrdiff_t i = (ptrdiff_t)merged.dims.size() - 1; i >= 0; --i) {
        merged.radix[i] = rmult;
        rmult *= merged.dims[i];
        if (i == 0)
            break;
    }
    merged.flat_size = rmult;

    size_t const R = rows.flat_size;
    size_t const C = cols.flat_size;

    size_t tail = 1;
    for (size_t i = rows.dims.size(); i < merged.dims.size(); ++i)
        tail *= merged.dims[i];

    for (size_t c = 0; c < C; ++c) {
        for (size_t r = 0; r < R; ++r) {
            size_t t            = r * tail + c;
            auto [ok, off]      = merged.offset_from_flat(t, view.phys_strides);
            dst[r + c * ld_dst] = ok ? view.data[off] : 0.0;
        }
    }
}

// Optional helper: pack only the diagonal where a pair of logical axes share the same origin.
// Returns number of diagonal elements written.
inline size_t PackDiagonalIfProjected(TensorView const &view, std::pair<int, int> const &logical_axes_same_origin, double *__restrict dst) {
    int const ax0 = logical_axes_same_origin.first;
    int const ax1 = logical_axes_same_origin.second;
    if (view.origin_id[ax0] != view.origin_id[ax1])
        return 0; // not a projection
    size_t const    dim    = view.logical_dims[ax0];
    int const       origin = view.origin_id[ax0];
    ptrdiff_t const stride = view.phys_strides[origin];
    for (size_t i = 0; i < dim; ++i)
        dst[i] = view.data[i * stride];
    return dim;
}

// ============================================================================
// Simple cost-based planner skeleton
// ============================================================================

struct CostModelParams {
    double mem_weight       = 1e-6;
    size_t max_temp_bytes   = 1 << 20;
    int    max_fuse         = 3;
    int    exhaustive_limit = 6;
};

enum class NodeType { LEAF, BINARY, FUSED };

struct PlanNode {
    NodeType                  type;
    int                       operand_index = -1;
    std::unique_ptr<PlanNode> left;
    std::unique_ptr<PlanNode> right;
    std::vector<int>          fused_operands;
    double                    flops      = 0.0;
    size_t                    temp_bytes = 0;
    double                    score      = 0.0;

    static std::unique_ptr<PlanNode> MakeLeaf(int idx) {
        auto p           = std::make_unique<PlanNode>();
        p->type          = NodeType::LEAF;
        p->operand_index = idx;
        return p;
    }
    static std::unique_ptr<PlanNode> MakeBinary(std::unique_ptr<PlanNode> L, std::unique_ptr<PlanNode> R) {
        auto p   = std::make_unique<PlanNode>();
        p->type  = NodeType::BINARY;
        p->left  = std::move(L);
        p->right = std::move(R);
        return p;
    }
    static std::unique_ptr<PlanNode> MakeFused(std::vector<int> const &ops) {
        auto p            = std::make_unique<PlanNode>();
        p->type           = NodeType::FUSED;
        p->fused_operands = ops;
        return p;
    }
};

class EinsumPlanner {
  public:
    EinsumPlanner(std::vector<OperandSizes> const &operands, CostModelParams params = {})
        : ops_(operands), params_(params), n_((int)operands.size()) {}

    std::unique_ptr<PlanNode> Plan() {
        if (n_ <= params_.exhaustive_limit)
            return PlanExhaustive_();
        return PlanGreedy_();
    }

    static void PrintPlan(PlanNode const *node, std::string const &indent = "") {
        if (!node)
            return;
        if (node->type == NodeType::LEAF) {
            std::cout << indent << "Leaf(op=" << node->operand_index << ")\n";
        } else if (node->type == NodeType::BINARY) {
            std::cout << indent << "Binary(cost=" << node->score << ", flops=" << node->flops << ", tmp=" << node->temp_bytes << ")\n";
            PrintPlan(node->left.get(), indent + "  ");
            PrintPlan(node->right.get(), indent + "  ");
        } else {
            std::cout << indent << "Fused(ops=[";
            for (size_t i = 0; i < node->fused_operands.size(); ++i) {
                if (i)
                    std::cout << ",";
                std::cout << node->fused_operands[i];
            }
            std::cout << "], cost=" << node->score << ")\n";
        }
    }

  private:
    int                       n_;
    std::vector<OperandSizes> ops_;
    CostModelParams           params_;

    static inline std::pair<double, size_t> EstimateBinaryCost_(OperandSizes const &L, OperandSizes const &R, CostModelParams const &p) {
        size_t M = L.M, K = L.K, N = R.N;
        double flops     = 2.0 * double(M) * double(K) * double(N);
        size_t tmp_bytes = M * N * L.elem_size;
        double score     = flops + p.mem_weight * double(tmp_bytes);
        return {score, tmp_bytes};
    }

    static std::unique_ptr<PlanNode> Clone_(PlanNode const *src) {
        auto out            = std::make_unique<PlanNode>();
        out->type           = src->type;
        out->operand_index  = src->operand_index;
        out->flops          = src->flops;
        out->temp_bytes     = src->temp_bytes;
        out->score          = src->score;
        out->fused_operands = src->fused_operands;
        if (src->left)
            out->left = Clone_(src->left.get());
        if (src->right)
            out->right = Clone_(src->right.get());
        return out;
    }

    std::unique_ptr<PlanNode> PlanExhaustive_() {
        struct Item {
            int                       idx;
            OperandSizes              sizes;
            std::unique_ptr<PlanNode> node;
        };
        std::vector<Item> items;
        items.reserve(n_);
        for (int i = 0; i < n_; ++i)
            items.push_back({i, ops_[i], PlanNode::MakeLeaf(i)});

        double                    best_score = std::numeric_limits<double>::infinity();
        std::unique_ptr<PlanNode> best_plan;

        std::function<void(std::vector<Item> &)> dfs = [&](std::vector<Item> &cur) {
            if (cur.size() == 1) {
                if (cur[0].node->score < best_score) {
                    best_score = cur[0].node->score;
                    best_plan  = Clone_(cur[0].node.get());
                }
                return;
            }
            for (size_t a = 0; a < cur.size(); ++a) {
                for (size_t b = a + 1; b < cur.size(); ++b) {
                    auto const &A = cur[a];
                    auto const &B = cur[b];
                    if (A.sizes.K != B.sizes.K)
                        continue;
                    auto [score_bin, tmp_bytes] = EstimateBinaryCost_(A.sizes, B.sizes, params_);
                    double new_score            = A.node->score + B.node->score + score_bin;
                    if (new_score >= best_score)
                        continue;

                    Item merged;
                    merged.sizes.M          = A.sizes.M;
                    merged.sizes.K          = A.sizes.K;
                    merged.sizes.N          = B.sizes.N;
                    merged.sizes.elem_size  = A.sizes.elem_size;
                    merged.node             = PlanNode::MakeBinary(Clone_(A.node.get()), Clone_(B.node.get()));
                    merged.node->flops      = score_bin;
                    merged.node->temp_bytes = tmp_bytes;
                    merged.node->score      = new_score;

                    std::vector<Item> next;
                    next.reserve(cur.size() - 1);
                    for (size_t t = 0; t < cur.size(); ++t)
                        if (t != a && t != b)
                            next.push_back(std::move(cur[t]));
                    next.push_back(std::move(merged));
                    dfs(next);
                }
            }
        };

        dfs(items);
        if (!best_plan)
            return PlanGreedy_();
        return best_plan;
    }

    std::unique_ptr<PlanNode> PlanGreedy_() {
        struct Item {
            OperandSizes              sizes;
            std::unique_ptr<PlanNode> node;
        };
        std::vector<Item> items;
        items.reserve(n_);
        for (int i = 0; i < n_; ++i) {
            Item it;
            it.sizes = ops_[i];
            it.node  = PlanNode::MakeLeaf(i);
            items.push_back(std::move(it));
        }
        while (items.size() > 1) {
            double best_score = std::numeric_limits<double>::infinity();
            size_t best_a = 0, best_b = 1;
            size_t best_tmp = 0;
            for (size_t a = 0; a < items.size(); ++a) {
                for (size_t b = a + 1; b < items.size(); ++b) {
                    OperandSizes const &A = items[a].sizes;
                    OperandSizes const &B = items[b].sizes;
                    if (A.K != B.K)
                        continue;
                    auto [score_bin, tmp_bytes] = EstimateBinaryCost_(A, B, params_);
                    if (score_bin < best_score) {
                        best_score = score_bin;
                        best_a     = a;
                        best_b     = b;
                        best_tmp   = tmp_bytes;
                    }
                }
            }
            // Merge
            auto merged             = Item{};
            merged.sizes.M          = items[best_a].sizes.M;
            merged.sizes.K          = items[best_a].sizes.K;
            merged.sizes.N          = items[best_b].sizes.N;
            merged.sizes.elem_size  = items[best_a].sizes.elem_size;
            merged.node             = PlanNode::MakeBinary(std::move(items[best_a].node), std::move(items[best_b].node));
            merged.node->flops      = best_score;
            merged.node->temp_bytes = best_tmp;
            merged.node->score      = best_score;
            if (best_b > best_a) {
                items.erase(items.begin() + best_b);
                items.erase(items.begin() + best_a);
            } else {
                items.erase(items.begin() + best_a);
                items.erase(items.begin() + best_b);
            }
            items.push_back(std::move(merged));
        }
        return std::move(items[0].node);
    }
};

// ============================================================================
// (Optional) Tiny inline example usage (can be removed in production builds)
// ============================================================================
int einsums_main() {
    // Example: project vector D_i as D_{ii} and pack its diagonal; and
    // verify repeated-index reduction A_{ijij}.

    // Repeated index reduction
    int const           I = 3, J = 2;
    std::vector<double> A(I * J * I * J);
    auto                idx4 = [&](int i, int j, int k, int l) { return ((i * J + j) * I + k) * J + l; };
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++)
            for (int k = 0; k < I; k++)
                for (int l = 0; l < J; l++)
                    A[idx4(i, j, k, l)] = (k == i && l == j) ? (10.0 * i + j) : 1.0;

    TensorView Aview;
    Aview.data         = A.data();
    Aview.logical_dims = {(size_t)I, (size_t)J, (size_t)I, (size_t)J}; // [i, j, i, j]
    Aview.origin_id    = {0, 1, 2, 3};
    Aview.phys_strides = {(ptrdiff_t)(J * I * J), (ptrdiff_t)(I * J), (ptrdiff_t)(J), (ptrdiff_t)(1)};
    Aview.label_id     = {0, 1, 0, 1};

    AxisGroup A_all(Aview, {0, 1, 2, 3});
    double    s = 0.0;
    for (size_t t = 0; t < A_all.flat_size; ++t) {
        auto [ok, off] = A_all.offset_from_flat(t, Aview.phys_strides);
        if (ok)
            s += Aview.data[off];
    }
    double s_ref = 0.0;
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++)
            s_ref += 10.0 * i + j;
    std::cout << "[Repeated] s=" << s << " ref=" << s_ref << "\n";

    // Projection example
    int const           M = 4, N = 3;
    std::vector<double> D(M), Mat(M * N), C(M * N), Cref(M * N);
    for (int i = 0; i < M; i++)
        D[i] = 1.0 + i;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            Mat[i * N + j] = 100 * i + j;

    TensorView Dview;
    Dview.data         = D.data();
    Dview.logical_dims = {(size_t)M, (size_t)M}; // use as D_{ii}
    Dview.origin_id    = {0, 0};                 // both logical axes share origin 0
    Dview.phys_strides = {(ptrdiff_t)1};
    Dview.label_id     = {0, 0};

    AxisGroup           rows(Dview, {0}), cols(Dview, {1});
    std::vector<double> Dmat(M * M);
    PackAxisGroup2D_Exact(Dview, rows, cols, Dmat.data(), /*ld_dst=*/M);

    // Scale rows: C = diag(D)*Mat
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            C[i * N + j]    = Dmat[i + i * M] * Mat[i * N + j];
            Cref[i * N + j] = D[i] * Mat[i * N + j];
        }
    double err = 0.0;
    for (int t = 0; t < M * N; t++)
        err = std::max(err, std::abs(C[t] - Cref[t]));
    std::cout << "[Proj] max err=" << err << "\n";

    // Canonical K ordering demo with two operands (toy):
    TensorView                    L = Aview, R = Aview; // pretend both share labels 0,1 on K
    std::vector<TensorView *>     views              = {&L, &R};
    std::vector<std::vector<int>> k_axes_per_operand = {{0, 1}, {2, 3}};
    auto                          canon              = DetermineCanonicalKLabels(views, k_axes_per_operand);
    auto                          kaxes              = ReorderKAxesToCanonical(views, k_axes_per_operand, canon);
    auto                          Lsizes             = MakeOperandSizesFromViewCanonical(L, /*free*/ {}, kaxes[0], /*left*/ true);
    auto                          Rsizes             = MakeOperandSizesFromViewCanonical(R, /*free*/ {}, kaxes[1], /*left*/ false);
    EinsumPlanner                 planner({Lsizes, Rsizes});
    auto                          plan = planner.Plan();
    EinsumPlanner::PrintPlan(plan.get());
    return 0;
}
} // namespace einsums2

int main(int argc, char **argv) {
    return einsums::start(einsums2::einsums_main, argc, argv);
}
#endif
