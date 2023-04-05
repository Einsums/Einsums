#pragma once

#if defined(EINSUMS_IN_PARALLEL)

#    include "einsums/_Common.hpp"
#    include "einsums/_Compiler.hpp"

#    include <mpi.h>

namespace einsums::mpi {

struct Comm;

enum class Error { Success = 0, Buffer, Count, Type, Tag, Comm };

template <typename ResultType>
using ErrorOr = einsums::ErrorOr<ResultType, ::einsums::mpi::Error>;

EINSUMS_EXPORT auto initialize(int *argc, char ***argv) -> ErrorOr<void>;
EINSUMS_EXPORT auto finalize() -> ErrorOr<void>;
EINSUMS_EXPORT auto initialized() -> ErrorOr<bool>;
EINSUMS_EXPORT auto finalized() -> ErrorOr<bool>;

EINSUMS_EXPORT auto size() -> ErrorOr<int>;
EINSUMS_EXPORT auto rank() -> ErrorOr<int>;

enum class ThreadLevel { Single = 0, Funneled, Serialized, Multiple };

EINSUMS_EXPORT auto query_thread() -> ErrorOr<ThreadLevel>;

#    define EINSUMS_MPI_TEST(condition)                                                                                                    \
        {                                                                                                                                  \
            Error mpi_error_code = static_cast<Error>(condition);                                                                          \
            if (mpi_error_code != Error::Success)                                                                                          \
                return mpi_error_code;                                                                                                     \
        }

struct Status {
    Status() : _status() {}
    Status(const Status &other) = default;
    Status(MPI_Status other) : _status(other) {}

    // Assignment operators
    auto operator=(const Status &other) -> Status & = default;
    auto operator=(const MPI_Status other) -> Status & {
        _status = other;
        return *this;
    }

    operator MPI_Status *() { return &_status; }
    operator MPI_Status() { return _status; }

    [[nodiscard]] auto get_count(const MPI_Datatype datatype) const -> ErrorOr<int> {
        int count{0};
        EINSUMS_MPI_TEST(MPI_Get_count(const_cast<MPI_Status *>(&_status), datatype, &count));
        return count;
    }

    [[nodiscard]] auto source() const -> int { return _status.MPI_SOURCE; }
    [[nodiscard]] auto tag() const -> int { return _status.MPI_TAG; }
    [[nodiscard]] auto error() const -> Error { return static_cast<Error>(_status.MPI_ERROR); }

    void source(int source) { _status.MPI_SOURCE = source; }
    void tag(int tag) { _status.MPI_TAG = tag; }
    void error(Error error) { _status.MPI_ERROR = static_cast<int>(error); }

  private:
    MPI_Status _status;
};

struct Request {
    Request() : _request(MPI_REQUEST_NULL) {}
    Request(MPI_Request other) : _request(other) {}
    Request(const Request &other) = default;

    auto operator=(const Request &other) -> Request & = default;
    auto operator=(const MPI_Request &other) -> Request & {
        _request = other;
        return *this;
    }

    auto operator==(const Request &other) -> bool { return (_request == other._request); }
    auto operator!=(const Request &other) -> bool { return (_request != other._request); }

    operator MPI_Request *() { return &_request; }
    operator MPI_Request() const { return _request; }

  private:
    MPI_Request _request;
};

struct Group {
    friend Comm;

  protected:
    MPI_Group _real_group;
};

struct Comm {
  protected:
    MPI_Comm _real_comm;

  public:
    EINSUMS_ALWAYS_INLINE Comm(MPI_Comm object) : _real_comm(object) {}
    EINSUMS_ALWAYS_INLINE Comm() : _real_comm(MPI_COMM_NULL) {}

    virtual ~Comm() = default;

    Comm(const Comm &object)                     = default;
    auto operator=(const Comm &object) -> Comm & = default;

    auto operator==(const Comm &object) -> bool { return (_real_comm == object._real_comm); }
    auto operator!=(const Comm &object) -> bool { return (_real_comm != object._real_comm); }

    EINSUMS_ALWAYS_INLINE operator MPI_Comm *() { return &_real_comm; }
    EINSUMS_ALWAYS_INLINE operator MPI_Comm() { return _real_comm; }

    auto operator=(const MPI_Comm &object) -> Comm & {
        _real_comm = object;
        return *this;
    }

    [[nodiscard]] virtual auto group() const -> ErrorOr<Group>;
    [[nodiscard]] virtual auto rank() const -> ErrorOr<int>;
};

struct Intracomm {};

} // namespace einsums::mpi

template <>
struct fmt::formatter<einsums::mpi::ThreadLevel> {
    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        // [ctx.begin(), ctx.end()) is a character range that contains a part of
        // the format string starting from the format specifications to be parsed,
        // e.g. in
        //
        //   fmt::format("{:f} - point of interest", point{1, 2});
        //
        // the range will contain "f} - point of interest". The formatter should
        // parse specifiers until '}' or the end of the range. In this example
        // the formatter should parse the 'f' specifier and return an iterator
        // pointing to '}'.

        // Please also note that this character range may be empty, in case of
        // the "{}" format string, so therefore you should check ctx.begin()
        // for equality with ctx.end().

        // Parse the presentation format and store it in the formatter:
        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw format_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the ThreadLevel p using the parsed format specification (presentation)
    // stored in this formatter.
    template <typename FormatContext>
    auto format(const einsums::mpi::ThreadLevel &p, FormatContext &ctx) const -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.
        switch (p) {
        case einsums::mpi::ThreadLevel::Single:
            return fmt::format_to(ctx.out(), "Single");
        case einsums::mpi::ThreadLevel::Funneled:
            return fmt::format_to(ctx.out(), "Funneled");
        case einsums::mpi::ThreadLevel::Serialized:
            return fmt::format_to(ctx.out(), "Serialized");
        case einsums::mpi::ThreadLevel::Multiple:
            return fmt::format_to(ctx.out(), "Multiple");
        }
    }
};

#endif