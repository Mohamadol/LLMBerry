#pragma once

/// Throw `exception(msg)` when `cond` is false.
///
/// Example:
///   ENSURE(a.ndim() == 2, std::invalid_argument, "Tensors must be 2D.");
#define ENSURE(cond, exception, msg) \
    do {                             \
        if (!(cond)) {               \
            throw exception(msg);    \
        }                            \
    } while (0)
