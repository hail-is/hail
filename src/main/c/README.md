# Vectorized Hail
These are a few pointers to help orient you.

## Building for a non-native architecture

By default this code builds for the widest registers your current machine
supports. To prevent this set these two CXXFLAGS *and specify an architecture*
before calling `make`:

```
-DHAIL_OVERRIDE_ARCH
-DHAIL_OVERRIDE_WIDTH=4
```

The first variable prevents automatic architecture detection.

The second variable states how many 64-bit values should be packed into a vector
intrinsic. Unfortunately, this cannot be set to one because libsimdpp did not
define `extract<0>` for size-one vectors.

To specify an architecture, set one of the variables listed in the
[libsimdpp documentation](http://p12tic.github.io/libsimdpp/v2.0%7Erc2/libsimdpp/arch/selection.html).

## `uint64vector`

In `ibs.h`, we define `uint64vector` in terms of `libsimdpp` 64-bit vectors. By
default, we set the length of the vector to `SIMDPP_FAST_INT64_SIZE`, which is
set to the length (in 64-bit units) of the longest register available on this
machine.

## Cache Optimization

In `ibsMat` we perform a simple cache-blocking optimization. We assume that the
input genotype arrays are multiples of the block size
(i.e. `CACHE_SIZE_IN_MATRIX_ROWS`).

## Alignment

We do not assume any provided memory is 32-bit aligned.
