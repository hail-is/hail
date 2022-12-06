Our design for pseudo-random number generation is inspired by [1], but several details differ. At a high level, the idea is:
* Assign to each random function invocation some unique identifier. In general we can't bound the size of the identifier. We use arrays of longs.
* Use a construction of a psuedo-random function to map unique identifiers to random streams of bits. Intuitively, it's as if we used the identifier to seed a stateful RNG.

The key property is that random function invocations with distinct identifiers produce independent random results, while invocations with the same identifier always produce the same result. Thus random function invocations are actually pure functions, with no side effects, which gives the compiler great freedom to optimize queries without affecting the results.

Psuedo-random functions are important building blocks in cryptography, and so they are very well studied, with many different practical constructions. We use the PMAC message authentication code, which depends on a tweakable block cipher, for which we use a reduced-round Threefish. Either or both of these pieces could be replaced with little effort, e.g. to improve performance.

# Threefish/Threefry:
We use the Threefish [2] block cipher, modified to use 20 rounds for efficiency (the full Threefish4x64 uses 72 rounds), as suggested by [3] (although we make use of the Threefish tweak). Reference implementation is `Threefry.encrypt`.

`threefish4x64` takes:
* key `K = (k_0, ..., k_3)`: 4 words
* tweak `T = (t_0, t_1)`: 2 words
* plaintext `P = (p_0, ..., p_3)`: 4 words

Intutively, this is a function taking a key and tweak as input, and returning a permutation on the space of all 256-bit blocks. The security claim is that if the key is chosen randomly, then for any choice of tweak, the resulting permutation "looks like" a uniformly chosen random permutation.

Like most (all?) block ciphers, it is constructed as a sequence of simpler permutations. Think of shuffling a deck of cards: each shuffle isn't that random (is easily distinguishable from a completely random permutation), but a sequence of seven shuffles is indistinguishable from a random permutation.

The simple permutations are called "rounds". Each round consists of applying a function "Mix" to pairs of 64-bit words, which is a bit-level permutation, followed by a permutation of the four words.

<img width="440" alt="threefish" src="https://user-images.githubusercontent.com/3430459/197852931-59bb6734-917b-4c4a-b3a2-7e2e9302a09f.png">

## key schedule
The key schedule turns the key and tweak into 6 subkeys, each 4 words. Subkey `s` is denoted `(k_{s,0}, ..., k_{s,3})`.

First compute two additional words `k_4 = C ^ k_0 ^ k_1 ^ k_2 ^ k_3` and `t_2 = t0 ^ t_1`, where `C = 0x1BD11BDAA9FC1A22`. Then
```
k_{s,0} = k_{s mod 5}
k_{s,1} = k_{s+1 mod 5} + t_{s mod 3}
k_{s,2} = k_{s+2 mod 5} + t_{s+1 mod 3}
k_{s,3} = k_{s+3 mod 5} + s
```

## an encryption round
Encryption is performed over 20 rounds. Let `v_i` be the `i`th word of the encryption state, initialized
```
v_i = p_i
```
Before round `d` if `d mod 4 = 0`, add subkey `s = d/4`
```
v_i += k_{s,i}
```
Then apply the `mix` function to adjacent pairs of words, where the rotation constant `r = R[d mod 8][j]` is looked up in a table.
```
mix(v_{2j}, v_{2j+1}, r)
```
`mix` is defined
```
mix(x0, x1, r) {
  x0 += x1
  rotL(x1, r)
  x1 ^= x0
}
```
<img width="190" alt="MIX" src="https://user-images.githubusercontent.com/3430459/197853087-ff3cee9d-002e-43a0-955b-7dd6747b90f2.png">

Lastly, the words are permuted
```
v_1, v_3 = v_3, v_1 
```

# PMAC
PMAC is a message authentication code. Intuitively, a MAC uses a block cipher to construct a function from abritrary length messages to 256 bit message tags. We extend this to a function from arbitrary length messages to "infinite" length message tags (really a very large finite length). The security claim is that if the block cipher used "looks like" a random permutation, then the MAC "looks like" a random function. In particular, for each message `m`, `pmac(m)` looks like a stream of random bits, and for distinct messages `m1` and `m2`, `pmac(m1)` and `pmac(m2)` look like completely independent streams of random bits. Yet this is a deterministic function, so computing `pmac` on the same message always produces the same stream of bits.

<img width="654" alt="Screen Shot 2022-10-25 at 11 53 13 AM" src="https://user-images.githubusercontent.com/3430459/197853175-d96ea5b0-4618-4f6c-b92a-0dfbf5da7fe2.png">

Many MAC constructions must process blocks sequentially. As we'll see below, this would add significant overhead to random number generation. PMAC has the property that blocks of the message can be processed in any order.

In our case, we use a modification of the PMAC1 construction in [4]. We restrict the message length to multiples of 64-bits for simplicity. Our modified PMAC is a function `pmac(nonce: Long, staticID: Long, message: Array[Long], counter: Long)`, defined as follows (reference implementation `Threefry.pmac`):
* Form a block `M[-1] = [nonce, staticID, 0L, 0L]`.
* Split `message` into blocks of 4 longs each, `M[0], ..., M[d]`, allowing the last block to be shorter.
* Let `E[i] = encrypt(key, [i, 0L], M[i])`, for `i=-1..d-1`, *all but the last block*
* Let `E` be the xor of all `E[i]`
* If the last block is not full, let `B` be `M[d]` padded by a single `1L` followed by `0L`s, to 4 longs. Otherwise, let `B = M[d]`.
* Compute the hash `H = E ^ B`.
* If the last block was full, compute the final MAC tag as
  * `T = encrypt(key, [-2, counter], H)`
* otherwise
  * `T = encrypt(key, [-3, counter], H)`
  
The counter is used to enable generating long streams of random numbers for each message, not just a single 256 bit tag. The intuition is that each message (plus nonce and staticID) gets reduced to a 256 bit hash, such that distinct messages are highly unlikely to have the same hash. Then for each value of the counter, we use a distinct random function (really a random permutation) from the space of hashes to the space of random outputs.

## Lazy computation
In practice, we don't need to save entire messages in memory. Instead we compute the hash on the fly.

The new type is `RNGState`. A value of this type consists of the data:
* `runningSum: IndexedSeq[Long]`: the xor of the encrypted contents of all full blocks
* `lastDynBlock: IndexedSeq[Long]`: the partial contents of the last block. The length of the sequence is `numWordsInLastDynBlock`
* `numWordsInLastDynBlock: Int`: the number of words (longs), in the range `[0, 4)`, currently contained in `lastDynBlock`
* `hasStaticSplit: Boolean`: whether the static block has been incorporated into `runningSum`
* `numDynBlocks: Int`: the number of completed blocks, not including the static block

This system is implemented using three IR nodes:
* `RNGStateLiteral` - creates an `RNGState` representing the empty message
* `RNGSplit(state: RNGState, dynBitstring: ?): RNGState` - appends to `lastDynBlock`. When the last block is full, encrypt it (using `numDynBlocks` for the tweak), and xor it into `runningSum`. Here `?` is either a single long, or an arbitrary sized tuple of longs.
* `ApplySeeded(..., rngState: RNGState, staticUID: Long)`
  * Statically, forms the static block `[nonce, staticUID, 0L, 0L]`, encrypts it, and embeds the result as a literal in the code.
  * At runtime, only needs to xor into the `runningSum` the encryped static block and the (possibly padded) `lastDynBlock`, and encrypt the result. Hence each `ApplySeeded` call only needs one invocation of the block cipher at runtime (more precisely, one invocation per 256 random bits needed by the random function). This minimizes the overhead of random number generation in inner loops, and is the reason for choose PMAC.

# UIDs
To use the above PMAC scheme, we need to assign a "message" to every random function invocation in the program. As long as each invocation gets a distinct message, the PMAC random function generates approximately independent randomness for each invocation.

We fix a key for the block cipher once and for all. It was generated randomly, and is hard coded in the compiler. This saves us from issues of users specifying "bad" keys. Instead, we reserve a part of the message to encode a session scoped uid. By changing that uid between sessions, we allow running identical code repeatedly with independent randomness.

## Static UIDs
We split the message into static and dynamic components. The static component consists of two longs. The first, called the "rng nonce", is a hail session constant. It replaces the old "global seed", allowing the same pipeline to run with independent randomness each session, unless determinism is specifically requested. The second component is stored in the `ApplySeeded` IR node. We simply maintain a global counter, and increment it each time an `ApplySeeded` node is constructed, ensuring that each node in a pipeline has a distinct static uid.

The dynamic component is needed to distinguish between different invocations of a single `ApplySeeded` node inside a looping construct. It is an arbitrary length message (though it will typically be quite small, probably less than 10 longs). It is constructed as follows:

## Dynamic UIDs
Every stream, table, or matrix table pipeline is transformed to explicitly generate a unique uid per stream entry, table row, and matrix table row/column. These uids are explicit in the IR as ordinary values/fields, so the compiler automatically preserves the RNG determinism.

## Putting it all together
Consider the example pipeline
```
mt = hl.utils.range_matrix_table(10, 10)
mt = mt.annotate_entries(a=hl.range(10).map(lambda i: hl.rand_int32(100)))
```
Before elaborating UIDs in the IR in python, the IR looks like this (after a little tidying):
```
!1 = MatrixRead [DropRowColUIDs, ...] // don't add uid fields
!3 = MatrixMapEntries(!1) {
(%g, %col, %row, %entry) =>
  !c0 = I32 [0]
  !c10 = I32 [10]
  !c1 = I32 [1]
  !s = StreamRange(!c0, !c10, !c1) [1, False]
  !s2 = StreamMap(!s) { (%elt) =>
    !c100 = I32 [100]
    ApplySeeded(!c100, %__rng_state) [rand_int32, 0, Int32] // unbound %__rng_state
  }
  !2 = ToArray(!s2)
  InsertFields !entry (a: !2)
}
```
Note that the `ApplySeeded` node is tageed with a static UID `0`, and references an unbound variable `__rng_state`. It is the responsibility of the `handle_randomness` pass to give proper definitions of `__rng_state` in any scope that needs it. After `handle_randomness` (and some more tidying), the IR looks like:
```
// Now MatrixRead adds row and col uids
!1 = MatrixRead [None, False, False, (MatrixRangeReader MatrixRangeReaderParameters(10,10,None) 8)]
!11 = MatrixMapEntries(!1) {
(%g, %col, %row, %entry) =>
  !2 = RNGStateLiteral  // RNGState corresponding to empty message
  !3 = GetField(%row) [__row_uid]   // get row and col uids
  !4 = GetField(%col) [__col_uid]
  !5 = MakeTuple(!3, !4) [(0 1)]
  %6 = RNGSplit(!2, !5)   // append row and col uids to message
  !c0 = I32 [0]
  !c10 = I32 [10]
  !c1 = I32 [1]
  !s = StreamRange(!c0, !c10, !c1) [1, False]
  !s2 = StreamMap(!s) { (%elt) =>
    !7 = Cast(%elt) [Int64]
    MakeTuple(!7, %elt) [(0 1)]   // map to stream of (uid, elt) pairs
  }
  !s3 = StreamMap(!s2) { (%elt2) =>
    !8 = GetTupleElement(%elt2) [0]
    %9 = RNGSplit(%6, !8)   // append stream element uid to message
    !c100 = I32 [100]
    // call random function with current message/RNGState %9 and static uid 0
    ApplySeeded(!c100, %9) [rand_int32, 0, Int32]
  }
  !10 = ToArray(!s3)
  InsertFields !entry (a: !10)
}
```
Note that because only 3 longs are added to the message, none of the `RNGSplit` calls generate any runtime code. They simply encode statically that the last block of the message at the time of the `ApplySeeded` call consists of the locals `[!3, !4, !8]`. Then the `ApplySeeded` just needs to pad the last block, xor it with the running sum (which is the encrypted static block, embedded as a constant in the code), and call the Threefry `encrypt` function just once.

# Security
Cryptogrophers have developed a very pragmatic theory of what makes for "good" pseudorandomness. One of the benefits of using cryptographic primitives (even while weakening some of the components for performance, as we do with Threefish) is that we can use this framework to evaluate how well users can trust the outputs of the RNG.

Using this theory for a quick sanity check, consider a pipeline with a 1e7 row by 1e7 column matrixtable, with 1e4 random function invocations per entry, running for a year on 1e23 cores. Let `b` be any boolean output of this pipeline. Let `P_1` and `P_2` be the probabilities that `b=1` in the scenarios where random functions are truly random, and using the above scheme, respectively. Then `abs(P_1 - P_2) < 3e-40`.

The only assumption in this bound is that Threefry is a secure block cipher, i.e. that the best attack against it is a brute force search of the space of all keys. The time bound comes from limiting how much of the key space the program is able to search. Clearly this will never be the weak link, and we can focus on how many random numbers are generated.

This is a very practically reasurring result. It says that users can really trust that their results--interpreted under a model of true randomness--are not skewed by our implementation of pseudorandomness.

# User interface
For the most part, users should not need to interact directly with the randomness infrastructure. If they don't, the default semantics are:
* Evaluating a hail expression multiple times in the same session always produces the same results
* Rebuilding an identical hail expression (e.g. `x = hl.rand_unif()` and `y = hl.rand_unif()`) evaluates with independent randomness.
* Running the same pipeline in multiple hail sessions uses independend randomness each time.

The last two can be overridden if needed:
* To build identical expressions using the same randomness, manually specify "seeds" (should we rename this?) on each random function call. E.g. `x = hl.rand_unif(seed=0)`. This overrides using the global counter to populate the static uid. It is guaranteed that user specified static uids never clash with automatically generated ones.
* To run the same pipeline in multiple sessions with the same randomness, manually specify the "global seed" on init: `hl.init(global_seed=0)`.


[1] "Splittable pseudorandom number generators using cryptographic hashing"
[2] "The Skein Hash Function Family"
[3] "Parallel random numbers: as easy as 1, 2, 3"
[4] Rogaway, "Efficient Instantiations of Tweakable Blockciphers and Refinements to Modes OCB and PMAC"
