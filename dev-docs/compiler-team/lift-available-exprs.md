# LiftAvailableExprs

`LiftAvailableExprs` (`is.hail.expr.ir.lowering`):

- rewrites an IR into a weak A-normal form;
- deduplicates repeated pure computations;
- hoists bindings as far out as is sound.

The ANF is scaffolding, not the goal: it introduces many single-use bindings that are
trivial to inline away later. The payoff is the genuinely shared or hoisted values.

## Nomenclature

- **ANF** — A-normal form: every compound subexpression is bound to a name, so operators take
  only atoms (refs and constants) as operands.
- **CSE** — common-subexpression elimination.
- **LICM** — loop-invariant code motion.
- **PRE** — partial redundancy elimination.
- **GVN** — global value numbering: hash-consing expressions to value numbers (VNs), so every
  form of one value shares a number.

## Motivation

Humans write abstractions, and abstractions are never zero-cost, whatever the C++ committee
says.

Our Python client builds expressions as a DAG, and when serialising that DAG to the IR tree
it deduplicates nodes shared by object identity. But two structurally equal expressions built
separately are distinct objects, so it might still emit the same subexpression many times
over — and lowering (MatrixIR → TableIR → streams) then manufactures duplication the client
never saw. The client has the harder job, deduplicating a DAG; we get a tree and structural
equality, and so our life is much, much easier.

The result is IR like the left of this, when we would rather compile the right:

```ml
map (fun row -> row.idx + (global.x + global.x)) rows

↦

let t0 = global.x
    t1 = t0 + t0
in map (\row -> row.idx + t1) rows
```

One evaluation of `t1` instead of `2 × length rows`, and the compiler downstream sees a smaller
tree.

## The idea

If you remember one thing: **name every strict subexpression, remember what is named, and put
each name as high as is sound**. Everything else is bookkeeping about where names may live.

Readers who took a compilers course will recognise GVN-PRE (VanDrunen & Hosking, CC 2004): a
linear analysis assigning every expression a value number and computing anticipability *over
VNs*, followed by a rewrite that inserts hoisted bindings and eliminates redundancies against
a VN-keyed availability map. The twist is that Hail IR is a tree, not a CFG: program order is
dominance order, so the paper's Insert and Eliminate fuse into one rebuild pass; scoping does
the work of basic blocks and bit vectors; and neither phase iterates to a fixed point — a
tree has no back edges, and the constructs that would be (loops, stream bodies) are walls.

Four transformations fall out of the two phases:

- **naming** (weak ANF): each liftable child of a strict edge is bound to a fresh name in an
  enclosing block. "Weak" because atoms stay inline and non-strict children keep their shape.
- **CSE**: an expression whose value number matches one already bound and in scope collapses
  to a reference to that binding.
- **code motion**: a binding lands at the shallowest point at which its free names all
  resolve, so values invariant in a stream or loop body float out of it (LICM).
- **PRE**: an expression pinned under a conditional still rises to a block that is certain to
  evaluate it anyway — a later occurrence on the spine, or (for total heads) an occurrence on
  every branch.

## What may be named; what may be reused; what may move

Three questions, three mechanisms.

### Naming

`isLiftable` is a shape test — may this expression be bound to a name at all?

- **Atoms** (refs and constants) may not: a ref to the binding is no smaller or cheaper than
  the atom itself; naming one buys nothing.
- **Stream-typed expressions** may not: a stream is not a value but a deferred loop — the
  emitter compiles it to control flow at its point of consumption, and it may be consumed
  only once, so a shared name is exactly what we must not offer.
- **Void-typed strict children** are skipped by the traversal itself (there is no value to
  bind; `RunAgg`'s body is the one legitimate carrier), as are expressions containing
  **aggregator intermediates**. This one needs background, because the machinery is opaque.
  Aggregation in Hail is not an expression over values; it compiles to a mutable state
  machine threaded through the IR. Each aggregator owns a numbered state slot:

  - `InitOp` initialises the slot;
  - `SeqOp` folds one observation into it;
  - `CombOp` merges two slots (partial results from different partitions);
  - `SerializeAggs`/`DeserializeAggs` ship slots between workers;
  - `ResultOp` reads the final value out.

  These are imperative reads and writes of shared mutable state wearing expression syntax, so
  structural equality does not imply value equality: two identical `ResultOp`s denote
  different values if a `SeqOp` runs between them. Anything containing one is never named or
  reused.

Everything else is named. Whether the name may then be reused or moved is a property of the
whole subtree, not the head.

### Reuse

A named binding is published — offered to structurally equal expressions — unless an effect
or nondeterminism hides anywhere within it:

- **Void-typed operations** (`WriteMetadata` and friends) and the impure writers
  (`WritePartition`, `WriteValue`) exist for their effects; collapsing two equal forms
  would elide one.
- **`UUID4`** must draw a fresh value per occurrence, so two occurrences are never "the same"
  computation.
- **`Apply("index_bgen", …)`** — the registry's one effectful plain function is an effect by name.

Note that `ConsoleLog` and `Die` are not considered effects - `ForwardLets` moves them
freely — if every path leads to them, earlier placement is unobservable in the same sense.

The lookup side needs no guard of its own: an effectful head draws a fresh value number per
occurrence, so equal forms of an effect never share a number — nothing effectful is ever
published, and nothing effectful ever matches a published entry.

### Motion

Hoisting past a non-strict edge changes how many times the expression runs: an `If` branch
may never run, a stream body may run once per element or not at all. Motion comes in two
strengths, gated by two facts.

*Down-safe motion* asks only that the expression be **clean** — deeply pure, deterministic,
and free of aggregator dependence, throughout the subtree. A placement is *down-safe* when
the expression is evaluated on every path leaving it; an anticipated later occurrence on the
spine witnesses exactly that. Down-safety never inspects the head: even a failure-capable
one — `mod`, a registered `Apply` — may hoist, firing its failure earlier rather than adding
one. (Busy occurrences — one on every branch of a conditional — license a weaker form that
*does* inspect the head; see the busy arm below.)

*Speculation* crosses an edge the source may have evaluated zero times (a possibly-empty
loop body), forcing an evaluation that never happened. That is unobservable only if the head
is also **total** — defined and error-free on every well-typed input. `isTotal` is a
conservative whitelist: building and projecting structs, tuples and arrays, casts,
comparisons, overflow-wrapping arithmetic. The instructive absences: integer division throws
on zero, and `Apply` — registered functions in general — may raise user-visible errors.

Everything that is not clean is named in place, preserving evaluation count and order.

## Worked examples

Deduplication and alias forwarding:

```ml
(a + 1, a + 1)  ↦  let t = a + 1 in (t, t)

let x = a + 1   ↦  let x = a + 1 in x + x
    y = a + 1 
in x + y
```

In the second, `y`'s value collapses to the ref `x`; rather than emit `let y = x`, the pass
records `y ↦ x` and rewrites uses. Expressions built over the alias then dedup too: `y + y`
becomes `x + x` and matches an available `x + x`.

LICM — a constant (or any total expression whose free names are bound  outside) rises through
a stream-body edge:

```ml
map (fun _ -> "a") s  ↦  let t = "a" in map (fun _ -> t) s
```

PRE — `a + 1` in the branch is *anticipated*: every execution of  this block evaluates
`a + 1` later, so hoisting it moves an evaluation earlier rather than adding one, and the
later occurrence collapses onto it:

```ml
let x = if c then a + 1 else b + 1  ↦  let y = a + 1
    y = a + 1                              x = if c then y else b + 1
in x + y                                in x + y
```

The same licence applies when the later occurrence is in the *same* binding's value
(`let x = (if c then a + 1 else b) * (a + 1)`), when the branch sits inside a nested block,
and when the later occurrence is a sibling on a non-block spine (`(if c then e else b, e)`).
Being down-safe, it also extends to failure-capable heads like `mod`: the anticipated
occurrence guarantees the failure was coming anyway.

Busy expressions — a conditional's branches are exhaustive alternatives, so a value on the
spine of *every* branch is evaluated whichever runs, and hoists even with no occurrence on
the enclosing spine:

```ml
if c then a + 1 else (a + 1) * b  ↦  let t = a + 1 in if c then t else t * b
```

Here totality is load-bearing where anticipation's licence was not: a *missing* predicate
skips all branches (the emitter jumps straight to the missing continuation), so busy-ness
proves evaluation only conditional on the predicate being present. A total head evaluated on
those rows is unobservable; a failure-capable one (`mod`) could fail on a row the source
never touched, so non-total busy values stay pinned.

## Aggregation contexts

Hail expressions occupy one of three scopes — `EVAL`, `AGG`, `SCAN` — and edges may shift
them (an `AggFilter`'s condition is an agg-scope expression; `StreamAgg` creates a fresh agg
context; `AggFold` promotes its child into the element scope). Two rules:

- an aggregation never crosses a non-strict edge — but CSE still applies *within* a context:

  ```ml
  streamAgg (fun _ -> (count (), count ())) xs
  ↦
  streamAgg (fun _ -> let n = count () in (n, n)) xs
  ```

- structural equality is not value equality across a context change. A count inside
  `aggFilter` is not the total count, so lookup must not reuse an entry bound over different
  elements.

  Edges that *reinterpret* the aggregable elements while agg expressions stay legal — the
  transformers `AggFilter`, `AggExplode`, `AggGroupBy`, `AggArrayPerElement` on their agg
  child; `StreamAgg`/`StreamAggScan` bodies — erect a lookup **wall**: agg- (scan-) dependent
  lookups see nothing bound above it. Edges that remove or replace the scope (`AggFold`'s
  promote/drop children; `TableAggregate`) need no wall — the stale entries are swapped out
  of reach.

Anticipation is not computed in agg or scan scope: a block's spine describes one evaluation,
not one evaluation per aggregated element, so PRE is eval-only.

## What we don't do

- **Speculating `Apply`**: registered functions take down-safe hoists — anticipation never
  adds an evaluation — but never cross a possibly-zero-evaluation edge, and never hoist on a
  busy licence. We could enable both later by adding a `noThrow` property on `Apply`.
- **Anticipation under agg/scan**, per the above.
- **Busy hoists out of `Coalesce` and short-circuit `lor`/`land`**: their arms are not
  exhaustive alternatives, so a shared arm expression is not certain to run.

## Implementation

The pass runs two phases per region. **Analyze** is one post-order traversal of the source
that stamps the marks, assigns value numbers, and records each node's anticipated set. Each
region numbers its values afresh: VNs never enter the IR — the canonical `__vn_` names
live only in the table's keys — so numbering restarting at each region is harmless.
**Rebuild** is one recursive rewrite producing the weak-ANF output, consulting the frozen
analysis: availability is keyed by VN, and anticipation probes are VN-set membership, so
motion cannot hide a match behind renaming.

### Regions, frames, levels

A **region** is the traversal state of one root IR expression: relational nodes lift each of
their IR children as a region of its own. Within one, names may be shared anywhere except
across a `dropEval` edge (e.g. `TableAggregate`'s query, `RelationalLet`'s value): such an
edge opens an ordinary frame whose `regionFloor` is raised — a hard floor no placement, not
even an anticipated hoist, may cross — and its transition clears the eval availability
entries for the duration, so nothing bound outside the edge is found within. A created agg
scope still tags the levels beneath it so bindings materialise with the right scope (a
`TableAggregate` aggregand gets AGG-tagged bindings at the query root):

```ml
("a", tableAggregate t ("a", "a"))
↦
let x = "a" in (x, tableAggregate t (let y = "a" in (y, y)))

tableAggregate t (max (row.idx * row.idx))
↦
tableAggregate t (agglet { t0 = row.idx; t1 = t0 * t0 } in max t1)
```

A **frame** is a landing spot for bindings: one at the root plus one per non-strict edge
on the path from the region root to the current expression. Frames are numbered by depth —
their **level**. Each frame owns an append-only `IRBuilder`; when the frame closes, its
accumulated bindings wrap the child in a `Block`. Each frame carries:

- `floor` — the deepest *conditionally evaluated* frame at or above it (`If`/`Switch` arms,
  `Coalesce` operands, short-circuit `lor`/`land` right operands, and strict-but-void
  children). Nothing may be placed shallower than the floor, except by anticipation.

  Stream and loop bodies do **not** raise the floor: hoisting a total expression out of a
  possibly-empty loop is safe and almost always profitable, whereas hoisting out of one
  branch of an `If` is a coin toss. The floor is a profitability guard, not a soundness one —
  soundness is `isTotal`'s job.
- `regionFloor` — the deepest `dropEval` frame at or above it; a hard floor that not even
  anticipation may cross.
- `aggWall`, `scanWall` — the deepest agg (scan) context change at or above it; bounds
  lookup as described above.
- `anticipated` — the VNs of the frame's site, fixed at frame creation (see below); a probe
  hit licenses a down-safe hoist to this frame.
- `escapes` — the shallowest enclosing level the frame's block still references; returned as
  the wrapped child's availability level when the frame closes.

Every subexpression is lifted to a pair `(newIR, level)`: the rewritten expression and its
**availability level**, the maximum level at which any of its free names is bound (names
bound outside the region contribute 0). That level is the shallowest frame the expression
could be placed at.

Operand ordering needs no fix-up pass:

- within a frame, builders are append-only and the traversal follows evaluation order, so a
  definition is appended before any expression using it is processed;
- across frames, an expression placed at level `l` only references names bound at levels
  `≤ l`, whose blocks lexically enclose it;
- the PRE hoist preserves this too: the decision fires while folding the conditional's
  children, so the hoisted binding is appended *before* the conditional itself is memoised.

### AvailableExprs

Per region, the expressions already bound to a name and eligible for reuse: three buffers —
`eval`, `agg`, `scan` — mirroring Hail's binding environments. Each holds one immutable map
`VN → Atom` per frame, tagged with the scope in which a binding landing at that level must
be declared. `eval` is always the buffer that lookups and insertions target; `agg` and
`scan` (each optional) hold entries usable per element of an enclosing aggregation.

Scope transitions swap whole buffers, in lockstep across every frame:

- **promote** installs the element environment as current (lifting an AGG binding's value
  evaluates it per element);
- **drop** discards a compartment and its entries;
- **create** opens a fresh context whose element environment extends the current eval
  entries (a shallow copy — slot immutability makes this copy-on-write).

A three-reference snapshot is saved before the edge and restored after: additions to
surviving compartments ride along; additions to dropped or created ones die with the buffer.

Lookup searches from the deepest frame outward and, for agg- (scan-) dependent expressions,
stops at the wall.

### Aliases

A block binding whose value collapses to an atom — a bare ref or a constant — becomes an
alias: uses are forwarded to the referent (each use takes a copy, preserving no-sharing) and
the binding dropped. The point is not tidiness: the alias keeps the *referent's* level — a
constant's is 0, having no free names — so expressions built on it may hoist past the frame
the alias was declared in. Dedup needs no forwarding help: the alias's name carries the
referent's value number, so `x + x` under `x ↦ xt` and `xt + xt` share a number by
construction, as do forms over an alias and over the constant itself (`x + a` and
`5 + a` under `x ↦ 5`).

Only atoms are forwarded: they are safe to duplicate and no larger than the ref they
replace. `Literal`, `EncodedLiteral` and `Str` are not atoms — duplicating them into use
sites would trade one binding for many copies — so they stay bound.

### Value numbering

Phase 1 hash-conses every node of the region to a **value number**: two expressions share a
VN exactly when the analysis can prove they denote the same value. The table key is the
node's *canonical form* — itself with each IR child replaced by a `Ref` naming the child's
VN — so structurally distinct forms of one value unify bottom-up. The rules:

- A name bound to a liftable value free of effects and aggregator intermediates shares the
  value's VN, so aliases and lifted nests vanish by construction: `ArrayLen(t)` and
  `ArrayLen(ToArray(s))` share a number when `t = ToArray(s)`. Agg-result dependence does not
  block the sharing — a VN is a key, and the walls guard reuse at lookup. Every other name —
  lambda and loop parameters, free names, relational refs, effectful or agg-intermediate
  bindings — is **opaque**: one fresh VN per name, so canonical forms never unify through one.
- Effect-bearing heads and aggregator-state reads draw a fresh VN per *occurrence*: equal
  effects are never "the same" computation. An effect deeper inside makes the enclosing VNs
  unique automatically, through the child's number.
- A block is transparent — its VN is its body's — when its bindings are free of effects and
  aggregator intermediates; otherwise it, too, is opaque.
- Relational children stay verbatim in the key (deep structural equality; rare).

The rebuild extends the same table: a rebuilt node's canonical form is hash-consed against
it, and a published binding's name carries its value's VN, so rebuilt forms stay
numbered like their sources — `ArrayLen(t)` in the output answers to the same number the
analysis gave `ArrayLen(ToArray(s))` in the source. A VN match is a *key*, never a
*licence*: reuse always goes through the walled availability lookup, and publication gates
on the marks.

### Anticipated sets

An expression is **anticipated** at a frame if it is certain to be evaluated whenever the
frame runs. Phase 1 records, for every node, the VNs certain to be evaluated whenever that
node is: its own (when clean and liftable), unioned across its strict, non-`dropEval`,
transition-free children — the same edges the rebuild names along — with block spines
recursing through EVAL binding values and the body, plus one extra arm at conditionals (the
busy arm, below). Computing anticipability *over VNs* is what keeps the frozen analysis
exact under motion: an alias contributes nothing new, a hoisted nest keeps its numbers, and
no form drifts out from under the sets. The analysis is a single post-order traversal;
a tree has no back edges, so nothing iterates.

Each frame fixes its anticipated set at creation — its site child's set, and the region root
takes the whole region's — because a site's own spine certainly runs whenever the site does.
This covers every position uniformly: block spines, non-block strict spines
(`(if c then e else b, e)` hoists `e`), and the region root itself. A VN evaluated on the
spine *behind* the probe point is found in availability first — lookup precedes placement —
so a probe hit never duplicates an evaluation, only moves one earlier.

Membership implies liftability and cleanliness, which guarantees the anticipated occurrence
dedups against the hoisted binding — otherwise the hoist would *add* an evaluation.

### The busy arm

At `If` and `Switch`, the anticipated set additionally takes the intersection of the
branches' sets (default included), filtered to **total** heads: a value on the spine of
every branch is evaluated whichever branch runs. The totality gate is the missingness
argument from the worked example: a missing predicate skips all branches, so busy-ness
proves evaluation only conditional on the predicate being present, and only a total head is
unobservable on the missing rows. Non-total VNs still enter anticipated sets through genuine
spine occurrences — down-safe as ever — so one set serves both licences. `Coalesce` and
short-circuit `lor`/`land` get no busy arm: their arms are not exhaustive alternatives.

The hoist needs no further machinery: inside the first branch, a busy value places above the
conditional — the decision fires while the conditional's children are being folded, so the
binding is appended before the conditional itself is memoised, in dependency order — and the
other branches' copies collapse onto it through ordinary availability.

### Marks

Four per-node facts steer naming, publishing, lookup and motion: does the node contain 
  - an aggregator intermediate (never named)?
  - an agg result (reusable, but only behind the matching wall)?
  - a scan result?
  - an effect or nondeterminism (named, never published or moved)?

Walking the subtree at every decision point would make the pass quadratic, so the facts are
cached in the shared `BaseIR.mark` word (`Memo`): each run reserves sixteen consecutive mark
values via `IrMetadata.nextFlags(16)` and stamps nodes with `base + bits`, one bit per fact.
Reads subtract the base and range-check, so a word stamped by any other pass — or never
stamped — decodes to zero, "clean".

`derive` computes a rebuilt node's facts from its head and its direct children's: a
one-level recurrence mirroring `ContainsAggIntermediate`, `ContainsAgg` and `ContainsScan`
(`Exists.scala`), plus the deep effect fact, seeded by `isEffect` at each head (what counts
as an effect is the Reuse section's business, encoded once there).

The recurrence is exact on both trees: the analysis stamps source nodes post-order, so its
filters (own-VN eligibility, name transparency, block transparency) read exact facts, and
the rebuild stamps every node it returns, including the wrapping `Block` a closing frame
builds, while atoms stay unstamped and correctly read clean — leaves compute nothing and
perform nothing.

Relational nodes are execution boundaries: their agg facts are their own business, but their
effects still happen, so a `UUID4` buried in a table pipeline (inside a `TableMapRows` body,
say) blocks reuse of the enclosing expression. This is one place the facts see more than the
walking predicates, which stop at IR children.

### Names

The input must satisfy `UniquelyNamed` (verified on entry), and the output must too. Fresh
names come from the compiler-wide `freshName()` (`__iruid_<n>`), so output uniqueness rests
on that counter's one contract: it only ever increases within a JVM.

## The `lift` traversal

`lift(region, ir)` has three cases.

**`Block`** — existing bindings stay in place: each value is re-lifted in its declared scope
(with the compartment transition applied around AGG/SCAN values), then republished under its
name — withheld from reuse if an aggregator intermediate or effect hides within; a value
collapsing to an atom becomes an alias. The body is lifted last and returned unwrapped —
flattening nested blocks into the frame. Hoists out of conditionals within a value are
licensed by the frame's fixed anticipated set: the value's own spine is in it, so a later
occurrence anywhere on the frame's spine counts.

**`LeafRef`** (`Ref`, `RelationalRef`) — resolve through the alias map; the availability
level is the referent's binding level (0 if bound outside the region).

**Everything else** — fold over the children, dispatching on the edge:

- *`dropEval` edge*: open a frame with `regionFloor` raised and eval availability cleared
  (see above); bindings dependent on the child land there and cannot escape.
- *Strict, non-void child*: lift it and, if liftable and free of aggregator intermediates,
  memoise it under a fresh name in the deepest frame — the ANF step; the binding is
  published for reuse unless it contains an effect. Its level joins the parent's running
  maximum.
- *Otherwise* (branch arm, stream or loop body, lazy operand, strict void child): open a
  fresh frame — floor raised if the edge is conditionally evaluated (or strict void), wall
  raised if the edge changes aggregation context — and lift the child inside it; bindings
  placed in the frame wrap the child in a block on the way out, and the frame's `escapes`
  joins the parent's level.

With all strict compound children now atoms, the node itself is decided:

- if an expression with the same value number is visible in `AvailableExprs` (respecting
  walls), reuse its atom;
- an expression that is not clean stays put — its parent memoises it in the current frame
  (a total head can hide an aggregation or an effect in an unliftable stream child, which
  pins no name; cleanliness is a property of the whole subtree, so it is still seen);
- a clean one may rise to its availability level: through the floor only where down-safe
  (it is anticipated at that level); to the floor regardless when its head is total. If the
  resulting level is strictly above the current frame, it is memoised there, published, and
  replaced by the ref;
- otherwise it, too, evaluates in place.

## Caveats

`IsStrict`/`NonStrict` — the per-edge strictness table — was synthesised from `Pretty`'s
`blockArgs` and from `Binds`, not from first principles. It has no independent oracle, and an
edge wrongly classified strict would move or dedup an expression across a lazy boundary.
Treat it as the pass's soft underbelly when auditing.

## Alternative Designs 

- **Syntax-based anticipation**.

  The first analysis compared *source syntax* against *rebuilt* queries while the rewrite 
  changed the tree under it. Three symptoms shared that root cause:

    - aliases and lifted nests hid matches, so the query side grew shallow alias resolution
      and an atom-unfolding map — and a miss was silent, a lost optimisation rather than an
      error
    - each spine position had to rescan its own suffix lazily (a cached suffix would miss
      aliases recorded after it), making anticipation quadratic
    - coverage was positional — the region root had no anticipated slot and non-block strict
      spines pushed none, so redundancy migrating to those positions never converged.

  We could have patched the holes one-by-one; used a dedicated anticipated slot for the region
  root (and another mechanism for non-block strict spines) would have fixed the known misses.

  Value numbers fix all these issues at once: an alias contributes nothing new to a VN set,
  a hoisted nest keeps its numbers, and "the VNs of this site" is one uniform rule with no
  position to forget.

- **Convergence by iteration.** 

  Rather than move a whole nest in one run, the pass could move one layer per run and let the
  surrounding `Optimize` loop find the fixed point. This was rejected as the loop's iteration 
  count is a heuristic. Canonical-form VNs make the point moot — a nest's inner and outer 
  layers share numbers with their hoisted forms, so whole nests move in a single run.

- **Hoisting busy values by surgery.**
  
  Lifting a value out of a conditional after the conditional is built means editing already
  emitted bindings. Instead the busy arm rides the ordinary anticipated-set probe: the decision 
  fires while the conditional's children are still being folded, so the hoisted binding is
  appended before the conditional itself — the same surgery-free moment as every other hoist,
  and `IRBuilder` stays append-only.
