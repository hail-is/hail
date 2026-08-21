package is.hail.expr.ir.lowering

import is.hail.ParameterizedTest
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.Scope._
import is.hail.expr.ir.agg.{PhysicalAggSig, TypedStateSig}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.implicits.forTesting._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.encoded.EBinaryOptional
import is.hail.types.physical.PInt64
import is.hail.types.virtual._

import org.scalactic.{Equivalence, Prettifier}
import org.scalatest.matchers.{MatchResult, Matcher}
import org.scalatest.matchers.dsl.MatcherFactory1
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

class LiftAvailableExprsSuite {

  private def liftTo(expected: BaseIR)(implicit ctx: ExecuteContext) =
    new MatcherFactory1[BaseIR, Equivalence] {
      override def matcher[T <: BaseIR: Equivalence]: Matcher[T] =
        Matcher[BaseIR] { input =>
          val lifted = LiftAvailableExprs(ctx, input)

          val prettyDiff =
            s"""  before = \n${Pretty.sexprStyle(input).trim}\n
               |   after = \n${Pretty.sexprStyle(lifted).trim}\n
               |expected = \n${Pretty.sexprStyle(expected).trim}\n
               | """.stripMargin

          MatchResult(
            matches = lifted isAlphaEquiv expected,
            rawFailureMessage = s"The lifted IR was not alpha-equivalent:\n$prettyDiff",
            rawNegatedFailureMessage = s"The lifted IR was alpha-equivalent:\n$prettyDiff",
          )
        }
    }

  implicit def irPrettifier(implicit ctx: ExecuteContext): Prettifier = {
    case ir: BaseIR => Pretty(ctx, ir)
    case x => Prettifier.default(x)
  }

  def ref(typ: Type) = Ref(Name("#undefined"), typ)
  def a = Ref(Name("a"), TInt32)
  def b = Ref(Name("b"), TInt32)
  def s = Ref(Name("sierra"), TStream(TInt32))
  def s2 = Ref(Name("tango"), TStream(TInt32))
  def cond = Ref(Name("cond"), TBoolean)
  def str = Str("a")

  // A binding that can neither be forwarded (not a ref) nor speculated (not total)
  def bindings = FastSeq(Binding(Name("x"), Die(ref(TString), TInt32, -1)))

  def testLiftAvailableExprs =
    FastSeq(
      // -- Atoms and constants are unchanged --
      I32(1) -> I32(1),
      ref(TInt32) -> ref(TInt32),
      str -> str,

      // -- Nested Blocks flatten --
      bindIR(a + b)(x => bindIR(b + a)(y => x + y)) ->
        bindIRs(a + b, b + a) { case Seq(x, y) => x + y },

      // -- Bindings whose value is a ref are forwarded to the referent --
      bindIR(a)(x => bindIR(b)(y => x + y)) -> (a + b),

      // -- Bindings whose value is a constant atom are forwarded, like refs --
      bindIR(I32(5))(x => maketuple(x, x)) -> maketuple(I32(5), I32(5)),

      // -- Forwarding unifies spellings: expressions built on the alias and on
      // the constant itself dedup to one binding --
      bindIR(I32(5))(x => maketuple(x + a, I32(5) + a)) ->
        bindIR(I32(5) + a)(t => maketuple(t, t)),

      // -- Blocks in binding values are lifted --
      bindIR(bindIR(a + I32(1))(y => y + I32(1)))(x => x + x) ->
        M.eval {
          for {
            y <- a + I32(1)
            x <- y + I32(1)
          } yield x + x
        },

      // -- Weak a-normal form: compound strict children are bound at the region root --
      maketuple(a + I32(1)) ->
        bindIR(a + I32(1))(a => maketuple(a)),

      // -- Common strict subexpressions dedup to one binding --
      maketuple(a + I32(1), a + I32(1)) ->
        bindIR(a + I32(1))(y => maketuple(y, y)),

      // -- Later occurrences of a named binding's value resolve to that binding --
      bindIR(a + 1)(x => maketuple(a + 1, x)) ->
        bindIR(a + 1)(x => maketuple(x, x)),

      // -- Bindings with a common value are forwarded to the first --
      bindIRs(a + 1, a + 1)(_ => ref(TInt32)) ->
        bindIR(a + 1)(_ => ref(TInt32)),

      // -- Lifting through strict positions --
      // Binary op: both args, bindings merged left-to-right
      bindIR(a + 1)(_ => ref(TInt32)) + bindIR(b + 1)(_ => ref(TInt32)) ->
        M.eval {
          for {
            _ <- a + 1
            _ <- b + 1
          } yield ref(TInt32) + ref(TInt32)
        },
      // Ifs
      If(Block(bindings, ref(TBoolean)), I32(1), I32(0)) ->
        Block(bindings, If(ref(TBoolean), I32(1), I32(0))),
      // Strict stream and zero args of a fold
      foldIR(Block(bindings, s), ref(TInt32))((_, _) => ref(TInt32)) ->
        Block(bindings, foldIR(s, ref(TInt32))((_, _) => ref(TInt32))),
      M.eval {
        for {
          x <- If(cond, a + 1, b + 1)
          y <- a + 1
        } yield x + y
      } -> M.eval {
        for {
          y <- a + 1
          x <- If(cond, y, b + 1)
        } yield x + y
      },

      // -- Anticipation includes the current binding's own spine: a later
      // occurrence within the same value licenses the hoist --
      M.eval {
        for {
          x <- If(cond, a + 1, b + 1) * (a + 1)
        } yield x
      } -> M.eval {
        for {
          t <- a + 1
          u <- If(cond, t, b + 1)
          x <- u * t
        } yield x
      },

      // -- Anticipation requires the later occurrence on the unconditional
      // spine: one hidden inside another If's branch does not license the
      // hoist --
      M.eval {
        for {
          x <- If(cond, a + 1, b + 1)
          y <- If(cond, a + 1, I32(0))
        } yield x + y
      } -> M.eval {
        for {
          x <- If(cond, a + 1, b + 1)
          y <- If(cond, a + 1, I32(0))
        } yield x + y
      },

      // -- Anticipation licenses motion of any deeply pure, deterministic
      // expression: a failure-capable head like mod is down-safe at its
      // spine occurrence, firing the failure earlier rather than adding
      // one --
      {
        def mod = Apply("mod", FastSeq(), FastSeq(a, I32(2)), TInt32)

        M.eval {
          for {
            x <- If(cond, mod, b + 1)
            y <- mod
          } yield x + y
        } -> M.eval {
          for {
            y <- mod
            x <- If(cond, y, b + 1)
          } yield x + y
        }
      },

      // -- Anticipation covers the region root's own spine: a later
      // occurrence as a sibling strict child licenses the hoist, with no
      // enclosing Block in the source --
      {
        def mod = Apply("mod", FastSeq(), FastSeq(a, I32(2)), TInt32)

        If(cond, mod, b + 1) + mod ->
          M.eval {
            for {
              t <- mod
              x <- If(cond, t, b + 1)
            } yield x + t
          }
      },

      // -- ...including non-Block strict spines like a tuple's fields --
      {
        def mod = Apply("mod", FastSeq(), FastSeq(a, I32(2)), TInt32)

        maketuple(If(cond, mod, b + 1), mod) ->
          M.eval {
            for {
              t <- mod
              x <- If(cond, t, b + 1)
            } yield maketuple(x, t)
          }
      },

      // -- ...and every non-strict site's spine: a hoist out of a
      // conditional in a loop body lands at the body frame where the
      // element is bound, not outside the loop --
      {
        def mod(x: IR) = Apply("mod", FastSeq(), FastSeq(x, I32(2)), TInt32)

        s.streamMap(x => If(cond, mod(x), b + 1) + mod(x)) ->
          s.streamMap { x =>
            M.eval {
              for {
                t <- mod(x)
                y <- If(cond, t, b + 1)
              } yield y + t
            }
          }
      },

      // -- Busy expressions: a total value on the spine of every branch
      // is evaluated whichever branch runs, so it hoists above the
      // conditional --
      M.eval {
        for {
          x <- If(cond, a + 1, (a + 1) * b)
        } yield x
      } -> M.eval {
        for {
          t <- a + 1
          x <- If(cond, t, t * b)
        } yield x
      },

      // -- A non-total busy head stays pinned: a missing predicate skips
      // both branches, and a hoisted failure-capable op would run on
      // those rows --
      {
        def mod = Apply("mod", FastSeq(), FastSeq(a, I32(2)), TInt32)

        If(cond, mod, mod * b) ->
          If(cond, mod, bindIR(mod)(t => t * b))
      },

      // -- The busy intersection spans a Switch's default and cases --
      Switch(b, a + 1, FastSeq((a + 1) * b)) ->
        bindIR(a + 1)(t => Switch(b, t, FastSeq(t * b))),

      // -- A busy nest hoists whole in one run: the shared VNs compose
      // bottom-up --
      {
        def len = ArrayLen(MakeArray(FastSeq(a), TArray(TInt32)))

        If(cond, len, len + 1) ->
          M.eval {
            for {
              t <- MakeArray(FastSeq(a), TArray(TInt32))
              u <- ArrayLen(t)
            } yield If(cond, u, u + 1)
          }
      },

      // -- Equal effects in both branches are not busy: each occurrence
      // has its own value --
      {
        def u = maketuple(UUID4("rng"))

        If(cond, u, u) ->
          If(cond, bindIR(UUID4("rng"))(x => maketuple(x)), bindIR(UUID4("rng"))(x => maketuple(x)))
      },

      // -- A UUID4 hiding under a non-strict edge blocks reuse
      {
        def u = If(cond, UUID4("rng"), str)

        maketuple(u, u) -> bindIRs(u, u) { case Seq(x, y) => maketuple(x, y) }
      },

      // -- ...and blocks the down-safe hoist: the branch draw and the
      // spine draw are distinct values --
      {
        def u = If(cond, UUID4("rng"), str)

        M.eval {
          for {
            x <- If(cond, u, str)
            y <- u
          } yield maketuple(x, y)
        } -> M.eval {
          for {
            x <- If(cond, u, str)
            y <- u
          } yield maketuple(x, y)
        }
      },

      // -- index_bgen, the registry's one effectful function, is E by
      // name: equal calls are not collapsed --
      {
        def idx = Apply(
          "index_bgen",
          FastSeq(TStruct("contig" -> TString, "position" -> TInt32)),
          FastSeq(
            ref(TString),
            ref(TString),
            ref(TDict(TString, TString)),
            ref(TBoolean),
            ref(TInt32),
          ),
          TInt64,
        )

        maketuple(idx, idx) -> bindIRs(idx, idx) { case Seq(x, y) => maketuple(x, y) }
      },

      // -- An effect hiding under a non-strict edge blocks reuse too:
      // collapsing equal spellings would drop a write --
      {
        def w = If(
          cond,
          WriteValue(
            ref(TString),
            ref(TString),
            ETypeValueWriter(TypedCodecSpec(EBinaryOptional, TString, BufferSpec.default)),
          ),
          str,
        )

        maketuple(w, w) -> bindIRs(w, w) { case Seq(x, y) => maketuple(x, y) }
      },

      // -- An effect buried in a relational child blocks reuse too: each
      // aggregate runs its pipeline, drawing its own UUID --
      {
        def agg = TableAggregate(
          TableRange(10, 1).mapGlobals(_.insert("u" -> UUID4("rng"))),
          ApplyAggOp(Count())(),
        )

        def expected = TableAggregate(
          TableRange(10, 1).mapGlobals(g => bindIR(UUID4("rng"))(u => g.insert("u" -> u))),
          ApplyAggOp(Count())(),
        )

        maketuple(agg, agg) -> bindIRs(expected, expected) { case Seq(x, y) => maketuple(x, y) }
      },

      // -- A movable-but-not-total head still takes a down-safe hoist:
      // the spine occurrence guarantees the evaluation happens anyway.
      // The enclosing ArrayLen hoists too: rebuilt as ArrayLen(t), its
      // unfolded spelling matches the anticipated set --
      M.eval {
        for {
          x <- If(cond, ArrayLen(ToArray(s2)), b)
          y <- ArrayLen(ToArray(s2))
        } yield x + y
      } -> M.eval {
        for {
          t <- ToArray(s2)
          u <- ArrayLen(t)
          x <- If(cond, u, b)
        } yield x + u
      },

      // -- Unfolding composes through nesting: each hoisted layer's
      // binding is recorded in source spelling, so a two-deep nest
      // hoists fully in one run --
      M.eval {
        for {
          x <- If(cond, Cast(ArrayLen(ToArray(s2)), TInt64), Cast(b, TInt64))
          y <- Cast(ArrayLen(ToArray(s2)), TInt64)
        } yield maketuple(x, y)
      } -> M.eval {
        for {
          t <- ToArray(s2)
          u <- ArrayLen(t)
          v <- Cast(u, TInt64)
          x <- If(cond, v, Cast(b, TInt64))
        } yield maketuple(x, v)
      },

      // -- Anticipation sees through nested blocks: the outer block's suffix
      // licenses a hoist from a branch inside an inner block's binding --
      M.eval {
        for {
          x <- M.eval {
            for {
              p <- If(cond, a + 1, b + 1)
            } yield p * p
          }
          y <- a + 1
        } yield x + y
      } -> M.eval {
        for {
          t <- a + 1
          p <- If(cond, t, b + 1)
          x <- p * p
        } yield x + t
      },

      // -- Anticipation never hoists across a dropEval wall: a binding placed
      // outside ArrayMaximalIndependentSet's tie-breaker could not be
      // referenced from within it. AMIS is the one dropEval edge that leaves
      // both aggregation environments untouched, so only the region floor
      // pins the hoist --
      {
        def payload =
          maketuple(I32(1))

        def amis(tb: IR) =
          ArrayMaximalIndependentSet(
            ref(TArray(TTuple(TInt64, TInt64))),
            Some((Name("l"), Name("r"), tb)),
          )

        Block(
          FastSeq(Binding(Name("x"), amis(If(IsNA(payload), F64(0), F64(1))))),
          payload,
        ) ->
          Block(
            FastSeq(Binding(
              Name("x"),
              amis(M.eval {
                for {
                  t <- payload
                  c <- IsNA(t)
                } yield If(c, F64(0), F64(1))
              }),
            )),
            payload,
          )
      },

      // -- NOT lifting through non-strict positions --
      // If consequent
      If(True(), Block(bindings, ref(TInt32)), I32(0)) ->
        If(True(), Block(bindings, ref(TInt32)), I32(0)),
      // StreamMap body
      s.streamMap(_ => Block(bindings, ref(TInt32))) ->
        s.streamMap(_ => Block(bindings, ref(TInt32))),
      // -- A strict TVoid child (RunAgg's body) is never named: void has no
      // value to bind. Its spine still runs exactly once with the parent,
      // so a clean value within is anticipated at the region root and
      // hoists; the effectful InitOp stays in the body's own frame --
      {
        def sig = PhysicalAggSig(Sum(), TypedStateSig(VirtualTypeWithReq(PInt64(true))))

        def runAgg(body: IR) =
          RunAgg(body, ResultOp(0, sig), FastSeq(sig.state))

        runAgg(
          Begin(FastSeq(
            InitOp(0, FastSeq(), sig),
            SeqOp(0, FastSeq(Cast(a, TInt64)), sig),
          ))
        ) ->
          Block(
            FastSeq(Binding(Name("c"), Cast(a, TInt64))),
            runAgg(
              Block(
                FastSeq(Binding(Name("i"), InitOp(0, FastSeq(), sig))),
                SeqOp(0, FastSeq(Ref(Name("c"), TInt64)), sig),
              )
            ),
          )
      },

      // -- Duplicate constants dedup to one binding --
      maketuple(Str("a"), Str("a")) -> bindIR(Str("a"))(x => maketuple(x, x)),

      // -- Constants rise through non-strict loop/agg-body edges: StreamMap body --
      s.streamMap(_ => Str("a")) -> bindIR(Str("a"))(x => s.streamMap(_ => x)),

      // -- A movable-but-not-total expression does not: the stream may run
      // zero times, and a forced evaluation can fail --
      s.streamMap(_ => ToArray(s2)) -> s.streamMap(_ => ToArray(s2)),

      // -- Constants do NOT rise out of conditionally-evaluated children; a
      // bare constant in an isolated branch evaluates to itself --
      If(True(), Str("a"), Str("b")) -> If(True(), Str("a"), Str("b")),

      // -- ...but the same constant in every branch is busy and hoists --
      If(True(), Str("a"), Str("a")) ->
        bindIR(Str("a"))(x => If(True(), x, x)),

      // -- A Block binding whose value is a non-atom constant (Str) is left in
      // place: only atoms are safe to duplicate into use sites --
      Block(FastSeq(Binding(Name("x"), Str("a"))), ref(TInt32)) ->
        Block(FastSeq(Binding(Name("x"), Str("a"))), ref(TInt32)),

      // -- A constant inside a dropEval wall is bound at that subtree's root, not the outer
      // root; the TableAggregate, a compound strict child, is lifted to the outer root --
      maketuple(Str("a"), TableAggregate(TableRange(10, 1), maketuple(Str("a"), Str("a")))) ->
        bindIRs(
          Str("a"),
          TableAggregate(TableRange(10, 1), bindIR(Str("a"))(inner => maketuple(inner, inner))),
        ) { case Seq(outer, agg) => maketuple(outer, agg) },

      // -- A dropEval wall that creates an agg scope contains its whole region: aggregands
      // referencing agg-scope names get AGG-tagged bindings at the wall's root --
      {
        def idx = Ref(TableIR.rowName, TStruct("idx" -> TInt32)).get("idx")

        TableAggregate(TableRange(10, 1), ApplyAggOp(Max())(idx * idx)) ->
          TableAggregate(
            TableRange(10, 1),
            Block(
              FastSeq(
                Binding(Name("x"), idx, Scope.AGG),
                Binding(Name("y"), Ref(Name("x"), TInt32) * Ref(Name("x"), TInt32), Scope.AGG),
              ),
              ApplyAggOp(Max())(Ref(Name("y"), TInt32)),
            ),
          )
      },

      // -- Loop-invariant code motion: an expression rises to the shallowest frame at which
      // all its free names are bound, provided its head operation is total. Ref-valued
      // bindings forward to their referent, so `x + x` hoists as `xt + xt`. Expressions
      // depending on the RelationalRef are pinned by the RelationalLet body; constants and
      // their derivatives rise above the RelationalLet entirely --
      {
        val collect = TableRange(10, 1).mapGlobals(_.insert("x" -> I32(1))).collect
        val literal = Literal(TStruct("y" -> TInt32), RowSeq(0))

        relationalBindIR(collect) { rng =>
          mapArray(rng.get("rows")) { r =>
            r.get("idx") +
              bindIR(rng.get("global").get("x"))(x => x + x) +
              literal.get("y")
          }
        } ->
          M.eval {
            for {
              lit <- literal
              y <- lit.get("y")
            } yield relationalBindIR(collect) { rng =>
              M.eval {
                for {
                  rows <- rng.get("rows")
                  global <- rng.get("global")
                  xt <- global.get("x")
                  xx <- xt + xt
                } yield rows.stream.streamMap { r =>
                  M.eval {
                    for {
                      idx <- r.get("idx")
                      t1 <- idx + xx
                    } yield t1 + y
                  }
                }.toArray
              }
            }
          }
      },
      // -- Expressions using the RelationalRef are pinned inside the RelationalLet's body
      // (its value is a dropEval wall); expressions in the body may refer to dominating
      // bindings outside it --
      {
        val value = TableRange(10, 1).mapGlobals(_.insert("x" -> I32(1))).collect

        maketuple(Str("a"), relationalBindIR(value)(rng => maketuple(Str("a"), rng.get("rows")))) ->
          M.eval {
            for {
              outer <- Str("a")
              rlet <- relationalBindIR(value)(_.get("rows").bind(maketuple(outer, _)))
            } yield maketuple(outer, rlet)
          }
      },
      // -- Equal aggregations in the same context share a binding --
      rangeIR(0, 10).streamAgg { _ =>
        maketuple(ApplyAggOp(Count())(), ApplyAggOp(Count())())
      } -> rangeIR(0, 10).streamAgg { _ =>
        M.eval {
          for {
            count <- ApplyAggOp(Count())()
          } yield maketuple(count, count)
        }
      },

      // -- Aggregations are not deduplicated across a context-transforming
      // edge: the filtered count is not the total count --
      rangeIR(0, 10).streamAgg { elem =>
        maketuple(
          ApplyAggOp(Count())(),
          AggFilter(
            Apply("mod", FastSeq(), FastSeq(elem, I32(2)), TInt32) ceq 0,
            ApplyAggOp(Count())(),
            isScan = false,
          ),
        )
      } -> rangeIR(0, 10).streamAgg { elem =>
        M.eval {
          for {
            count <- ApplyAggOp(Count())()
            zero <- M.lift[AGG.type] {
              for {
                mod <- Apply("mod", FastSeq(), FastSeq(elem, I32(2)), TInt32)
                zero <- mod ceq 0
              } yield zero
            }
            filtered <- AggFilter(zero, ApplyAggOp(Count())(), isScan = false)
          } yield maketuple(count, filtered)
        }
      },

      // -- Structurally equal reads of mutable aggregator state are not
      // deduplicated: an InitOp or SeqOp may intervene between them --
      {
        def read = maketuple(ResultOp(
          0,
          PhysicalAggSig(Sum(), TypedStateSig(VirtualTypeWithReq(PInt64(true)))),
        ))

        maketuple(read, read) -> maketuple(read, read)
      },
    ) ++
      FastSeq(Scope.AGG, Scope.SCAN).flatMap { scope =>
        val isScan = scope == Scope.SCAN

        FastSeq(
          // -- EVAL -> AGG / SCAN
          AggFilter(
            Block(FastSeq(Binding(Name("x"), IsNA(ref(TBoolean)))), ref(TBoolean)),
            ref(TInt32),
            isScan,
          ) ->
            Block(
              FastSeq(Binding(Name("x"), IsNA(ref(TBoolean)), scope)),
              AggFilter(ref(TBoolean), ref(TInt32), isScan),
            ),

          // -- Nested blocks in AGG/SCAN context: all EVAL bindings promoted --
          AggFilter(
            Block(
              FastSeq(Binding(
                Name("x"),
                Block(
                  FastSeq(Binding(Name("y"), a + I32(1))),
                  Ref(Name("y"), TInt32) + I32(2),
                ),
              )),
              ref(TBoolean),
            ),
            ref(TInt32),
            isScan,
          ) ->
            Block(
              FastSeq(
                Binding(Name("y"), a + I32(1), scope),
                Binding(Name("x"), Ref(Name("y"), TInt32) + I32(2), scope),
              ),
              AggFilter(ref(TBoolean), ref(TInt32), isScan),
            ),
          // -- AGG/SCAN bindings in EVAL context are preserved --
          Block(FastSeq(Binding(Name("x"), b + I32(2), scope)), ref(TInt32)) + I32(1) ->
            Block(FastSeq(Binding(Name("x"), b + I32(2), scope)), ref(TInt32) + I32(1)),
          // -- A constant in an AGG context gets a Scope.AGG-tagged root binding --
          AggFilter(Str("a"), ref(TInt32), isScan) ->
            Block(
              FastSeq(Binding(Name("x"), Str("a"), scope)),
              AggFilter(Ref(Name("x"), TString), ref(TInt32), isScan),
            ),

          // -- The same constant in EVAL and AGG contexts gets two separate bindings; the
          // AggFilter, itself a compound strict child, is lifted too --
          maketuple(Str("a"), AggFilter(Str("a"), ref(TInt32), isScan)) ->
            Block(
              FastSeq(
                Binding(Name("x"), Str("a"), Scope.EVAL),
                Binding(Name("y"), Str("a"), scope),
                Binding(Name("z"), AggFilter(Ref(Name("y"), TString), ref(TInt32), isScan)),
              ),
              maketuple(Ref(Name("x"), TString), Ref(Name("z"), TInt32)),
            ),
        )
      }

  @ParameterizedTest
  def testLiftAvailableExprs(input: IR, expected: IR)(implicit ctx: ExecuteContext): Unit =
    input should liftTo(expected)
}
