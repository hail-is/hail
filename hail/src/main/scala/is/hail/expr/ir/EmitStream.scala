package is.hail.expr.ir

import is.hail.utils._
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.types.physical._
import is.hail.annotations.{Region, StagedRegionValueBuilder}

import scala.language.existentials

object EmitStream {
  sealed trait Init[+S]
  object Missing extends Init[Nothing]
  case class Start[S](s0: S) extends Init[S]

  sealed trait Step[+A, +S]
  object EOS extends Step[Nothing, Nothing]
  case class Yield[A, S](elt: A, s: S) extends Step[A, S]

  def stepIf[A, S, X](k: Step[A, S] => Code[X], c: Code[Boolean], a: A, s: S): Code[X] =
    c.mux(k(Yield(a, s)), k(EOS))

  trait Parameterized[-P, +A] { self =>
    type S
    val stateP: ParameterPack[S]

    // - 'step' must maintain the following invariant: step(..., emptyState, k) = k(EOS); it should
    // hopefully be a cheap operation to compute this.
    // - emptyState is unfortunately needed in some situations to get around the lack of "Option[T]"s
    // being (easily) possible in bytecode.
    def emptyState: S

    def length(s0: S): Option[Code[Int]]

    def init(
      mb: MethodBuilder,
      jb: JoinPointBuilder,
      param: P
    )(k: Init[S] => Code[Ctrl]): Code[Ctrl]

    def step(
      mb: MethodBuilder,
      jb: JoinPointBuilder,
      state: S
    )(k: Step[A, S] => Code[Ctrl]): Code[Ctrl]

    def addSetup[Q <: P](setup: Q => Code[Unit], cleanup: Code[Unit] = Code._empty): Parameterized[Q, A] =
      guardParam({ (param, k) => Code(setup(param), k(Some(param))) }, cleanup)

    def guardParam[Q](
      f: (Q, Option[P] => Code[Ctrl]) => Code[Ctrl],
      cleanup: Code[Unit] = Code._empty
    ): Parameterized[Q, A] = new Parameterized[Q, A] {
      type S = self.S
      val stateP: ParameterPack[S] = self.stateP
      def emptyState: S = self.emptyState
      def length(s0: S): Option[Code[Int]] = self.length(s0)
      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: Q)(k: Init[S] => Code[Ctrl]): Code[Ctrl] = {
        val missing = jb.joinPoint()
        missing.define { _ => k(Missing) }
        f(param, {
          case Some(newParam) => self.init(mb, jb, newParam) {
            case Missing => missing(())
            case Start(s) => k(Start(s))
          }
          case None => missing(())
        })
      }
      def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[A, S] => Code[Ctrl]): Code[Ctrl] =
        self.step(mb, jb, state) {
          case EOS => Code(cleanup, k(EOS))
          case v => k(v)
        }
    }

    def map[B](f: A => B): Parameterized[P, B] =
      contMap[B] { (a, k) => k(f(a)) }

    def contMap[B](
      f: (A, B => Code[Ctrl]) => Code[Ctrl]
    ): Parameterized[P, B] = new Parameterized[P, B] {
      type S = self.S
      val stateP: ParameterPack[S] = self.stateP
      def emptyState: S = self.emptyState
      def length(s0: S): Option[Code[Int]] = self.length(s0)
      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
        self.init(mb, jb, param) {
          case Missing => k(Missing)
          case Start(s) => k(Start(s))
        }
      def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[B, S] => Code[Ctrl]): Code[Ctrl] =
        self.step(mb, jb, state) {
          case EOS => k(EOS)
          case Yield(a, s) => f(a, b => k(Yield(b, s)))
        }
    }

    def filterMap[B](f: (A, Option[B] => Code[Ctrl]) => Code[Ctrl]): Parameterized[P, B] = new Parameterized[P, B] {
      type S = self.S
      implicit val stateP: ParameterPack[S] = self.stateP
      def emptyState: S = self.emptyState
      def length(s0: S): Option[Code[Int]] = None
      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
        self.init(mb, jb, param)(k)
      def step(mb: MethodBuilder, jb: JoinPointBuilder, s0: S)(k: Step[B, S] => Code[Ctrl]): Code[Ctrl] = {
        val pull = jb.joinPoint[S](mb)
        pull.define(self.step(mb, jb, _) {
          case EOS => k(EOS)
          case Yield(a, s) => f(a, {
            case None => pull(s)
            case Some(b) => k(Yield(b, s))
          })
        })
        pull(s0)
      }
    }

    def scan[B: ParameterPack](dummy: B)(
      zero: B,
      op: (A, B, B => Code[Ctrl]) => Code[Ctrl]
    ): Parameterized[P, B] = new Parameterized[P, B] {
      implicit val sP = self.stateP
      type S = (self.S, B, Code[Boolean])
      val stateP: ParameterPack[S] = implicitly
      def emptyState: S = (self.emptyState, dummy, false)
      def length(s0: S): Option[Code[Int]] = self.length(s0._1).map(_ + 1)

      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
        self.init(mb, jb, param) {
          case Missing => k(Missing)
          case Start(s0) => k(Start((s0, zero, true)))
        }

      def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[B, S] => Code[Ctrl]): Code[Ctrl] = {
        val yield_ = jb.joinPoint[(B, self.S)](mb)
        yield_.define { case (b, s1) => k(Yield(b, (s1, b, false))) }
        val (s, b, isFirstStep) = state
        isFirstStep.mux(
          yield_((b, s)),
          self.step(mb, jb, s) {
            case EOS => k(EOS)
            case Yield(a, s1) => op(a, b, b1 => yield_((b1, s1)))
          })
      }
    }
  }

  val missing: Parameterized[Any, Nothing] = new Parameterized[Any, Nothing] {
    type S = Unit
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = ()
    def length(s0: S): Option[Code[Int]] = Some(0)
    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: Any)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      k(Missing)
    def step(mb: MethodBuilder, jb: JoinPointBuilder, s: S)(k: Step[Nothing, S] => Code[Ctrl]): Code[Ctrl] =
      k(EOS)
  }

  def range(
    start: Code[Int],
    incr: Code[Int]
  ): Parameterized[Code[Int], Code[Int]] = new Parameterized[Code[Int], Code[Int]] {
    // stream parameter will be the length of the stream
    type S = (Code[Int], Code[Int])
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = (0, 0)
    def length(s0: S): Option[Code[Int]] = Some(s0._1)

    def init(mb: MethodBuilder, jb: JoinPointBuilder, len: Code[Int])(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      k(Start((len, start)))

    def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[Code[Int], S] => Code[Ctrl]): Code[Ctrl] = {
      val (nLeft, acc) = state
      stepIf(k, nLeft > 0, acc, (nLeft - 1, acc + incr))
    }
  }

  def sequence[A: ParameterPack](elements: Seq[A]): Parameterized[Any, A] = new Parameterized[Any, A] {
    type S = Code[Int]
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = elements.length
    def length(s0: S): Option[Code[Int]] = Some(const(elements.length) - s0)

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: Any)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      k(Start(0))

    def step(mb: MethodBuilder, jb: JoinPointBuilder, idx: S)(k: Step[A, S] => Code[Ctrl]): Code[Ctrl] = {
      val eos = jb.joinPoint()
      val yld = jb.joinPoint[A](mb)
      eos.define { _ => k(EOS) }
      yld.define { a => k(Yield(a, idx + 1)) }
      JoinPoint.switch(idx, eos, elements.map { elt =>
        val j = jb.joinPoint()
        j.define { _ => yld(elt) }
        j
      })
    }
  }

  def compose[A, B, C](
    outer: Parameterized[A, B],
    inner: Parameterized[B, C]
  ): Parameterized[A, C] = new Parameterized[A, C] {
    implicit val outSP = outer.stateP
    implicit val innSP = inner.stateP
    type S = (outer.S, inner.S)
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = (outer.emptyState, inner.emptyState)
    def length(s0: S): Option[Code[Int]] = None

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: A)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      outer.init(mb, jb, param) {
        case Missing => k(Missing)
        case Start(outS0) => k(Start((outS0, inner.emptyState)))
      }

    def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[C, S] => Code[Ctrl]): Code[Ctrl] = {
      val stepInner = jb.joinPoint[S](mb)
      val stepOuter = jb.joinPoint[outer.S](mb)
      stepInner.define { case (outS, innS) =>
        inner.step(mb, jb, innS) {
          case EOS => stepOuter(outS)
          case Yield(innElt, innS1) => k(Yield(innElt, (outS, innS1)))
        }
      }
      stepOuter.define(outer.step(mb, jb, _) {
        case EOS => k(EOS)
        case Yield(outElt, outS) => inner.init(mb, jb, outElt) {
          case Missing => stepOuter(outS)
          case Start(innS) => stepInner((outS, innS))
        }
      })
      stepInner(state)
    }
  }

  def leftJoinRightDistinct[P, A, B: ParameterPack](
    left: Parameterized[P, A],
    right: Parameterized[P, B],
    rNil: B,
    comp: (A, B) => Code[Int]
  ): Parameterized[P, (A, B)] = new Parameterized[P, (A, B)] {
    implicit val lsP = left.stateP
    implicit val rsP = right.stateP
    type S = (left.S, right.S, (B, Code[Boolean]))
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = (left.emptyState, right.emptyState, (rNil, false))
    def length(s0: S): Option[Code[Int]] = left.length(s0._1)

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(
      k: Init[S] => Code[Ctrl]
    ): Code[Ctrl] = {
      val missing = jb.joinPoint()
      missing.define { _ => k(Missing) }
      left.init(mb, jb, param) {
        case Missing => missing(())
        case Start(lS) => right.init(mb, jb, param) {
          case Missing => missing(())
          case Start(rS) => k(Start((lS, rS, (rNil, false))))
        }
      }
    }

    def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(
      k: Step[(A, B), S] => Code[Ctrl]
    ): Code[Ctrl] = {
      val (lS0, rS0, (rPrev, somePrev)) = state
      left.step(mb, jb, lS0) {
        case EOS => k(EOS)
        case Yield(lElt, lS) =>
          val push = jb.joinPoint[(B, right.S, (B, Code[Boolean]))](mb)
          val pull = jb.joinPoint[right.S](mb)
          val compare = jb.joinPoint[(B, right.S)](mb)
          push.define { case (rElt, rS, rPrevOpt) =>
            k(Yield((lElt, rElt), (lS, rS, rPrevOpt)))
          }
          pull.define(right.step(mb, jb, _) {
            case EOS => push((rNil, right.emptyState, (rNil, false)))
            case Yield(rElt, rS) => compare((rElt, rS))
          })
          compare.define { case (rElt, rS) =>
            ParameterPack.let(mb, comp(lElt, rElt)) { c =>
              (c > 0).mux(
                pull(rS),
                (c < 0).mux(
                  push((rNil, rS, (rElt, true))),
                  push((rElt, rS, (rElt, true)))))
            }
          }
          somePrev.mux(
            compare((rPrev, rS0)),
            pull(rS0))
      }
    }
  }

  private[ir] def apply(
    emitter: Emit,
    streamIR0: IR,
    env0: Emit.E,
    rvas: Emit.RVAS,
    er: EmitRegion,
    container: Option[AggContainer]
  ): EmitStream = {
    val fb = emitter.mb.fb

    def present(v: Code[_]): EmitTriplet = EmitTriplet(Code._empty, false, v)

    def emitIR(ir: IR, env: Emit.E): EmitTriplet =
      emitter.emit(ir, env, rvas, er, container)

    def emitStream(streamIR: IR, env: Emit.E): Parameterized[Any, EmitTriplet] =
      streamIR match {
        case NA(_) =>
          missing

        case MakeStream(elements, t) =>
          val e = t.elementType.physicalType
          implicit val eP = TypedTriplet.pack(e)
          sequence(elements.map { ir => TypedTriplet(e, emitIR(ir, env)) })
            .map(_.untyped)

        case StreamRange(startIR, stopIR, stepIR) =>
          val step = fb.newField[Int]("sr_step")
          val start = fb.newField[Int]("sr_start")
          val stop = fb.newField[Int]("sr_stop")
          val llen = fb.newField[Long]("sr_llen")
          range(start, step)
            .map(present)
            .guardParam { (_, k) =>
              val startt = emitIR(startIR, env)
              val stopt = emitIR(stopIR, env)
              val stept = emitIR(stepIR, env)
              Code(startt.setup, stopt.setup, stept.setup,
                (startt.m || stopt.m || stept.m).mux(
                  k(None),
                  Code(
                    start := startt.value,
                    stop := stopt.value,
                    step := stept.value,
                    (step ceq 0).orEmpty(Code._fatal("Array range cannot have step size 0.")),
                    llen := (step < 0).mux(
                      (start <= stop).mux(0L, (start.toL - stop.toL - 1L) / (-step).toL + 1L),
                      (start >= stop).mux(0L, (stop.toL - start.toL - 1L) / step.toL + 1L)),
                    (llen > const(Int.MaxValue.toLong)).mux(
                      Code._fatal("Array range cannot have more than MAXINT elements."),
                      k(Some(llen.toI))))))
            }

        case ToStream(containerIR) =>
          val pType = containerIR.pType.asInstanceOf[PContainer]
          val eltPType = pType.elementType
          val region = er.region
          val aoff = fb.newField[Long]("a_off")
          range(0, 1)
            .map { i =>
              EmitTriplet(Code._empty,
                pType.isElementMissing(region, aoff, i),
                Region.loadIRIntermediate(eltPType)(pType.elementOffsetInRegion(region, aoff, i)))
            }
            .guardParam { (_, k) =>
              val arrt = emitIR(containerIR, env)
              val len = pType.loadLength(region, aoff)
              Code(arrt.setup,
                arrt.m.mux(
                  k(None),
                  Code(aoff := arrt.value, k(Some(len)))))
            }

        case Let(name, valueIR, childIR) =>
          val valueType = valueIR.pType
          val valueTI = coerce[Any](typeToTypeInfo(valueType))
          val vm = fb.newField[Boolean](name + "_missing")
          val vv = fb.newField(name)(valueTI)
          val valuet = emitIR(valueIR, env)
          val bodyEnv = env.bind(name -> ((valueTI, vm, vv)))
          emitStream(childIR, bodyEnv)
            .addSetup(_ => Code(
              valuet.setup,
              vm := valuet.m,
              vv := vm.mux(defaultValue(valueType), valuet.v)))

        case ArrayMap(childIR, name, bodyIR) =>
          val childEltType = childIR.pType.asInstanceOf[PStreamable].elementType
          val childEltTI = coerce[Any](typeToTypeInfo(childEltType))
          emitStream(childIR, env).map { eltt =>
            val eltm = fb.newField[Boolean](name + "_missing")
            val eltv = fb.newField(name)(childEltTI)
            val bodyt = emitIR(bodyIR, env.bind(name -> ((childEltTI, eltm, eltv))))
            EmitTriplet(
              Code(eltt.setup,
                eltm := eltt.m,
                eltv := eltm.mux(defaultValue(childEltType), eltt.v),
                bodyt.setup),
              bodyt.m,
              bodyt.v)
          }

        case ArrayFilter(childIR, name, condIR) =>
          val childEltType = childIR.pType.asInstanceOf[PStreamable].elementType
          val childEltTI = coerce[Any](typeToTypeInfo(childEltType))
          emitStream(childIR, env).filterMap { (eltt, k) =>
            val eltm = fb.newField[Boolean](name + "_missing")
            val eltv = fb.newField(name)(childEltTI)
            val condt = emitIR(condIR, env.bind(name -> ((childEltTI, eltm, eltv))))
            Code(
              eltt.setup,
              eltm := eltt.m,
              eltv := eltm.mux(defaultValue(childEltType), eltt.v),
              condt.setup,
              (condt.m || !condt.value[Boolean]).mux(
                k(None),
                k(Some(EmitTriplet(Code._empty, eltm, eltv)))))
          }

        case ArrayFlatMap(outerIR, name, innerIR) =>
          val outerEltType = outerIR.pType.asInstanceOf[PStreamable].elementType
          val outerEltTI = coerce[Any](typeToTypeInfo(outerEltType))
          val eltm = fb.newField[Boolean](name + "_missing")
          val eltv = fb.newField(name)(outerEltTI)
          val innerEnv = env.bind(name -> ((outerEltTI, eltm, eltv)))
          val outer = emitStream(outerIR, env)
          val inner = emitStream(innerIR, innerEnv)
            .addSetup[EmitTriplet] { eltt => Code(
              eltt.setup,
              eltm := eltt.m,
              eltv := eltm.mux(defaultValue(outerEltType), eltt.v))
            }
          compose(outer, inner)

        case ArrayLeftJoinDistinct(leftIR, rightIR, leftName, rightName, compIR, joinIR) =>
          val l = leftIR.pType.asInstanceOf[PStreamable].elementType
          val r = rightIR.pType.asInstanceOf[PStreamable].elementType
          implicit val lP = TypedTriplet.pack(l)
          implicit val rP = TypedTriplet.pack(r)
          val (leltm, leltv) = lP.newFields(fb, "join_lelt")
          val (reltm, reltv) = rP.newFields(fb, "join_relt")
          val env2 = env
            .bind(leftName -> ((typeToTypeInfo(l), leltm, leltv)))
            .bind(rightName -> ((typeToTypeInfo(r), reltm, reltv)))

          def compare(lelt: TypedTriplet[l.type], relt: TypedTriplet[r.type]): Code[Int] = {
            val compt = emitIR(compIR, env2)
            Code(
              lelt.storeTo(leltm, leltv),
              relt.storeTo(reltm, reltv),
              compt.setup,
              compt.m.orEmpty(Code._fatal("ArrayLeftJoinDistinct: comp can't be missing")),
              coerce[Int](compt.v))
          }

          leftJoinRightDistinct(
            emitStream(leftIR, env).map(TypedTriplet(l, _)),
            emitStream(rightIR, env).map(TypedTriplet(r, _)),
            TypedTriplet.missing(r),
            compare
          ).map { case (lelt, relt) =>
              val joint = emitIR(joinIR, env2)
              EmitTriplet(Code(
                lelt.storeTo(leltm, leltv),
                relt.storeTo(reltm, reltv),
                joint.setup), joint.m, joint.v) }

        case ArrayScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val e = childIR.pType.asInstanceOf[PStreamable].elementType
          val a = zeroIR.pType
          implicit val eP = TypedTriplet.pack(e)
          implicit val aP = TypedTriplet.pack(a)
          val (eltm, eltv) = eP.newFields(fb, "scan_elt")
          val (accm, accv) = aP.newFields(fb, "scan_acc")
          val zerot = emitIR(zeroIR, env)
          val bodyt = emitIR(bodyIR, env
            .bind(accName -> ((typeToTypeInfo(a), accm, accv)))
            .bind(eltName -> ((typeToTypeInfo(e), eltm, eltv))))
          emitStream(childIR, env).scan(TypedTriplet.missing(a))(
            TypedTriplet(a, zerot),
            (eltt, acc, k) => {
              val elt = TypedTriplet(e, eltt)
              Code(
                elt.storeTo(eltm, eltv),
                acc.storeTo(accm, accv),
                k(TypedTriplet(a, bodyt)))
            }).map(_.untyped)

        case ArrayAggScan(childIR, name, query) =>
          val res = genUID()
          val extracted =
            try {
              agg.Extract(CompileWithAggregators.liftScan(query), res)
            } catch {
              case e: agg.UnsupportedExtraction =>
                fatal(s"BUG: lowered aggscan to a stream, but this agg is not supported: $e")
            }

          val (newContainer, aggSetup, aggCleanup) =
            AggContainer.fromFunctionBuilder(extracted.aggs, fb, "array_agg_scan")
          val initIR = Optimize(extracted.init, noisy = true,
            context = "ArrayAggScan/StagedExtractAggregators/postAggIR", emitter.ctx)
          val seqPerEltIR = Optimize(extracted.seqPerElt, noisy = true,
            context = "ArrayAggScan/StagedExtractAggregators/init", emitter.ctx)
          val postAggIR = Optimize[IR](Let(res, extracted.results, extracted.postAggIR), noisy = true,
            context = "ArrayAggScan/StagedExtractAggregators/perElt", emitter.ctx)

          val e = coerce[PStreamable](childIR.pType).elementType
          val a = postAggIR.pType
          implicit val eP = TypedTriplet.pack(e)
          implicit val aP = TypedTriplet.pack(a)
          val (eltm, eltv) = eP.newFields(fb, "aggscan_elt")
          val (postm, postv) = aP.newFields(fb, "aggscan_new_elt")
          val bodyEnv = env.bind(name -> ((typeToTypeInfo(e), eltm, eltv)))
          val init = emitter.emit(initIR, env, None, er, Some(newContainer))
          val seqPerElt = emitter.emit(seqPerEltIR, bodyEnv, None, er, Some(newContainer))
          val postt = emitter.emit(postAggIR, bodyEnv, None, er, Some(newContainer))

          emitStream(childIR, env)
            .contMap[EmitTriplet] { (eltt, k) => Code(
              TypedTriplet(e, eltt).storeTo(eltm, eltv),
              TypedTriplet(a, postt).storeTo(postm, postv),
              seqPerElt.setup,
              k(EmitTriplet(Code._empty, postm, postv)))
            }
            .addSetup(
              _ => Code(aggSetup, init.setup),
              aggCleanup
            )

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }

    EmitStream(
      emitStream(streamIR0, env0),
      streamIR0.pType.asInstanceOf[PStreamable].elementType)
  }
}

case class EmitStream(
  stream: EmitStream.Parameterized[Any, EmitTriplet],
  elementType: PType
) {
  import EmitStream._
  private implicit val sP = stream.stateP

  def toArrayIterator(mb: MethodBuilder): ArrayIteratorTriplet = {
    val state = sP.newLocals(mb)

    ArrayIteratorTriplet(
      Code._empty,
      stream.length(state.load),
      (cont: (Code[Boolean], Code[_]) => Code[Unit]) => {
        val m = mb.newField[Boolean]("stream_missing")

        val setup =
          state := JoinPoint.CallCC[stream.S] { (jb, ret) =>
            stream.init(mb, jb, ()) {
              case Missing => Code(m := true, ret(stream.emptyState))
              case Start(s0) => Code(m := false, ret(s0))
            }
          }

        val addElements =
          JoinPoint.CallCC[Unit] { (jb, ret) =>
            val loop = jb.joinPoint()
            loop.define { _ => stream.step(mb, jb, state.load) {
              case EOS => ret(())
              case Yield(elt, s1) => Code(
                elt.setup,
                cont(elt.m, elt.value),
                state := s1,
                loop(()))
            } }
            loop(())
          }

        EmitArrayTriplet(setup, Some(m), addElements)
      })
  }
}
