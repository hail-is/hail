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
  case class Skip[S](s: S) extends Step[Nothing, S]
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

    def map[B](f: A => B): Parameterized[P, B] =
      contMap[B] { (a, k) => k(f(a)) }

    def contMap[B](
      f: (A, B => Code[Ctrl]) => Code[Ctrl],
      setup: Code[Unit] = Code._empty,
      cleanup: Code[Unit] = Code._empty
    ): Parameterized[P, B] = new Parameterized[P, B] {
      type S = self.S
      val stateP = self.stateP
      def emptyState = self.emptyState
      def length(s0: S): Option[Code[Int]] = self.length(s0)
      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
        self.init(mb, jb, param) {
          case Missing => k(Missing)
          case Start(s) => Code(setup, k(Start(s)))
        }
      def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[B, S] => Code[Ctrl]): Code[Ctrl] =
        self.step(mb, jb, state) {
          case EOS => Code(cleanup, k(EOS))
          case Skip(s) => k(Skip(s))
          case Yield(a, s) => f(a, b => k(Yield(b, s)))
        }
    }

    def filterMap[B](f: (A, Option[B] => Code[Ctrl]) => Code[Ctrl]): Parameterized[P, B] = new Parameterized[P, B] {
      type S = self.S
      val stateP = self.stateP
      def emptyState = self.emptyState
      def length(s0: S): Option[Code[Int]] = None
      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
        self.init(mb, jb, param)(k)
      def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[B, S] => Code[Ctrl]): Code[Ctrl] =
        self.step(mb, jb, state) {
          case EOS => k(EOS)
          case Skip(s) => k(Skip(s))
          case Yield(a, s) => f(a, {
            case None => k(Skip(s))
            case Some(b) => k(Yield(b, s))
          })
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
            case Skip(s1) => k(Skip((s1, b, false)))
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

  def range[P](
    // initialize(param, k) = ...k(None) or k(Some((len, start)))...
    initialize: (P, Option[(Code[Int], Code[Int])] => Code[Ctrl]) => Code[Ctrl],
    incr: Code[Int] => Code[Int]
  ): Parameterized[P, Code[Int]] = new Parameterized[P, Code[Int]] {
    type S = (Code[Int], Code[Int])
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = (0, 0)
    def length(s0: S): Option[Code[Int]] = Some(s0._1)

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      initialize(param, {
        case None => k(Missing)
        case Some(s0) => k(Start(s0))
      })

    def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[Code[Int], S] => Code[Ctrl]): Code[Ctrl] = {
      val (pos, idx) = state
      stepIf(k, pos > 0, idx, (pos - 1, incr(idx)))
    }
  }

  def sequence[P, A](
    initialize: P => Code[Unit],
    elements: Seq[A]
  ): Parameterized[P, A] = new Parameterized[P, A] {
    type S = Code[Int]
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = elements.length
    def length(s0: S): Option[Code[Int]] = Some(const(elements.length) - s0)

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      Code(initialize(param), k(Start(0)))

    def step(mb: MethodBuilder, jb: JoinPointBuilder, idx: S)(k: Step[A, S] => Code[Ctrl]): Code[Ctrl] = {
      val eos = jb.joinPoint()
      eos.define { _ => k(EOS) }
      JoinPoint.switch(idx, eos, elements.map { elt =>
        val j = jb.joinPoint()
        j.define { _ => k(Yield(elt, idx + 1)) }
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
      val (outS, innS) = state
      inner.step(mb, jb, innS) {
        case EOS => outer.step(mb, jb, outS) {
          case EOS => k(EOS)
          case Skip(outS1) => k(Skip((outS1, inner.emptyState)))
          case Yield(outElt, outS1) => inner.init(mb, jb, outElt) {
            case Missing => k(Skip((outS1, inner.emptyState)))
            case Start(innS1) => k(Skip((outS1, innS1)))
          }
        }
        case Skip(innS1) => k(Skip((outS, innS1)))
        case Yield(innElt, innS1) => k(Yield(innElt, (outS, innS1)))
      }
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
    def emptyState = (left.emptyState, right.emptyState, (rNil, false))
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
        case Skip(lS) => k(Skip((lS, rS0, (rPrev, somePrev))))
        case Yield(lElt, lS) =>
          val push = jb.joinPoint[(B, right.S, (B, Code[Boolean]))](mb)
          val pull = jb.joinPoint[right.S](mb)
          val compare = jb.joinPoint[(B, right.S)](mb)
          push.define { case (rElt, rS, rPrevOpt) =>
            k(Yield((lElt, rElt), (lS, rS, rPrevOpt)))
          }
          pull.define(right.step(mb, jb, _) {
            case EOS => push((rNil, right.emptyState, (rNil, false)))
            case Skip(rS) => pull(rS)
            case Yield(rElt, rS) => compare((rElt, rS))
          })
          compare.define { case (rElt, rS) =>
            ParameterPack.localMemoize(mb, comp(lElt, rElt)) { c =>
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

    def emitIR(ir: IR, env: Emit.E): EmitTriplet =
      emitter.emit(ir, env, rvas, er, container)

    def emitStream(streamIR: IR, env: Emit.E): Parameterized[Any, EmitTriplet] =
      emitPStream[Any](streamIR, env, _ => Code._empty)

    def emitPStream[E](streamIR: IR, env: Emit.E, setupEnv: E => Code[Unit]): Parameterized[E, EmitTriplet] =
      streamIR match {

        case NA(_) =>
          missing

        case MakeStream(elements, pType) =>
          sequence(setupEnv, elements.map(emitIR(_, env)))

        case StreamRange(startIR, stopIR, stepIR) =>
          val step = fb.newField[Int]("sr_step")
          val start = fb.newField[Int]("sr_start")
          val stop = fb.newField[Int]("sr_stop")
          val llen = fb.newField[Long]("sr_llen")
          range[E](
            (e, k) => {
              val startt = emitIR(startIR, env)
              val stopt = emitIR(stopIR, env)
              val stept = emitIR(stepIR, env)
              Code(setupEnv(e), startt.setup, stopt.setup, stept.setup,
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
                      k(Some((llen.toI, start)))))))
            },
            i => i + step
          )
            .map { i => EmitTriplet(Code._empty, false, i) }

        case ToStream(containerIR) =>
          val pType = containerIR.pType.asInstanceOf[PContainer]
          val eltPType = pType.elementType
          val region = er.region
          val aoff = fb.newField[Long]("a_off")
          val len = pType.loadLength(region, aoff)
          range[E](
            (e, k) => {
              val arrt = emitIR(containerIR, env)
              Code(setupEnv(e),
                arrt.setup,
                arrt.m.mux(k(None), Code(aoff := arrt.value, k(Some((len, 0))))))
            },
            i => i + 1
          )
            .map { i =>
              EmitTriplet(Code._empty,
                pType.isElementMissing(region, aoff, i),
                Region.loadIRIntermediate(eltPType)(pType.elementOffsetInRegion(region, aoff, i)))
            }

        case Let(name, valueIR, childIR) =>
          val valueType = valueIR.pType
          val valueTI = coerce[Any](typeToTypeInfo(valueType))
          val vm = fb.newField[Boolean](name + "_missing")
          val vv = fb.newField(name)(valueTI)
          val valuet = emitIR(valueIR, env)
          val bodyEnv = env.bind(name -> ((valueTI, vm, vv)))
          def setupBodyEnv(e: E): Code[Unit] =
            Code(setupEnv(e),
              valuet.setup,
              vm := valuet.m,
              vv := vm.mux(defaultValue(valueType), valuet.v))
          emitPStream[E](childIR, bodyEnv, setupBodyEnv)

        case ArrayMap(childIR, name, bodyIR) =>
          val childEltType = childIR.pType.asInstanceOf[PStreamable].elementType
          val childEltTI = coerce[Any](typeToTypeInfo(childEltType))
          emitPStream(childIR, env, setupEnv).map { eltt =>
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
          emitPStream(childIR, env, setupEnv).filterMap { (eltt, k) =>
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
          def setupInnerEnv(eltt: EmitTriplet): Code[Unit] =
            Code(eltt.setup,
              eltm := eltt.m,
              eltv := eltm.mux(defaultValue(outerEltType), eltt.v))
          val outer = emitPStream(outerIR, env, setupEnv)
          val inner = emitPStream(innerIR, innerEnv, setupInnerEnv)
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
          val compt = emitIR(compIR, env2)
          val joint = emitIR(joinIR, env2)
          leftJoinRightDistinct[E, TypedTriplet[l.type], TypedTriplet[r.type]](
            emitPStream(leftIR, env, setupEnv).map(TypedTriplet(l, _)),
            emitStream(rightIR, env).map(TypedTriplet(r, _)),
            TypedTriplet.missing(r),
            (lelt, relt) => Code(
              lelt.storeTo(leltm, leltv),
              relt.storeTo(reltm, reltv),
              compt.setup,
              compt.m.orEmpty(Code._fatal("ArrayLeftJoinDistinct: comp can't be missing")),
              coerce[Int](compt.v))
          )
            .map { case (lelt, relt) =>
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
          emitPStream(childIR, env, setupEnv).scan(TypedTriplet.missing(a))(
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
          val initIR = Optimize(extracted.init, noisy = true, canGenerateLiterals = true,
            context = Some("ArrayAggScan/StagedExtractAggregators/postAggIR"))
          val seqPerEltIR = Optimize(extracted.seqPerElt, noisy = true, canGenerateLiterals = false,
            context = Some("ArrayAggScan/StagedExtractAggregators/init"))
          val postAggIR = Optimize(Let(res, extracted.results, extracted.postAggIR), noisy = true, canGenerateLiterals = false,
            context = Some("ArrayAggScan/StagedExtractAggregators/perElt"))

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

          emitPStream(childIR, env, setupEnv).contMap(
            (eltt, k) => Code(
              TypedTriplet(e, eltt).storeTo(eltm, eltv),
              TypedTriplet(a, postt).storeTo(postm, postv),
              seqPerElt.setup,
              k(EmitTriplet(Code._empty, postm, postv))),
            Code(aggSetup, init.setup),
            aggCleanup)

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }

    EmitStream(
      emitStream(streamIR0, env0),
      streamIR0.pType.asInstanceOf[PStreamable].elementType)
  }

  def apply(fb: EmitFunctionBuilder[_], ir: IR): EmitStream =
    apply(new Emit(fb.apply_method, 1), ir, Env.empty,
      None, EmitRegion.default(fb.apply_method), None)
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
              case Skip(s1) => Code(state := s1, loop(()))
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
