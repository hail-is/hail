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
  object Empty extends Init[Nothing]
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
    def dummyState: S

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
      def dummyState = self.dummyState
      def length(s0: S): Option[Code[Int]] = self.length(s0)
      def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
        self.init(mb, jb, param) {
          case m@(Missing | Empty) => k(m)
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
      def dummyState = self.dummyState
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
  }

  val missing: Parameterized[Any, Nothing] = new Parameterized[Any, Nothing] {
    type S = Unit
    val stateP: ParameterPack[S] = implicitly
    def dummyState: S = ()
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
    def dummyState: S = (0, 0)
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
        // status == 2 iff the inner stream is missing
        // status == 1 iff the inner stream is actually empty (so the state is invalid)
        // status == 0 otherwise
        val status = mb.newField[Int]("stream_status")

        val setup =
          state := JoinPoint.CallCC[stream.S] { (jb, ret) =>
            stream.init(mb, jb, ()) {
              case Missing => Code(status := 2, ret(stream.dummyState))
              case Empty => Code(status := 1, ret(stream.dummyState))
              case Start(s0) => Code(status := 0, ret(s0))
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
            JoinPoint.mux((status ceq 0), loop, ret)
          }

        EmitArrayTriplet(setup, Some(status ceq 2), addElements)
      })
  }
}
