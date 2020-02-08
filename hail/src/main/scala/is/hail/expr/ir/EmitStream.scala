package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.types.physical._
import is.hail.io.{AbstractTypedCodecSpec, InputBuffer}
import is.hail.utils._

import scala.language.existentials
import scala.reflect.ClassTag

object EmitStream {
  sealed trait Init[+S]
  object Missing extends Init[Nothing]
  case class Start[S](s0: S) extends Init[S]

  sealed trait Step[+A, +S]
  object EOS extends Step[Nothing, Nothing]
  case class Yield[A, S](elt: A, s: S) extends Step[A, S]

  def stepIf[A, S, X](k: Step[A, S] => Code[X], c: Code[Boolean], a: A, s: S): Code[X] =
    c.mux(k(Yield(a, s)), k(EOS))

  def zip[P](
    streams: IndexedSeq[Parameterized[P, EmitTriplet]],
    behavior: ArrayZipBehavior,
    f: (IndexedSeq[EmitTriplet], EmitTriplet => Code[Ctrl]) => Code[Ctrl]
  ): Parameterized[P, EmitTriplet] = new Parameterized[P, EmitTriplet] {
    type S = IndexedSeq[_]
    implicit val stateP: ParameterPack[S] = ParameterPack.array(streams.map(_.stateP))

    def emptyState: IndexedSeq[_] = streams.map(_.emptyState)

    override def length(s0: IndexedSeq[_]): Option[Code[Int]] = behavior match {
      case ArrayZipBehavior.AssertSameLength =>
        streams.zip(s0)
          .map { case (stream, state) => stream.length(state.asInstanceOf[stream.S]) }
          .reduce[Option[Code[Int]]] {
            case (Some(l1), Some(l2)) => Some((l1.cne(l2).mux(Code._fatal(const("zip: length mismatch: ")
              .concat(l1.toS).concat(", ").concat(l2.toS)), l1)))
            case _ => None
          }
      case ArrayZipBehavior.TakeMinLength =>
        streams.zip(s0)
          .map { case (stream, state) => stream.length(state.asInstanceOf[stream.S]) }
          .reduce[Option[Code[Int]]] {
            case (Some(l1), Some(l2)) => Some((l1 < l2).mux(l1, l2))
            case _ => None
          }
      case ArrayZipBehavior.ExtendNA =>
        streams.zip(s0)
          .map { case (stream, state) => stream.length(state.asInstanceOf[stream.S]) }
          .reduce[Option[Code[Int]]] {
            case (Some(l1), Some(l2)) => Some((l1 > l2).mux(l1, l2))
            case _ => None
          }
      case ArrayZipBehavior.AssumeSameLength =>
        streams.zip(s0)
          .flatMap { case (stream, state) => stream.length(state.asInstanceOf[stream.S]) }
          .headOption
    }

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(
      k: Init[S] => Code[Ctrl]
    ): Code[Ctrl] = {
      val missing = jb.joinPoint()
      missing.define { _ => k(Missing) }

      def loop(i: Int, ab: ArrayBuilder[Any]): Code[Ctrl] = {
        if (i == streams.length)
          k(Start(ab.result(): IndexedSeq[_]))
        else
          streams(i).init(mb, jb, param) {
            case Missing => missing(())
            case Start(s) =>
              ab += s
              loop(i + 1, ab)
          }
      }

      loop(0, new ArrayBuilder)
    }

    def step(mb: MethodBuilder, jb: JoinPointBuilder, state: IndexedSeq[_])(k: Step[EmitTriplet, S] => Code[Ctrl]): Code[Ctrl] = {
      val eos = jb.joinPoint()
      eos.define(_ => k(EOS))
      behavior match {
        case ArrayZipBehavior.AssertSameLength =>
          val anyEOS = mb.newLocal[Boolean]
          val allEOS = mb.newLocal[Boolean]
          val labels = (0 to streams.size).map(_ => jb.joinPoint())

          val ab = new ArrayBuilder[(EmitTriplet, Any)]
          labels.indices.foreach { i =>
            if (i == streams.size) {
              val abr = ab.result()
              val elts = abr.map(_._1)
              val ss = abr.map(_._2)
              labels(i).define(_ => anyEOS.mux(
                allEOS.mux(
                  eos(()),
                  Code._fatal("zip: length mismatch")),
                f(elts, { b => k(Yield(b, ss)) })))
            } else {
              val streamI = streams(i)
              labels(i).define(_ => streamI.step(mb, jb, state(i).asInstanceOf[streamI.S]) {
                case EOS =>
                  Code(anyEOS := true, labels(i + 1)(()))
                case Yield(elt, s) =>
                  ab += ((elt, s))
                  Code(allEOS := false, labels(i + 1)(()))
              })
            }
          }
          Code(anyEOS := false, allEOS := true, labels(0)(()))
        case ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength =>
          def loop(i: Int, ab: ArrayBuilder[(EmitTriplet, Any)]): Code[Ctrl] = {
            if (i == streams.length) {
              val abr = ab.result()
              val elts = abr.map(_._1)
              val ss = abr.map(_._2)
              f(elts, { b => k(Yield(b, ss)) })
            } else {
              val streamI = streams(i)
              streamI.step(mb, jb, state(i).asInstanceOf[streamI.S]) {
                case Yield(elt, s) =>
                  ab += ((elt, s))
                  loop(i + 1, ab)
                case EOS => eos(())
              }
            }
          }

          loop(0, new ArrayBuilder)

        case ArrayZipBehavior.ExtendNA =>
          val allEOS = mb.newLocal[Boolean]
          val missing = streams.map(_ => mb.newLocal[Boolean])
          val labels = (0 to streams.size).map(_ => jb.joinPoint())

          val ab = new ArrayBuilder[(Code[_], Any)]
          labels.indices.foreach { i =>
            if (i == streams.size) {
              val abr = ab.result()
              val elts = missing.zip(abr).map { case (m, (v, _)) => EmitTriplet(Code._empty, m, v) }
              val ss = abr.map(_._2)
              labels(i).define(_ => allEOS.mux(
                eos(()),
                f(elts, { b => k(Yield(b, ss)) })))
            } else {
              val streamI = streams(i)
              labels(i).define(_ => streamI.step(mb, jb, state(i).asInstanceOf[streamI.S]) {
                case EOS =>
                  Code(missing(i) := true, labels(i + 1)(()))
                case Yield(elt, s) =>
                  ab += ((elt.v, s))
                  Code(allEOS := false, elt.setup, missing(i) := elt.m, labels(i + 1)(()))
              })
            }
          }
          Code(allEOS := true, labels(0)(()))
      }
    }
  }

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

  def read[P](dec: Code[InputBuffer] => Code[Long]): Parameterized[Code[InputBuffer], Code[Long]] = new Parameterized[Code[InputBuffer], Code[Long]] {
    type S = Code[InputBuffer]
    val stateP: ParameterPack[S] = implicitly

    def emptyState: S = Code._null

    def length(s0: S): Option[Code[Int]] = None

    def init(
      mb: MethodBuilder,
      jb: JoinPointBuilder,
      buf: Code[InputBuffer]
    )(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      k(Start(buf))

    def step(
      mb: MethodBuilder,
      jb: JoinPointBuilder,
      state: S
    )(k: Step[Code[Long], S] => Code[Ctrl]): Code[Ctrl] = {
      stepIf(k, state.readByte().toZ, dec(state), state)
    }
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

  def fromIterator[A <: AnyRef : ClassTag]: Parameterized[Code[Iterator[A]], Code[A]] =
    new Parameterized[Code[Iterator[A]], Code[A]] {
      type S = Code[Iterator[A]]
      val stateP: ParameterPack[S] = implicitly
      def emptyState: S = Code.invokeScalaObject[Iterator[Nothing]](Iterator.getClass, "empty")
      def length(s0: S): Option[Code[Int]] = None

      def init(mb: MethodBuilder, jb: JoinPointBuilder, iter: S)(
        k: Init[S] => Code[Ctrl]
      ): Code[Ctrl] =
        k(Start(iter))

      def step(mb: MethodBuilder, jb: JoinPointBuilder, iter: S)(
        k: Step[Code[A], S] => Code[Ctrl]
      ): Code[Ctrl] =
        iter.hasNext.mux(
          ParameterPack.let(mb, iter.next()) { elt => k(Yield(elt, iter)) },
          k(EOS))
    }

  def mux[P, A: ParameterPack](
    cond: Code[Boolean],
    left: Parameterized[P, A],
    right: Parameterized[P, A]
  ): Parameterized[P, A] = new Parameterized[P, A] {
    implicit val lsP = left.stateP
    implicit val rsP = right.stateP
    type S = (Code[Boolean], left.S, right.S)
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = (true, left.emptyState, right.emptyState)

    def length(s0: S): Option[Code[Int]] =
      (left.length(s0._2) liftedZip right.length(s0._3))
        .map { case (lLen, rLen) => cond.mux(lLen, rLen) }

    def init(mb: MethodBuilder, jb: JoinPointBuilder, param: P)(k: Init[S] => Code[Ctrl]): Code[Ctrl] = {
      val missing = jb.joinPoint()
      val start = jb.joinPoint[S](mb)
      missing.define { _ => k(Missing) }
      start.define { s => k(Start(s)) }
      cond.mux(
        left.init(mb, jb, param) {
          case Start(s0) => start((true, s0, right.emptyState))
          case Missing => missing(())
        },
        right.init(mb, jb, param) {
          case Start(s0) => start((false, left.emptyState, s0))
          case Missing => missing(())
        })
    }

    def step(mb: MethodBuilder, jb: JoinPointBuilder, state: S)(k: Step[A, S] => Code[Ctrl]): Code[Ctrl] = {
      val (useLeft, lS, rS) = state
      val eos = jb.joinPoint()
      val push = jb.joinPoint[(A, left.S, right.S)](mb)
      eos.define { _ => k(EOS) }
      push.define { case (elt, lS, rS) => k(Yield(elt, (useLeft, lS, rS))) }
      useLeft.mux(
        left.step(mb, jb, lS) {
          case Yield(a, lS1) => push((a, lS1, rS))
          case EOS => eos(())
        },
        right.step(mb, jb, rS) {
          case Yield(a, rS1) => push((a, lS, rS1))
          case EOS => eos(())
        })
    }
  }

  def decode[T](region: Code[Region], spec: AbstractTypedCodecSpec)(
    dec: spec.StagedDecoderF[T]
  ): Parameterized[Code[InputBuffer], Code[T]] = new Parameterized[Code[InputBuffer], Code[T]] {
    type S = Code[InputBuffer]
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = Code._null
    def length(s0: S): Option[Code[Int]] = None

    def init(mb: MethodBuilder, jb: JoinPointBuilder, ib: Code[InputBuffer])(k: Init[S] => Code[Ctrl]): Code[Ctrl] =
      k(Start(ib))

    def step(mb: MethodBuilder, jb: JoinPointBuilder, ib: Code[InputBuffer])(
      k: Step[Code[T], S] => Code[Ctrl]
    ): Code[Ctrl] =
      (ib.isNull || !ib.readByte().toZ).mux(k(EOS), k(Yield(dec(region, ib), ib)))
  }

  private[ir] def apply(
    emitter: Emit,
    streamIR0: IR,
    env0: Emit.E,
    er: EmitRegion,
    container: Option[AggContainer]
  ): EmitStream = {
    val fb = emitter.mb.fb

    def present(v: Code[_]): EmitTriplet = EmitTriplet(Code._empty, false, v)

    def emitIR(ir: IR, env: Emit.E): EmitTriplet =
      emitter.emit(ir, env, er, container)

    def emitStream(streamIR: IR, env: Emit.E): Parameterized[Any, EmitTriplet] =
      streamIR match {
        case NA(_) =>
          missing

        case In(i, t@PStream(eltPType, _)) =>
          val EmitTriplet(_, m, v) = emitter.normalArgument(i, t)
          fromIterator[RegionValue]
            .map { (rv: Code[RegionValue]) =>
              present(Region.loadIRIntermediate(eltPType)(rv.invoke[Long]("getOffset")))
            }
            .guardParam { (_, k) =>
              m.mux(k(None), k(Some(coerce[Iterator[RegionValue]](v))))
            }

        case ReadPartition(path, spec, requestedType) =>
          val p = emitIR(path, env)
          val pathString = path.pType.asInstanceOf[PString].loadString(p.value[Long])

          val (_, dec) = spec.buildEmitDecoderF(requestedType, fb)

          read(dec(er.region, _)).map(present).guardParam { (_, k) =>
            val rowBuf = spec.buildCodeInputBuffer(fb.getUnsafeReader(pathString, true))
            Code(p.setup, p.m.mux(k(None), k(Some(rowBuf))))
          }

        case x@MakeStream(elements, t) =>
          val e = coerce[PStreamable](x.pType).elementType
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
                pType.isElementMissing(aoff, i),
                Region.loadIRIntermediate(eltPType)(pType.elementOffset(aoff, i)))
            }
            .guardParam { (_, k) =>
              val arrt = emitIR(containerIR, env)
              val len = pType.loadLength(aoff)
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

        case ArrayZip(as, names, body, behavior) =>
          val streams = as.map(emitStream(_, env))
          val childEltTypes = as.map(_.pType.asInstanceOf[PStreamable].elementType)

          EmitStream.zip[Any](streams, behavior, { (xs, k) =>
            val mv = names.zip(childEltTypes).map { case (name, t) =>
              val ti = typeToTypeInfo(t)
              val eltm = fb.newField[Boolean](name + "_missing")
              val eltv = fb.newField(name)(ti)
              (t, ti, eltm, eltv)
            }
            val bodyt = emitIR(body,
              env.bindIterable(names.zip(mv.map { case (_, ti, m, v) => (ti, m.load(), v.load()) })))
            k(EmitTriplet(
              Code(xs.zip(mv).foldLeft[Code[Unit]](Code._empty[Unit]) { case (acc, (et, (t, ti, m, v))) =>
                Code(acc,
                  et.setup,
                  m := et.m,
                  v.storeAny(m.mux(defaultValue(t), et.v)))
              },
                bodyt.setup),
              bodyt.m,
              bodyt.v)
            )
          })

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

        case x@RunAggScan(array, name, init, seqs, result, _) =>
          val aggs = x.physicalSignatures
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromFunctionBuilder(aggs, fb, "array_agg_scan")

          val producerElementPType = coerce[PStreamable](array.pType).elementType
          val resultPType = result.pType
          implicit val eP = TypedTriplet.pack(producerElementPType)
          implicit val aP = TypedTriplet.pack(resultPType)
          val (eltm, eltv) = eP.newFields(fb, "aggscan_elt")
          val (postm, postv) = aP.newFields(fb, "aggscan_new_elt")
          val bodyEnv = env.bind(name -> ((typeToTypeInfo(producerElementPType), eltm, eltv)))
          val cInit = emitter.emit(init, env, er, Some(newContainer))
          val seqPerElt = emitter.emit(seqs, bodyEnv, er, Some(newContainer))
          val postt = emitter.emit(result, bodyEnv, er, Some(newContainer))

          emitStream(array, env)
            .map[EmitTriplet] { eltt =>
              EmitTriplet(
                Code(
                  TypedTriplet(producerElementPType, eltt).storeTo(eltm, eltv),
                  TypedTriplet(resultPType, postt).storeTo(postm, postv),
                  seqPerElt.setup),
                postm,
                postv)
            }.addSetup(
            _ => Code(aggSetup, cInit.setup),
            aggCleanup
          )

        case If(condIR, thn, els) =>
          val t = thn.pType.asInstanceOf[PStreamable].elementType
          implicit val tP = TypedTriplet.pack(t)
          val cond = fb.newField[Boolean]
          mux(cond,
            emitStream(thn, env).map(TypedTriplet(t, _)),
            emitStream(els, env).map(TypedTriplet(t, _))
          ).map(_.untyped)
            .guardParam { (param, k) =>
              val condt = emitIR(condIR, env)
              Code(condt.setup, condt.m.mux(
                k(None),
                Code(cond := condt.value, k(Some(param)))))
            }

        case ReadPartition(pathIR, spec, rowType) =>
          val (returnedRowPType, rowDec) = spec.buildEmitDecoderF[Long](rowType, fb)
          decode(er.region, spec)(rowDec)
            .map(present)
            .guardParam { (_, k) =>
              val patht = emitIR(pathIR, env)
              val pathString = Code.invokeScalaObject[Region, Long, String](
                PString.getClass, "loadString", er.region, patht.value)
              Code(patht.setup, patht.m.mux(k(None),
                k(Some(spec.buildCodeInputBuffer(fb.getUnsafeReader(pathString, true))))))
            }

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
