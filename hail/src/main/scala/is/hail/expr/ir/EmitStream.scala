 package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValue, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.types.physical._
import is.hail.io.{AbstractTypedCodecSpec, InputBuffer}
import is.hail.utils._

import scala.language.{existentials, higherKinds}
import scala.reflect.ClassTag

case class EmitStreamContext(mb: EmitMethodBuilder[_])

abstract class COption[+A] { self =>
  def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl]

  def cases(mb: EmitMethodBuilder[_])(none: Code[Unit], some: A => Code[Unit]): Code[Unit] = {
    implicit val ctx = EmitStreamContext(mb)
    val L = CodeLabel()
    Code(
      self(Code(none, L.goto), (a) => Code(some(a), L.goto)),
      L)
  }

  def map[B](f: A => B): COption[B] = new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      self.apply(none, a => some(f(a)))
  }

  def mapCPS[B](f: (A, B => Code[Ctrl]) => Code[Ctrl]): COption[B] =  new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      self.apply(none, a => f(a, some))
  }

  def addSetup(f: Code[Unit]): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      Code(f, self.apply(none, some))
  }

  def doIfNone(f: Code[Unit]): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      self.apply(Code(f, none), some)
  }

  def flatMap[B](f: A => COption[B]): COption[B] = new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      self(Code(L, none), f(_).apply(L.goto, some))
    }
  }

  def filter(cond: Code[Boolean]): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      self(Code(L, none), (a) => cond.mux(L.goto, some(a)))
    }
  }

  def flatMapCPS[B](f: (A, EmitStreamContext, COption[B] => Code[Ctrl]) => Code[Ctrl]): COption[B] = new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      self(Code(L, none), f(_, ctx, (b) => b(L.goto, some)))
    }
  }
}

object COption {
  def apply[A](missing: Code[Boolean], value: A): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      missing.mux(none, some(value))
  }

  object None extends COption[Nothing] {
    def apply(none: Code[Ctrl], some: Nothing => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      none
  }

  def present[A](value: A): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      some(value)
  }

  def lift[A](opts: IndexedSeq[COption[A]]): COption[IndexedSeq[A]] = new COption[IndexedSeq[A]] {
    def apply(none: Code[Ctrl], some: IndexedSeq[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      def nthOpt(i: Int, acc: IndexedSeq[A]): Code[Ctrl] =
        if (i == 0)
          opts(i)(Code(L, none), a => nthOpt(i+1, acc :+ a))
        else if (i == opts.length - 1)
          opts(i)(L.goto, a => some(acc :+ a))
        else
          opts(i)(L.goto, a => nthOpt(i+1, acc :+ a))

      nthOpt(0, FastIndexedSeq())
    }
  }

  // Returns a COption value equivalent to 'left' when 'useLeft' is true,
  // otherwise returns a value equivalent to 'right'. In the case where neither
  // 'left' nor 'right' are missing, uses 'fuse' to combine the values.
  // Presumably 'fuse' dynamically chooses one or the other based on the same
  // boolean passed in 'useLeft. 'fuse' is needed because we don't require
  // a temporary.
  def choose[A](useLeft: Code[Boolean], left: COption[A], right: COption[A], fuse: (A, A) => A): COption[A] =
    (left, right) match {
      case (COption.None, COption.None) => COption.None
      case (_, COption.None) =>
        left.filter(!useLeft)
      case (COption.None, _) =>
        right.filter(useLeft)
      case _ => new COption[A] {
        var l: Option[A] = scala.None
        var r: Option[A] = scala.None
        def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
          val L = CodeLabel()
          val M = CodeLabel()
          val runLeft = left(Code(L, none), a => { l = Some(a); M.goto })
          val runRight = right(L.goto, a => { r = Some(a); M.goto })
          Code(
            useLeft.mux(runLeft, runRight),
            M, some(fuse(l.get, r.get)))
        }
      }
    }

  def fromEmitCode(et: EmitCode): COption[PCode] = new COption[PCode] {
    def apply(none: Code[Ctrl], some: PCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      Code(et.setup, et.m.mux(none, some(et.pv)))
    }
  }

  def toEmitCode(opt: COption[PCode], t: PType, mb: EmitMethodBuilder[_]): EmitCode = {
    implicit val ctx = EmitStreamContext(mb)
    val m = mb.newLocal[Boolean]()
    val v = mb.newPLocal(t)
    val L = CodeLabel()
    EmitCode(
      Code(
        opt(Code(m := true, L.goto),
          a => Code(m := false, v := a, L.goto)),
        L),
      m, v.load())
  }
}

object CodeStream { self =>
  case class Source[+A](setup0: Code[Unit], close0: Code[Unit], setup: Code[Unit], close: Code[Unit], pull: Code[Ctrl])

  abstract class Stream[+A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A]

    def fold(mb: EmitMethodBuilder[_], init: => Code[Unit], f: (A) => Code[Unit], ret: => Code[Ctrl]): Code[Ctrl] = {
      implicit val ctx = EmitStreamContext(mb)
      val Ltop = CodeLabel()
      val Lafter = CodeLabel()
      val s = apply(Lafter.goto, (a) => Code(f(a), Ltop.goto: Code[Ctrl]))
      Code(
        init,
        s.setup0,
        s.setup,
        Ltop,
        s.pull,
        Lafter,
        s.close,
        s.close0,
        ret)
    }

    def forEach(mb: EmitMethodBuilder[_])(f: A => Code[Unit]): Code[Unit] =
      CodeStream.forEach(mb, this, f)

    def mapCPS[B](
      f: (EmitStreamContext, A, B => Code[Ctrl]) => Code[Ctrl],
      setup0: Option[Code[Unit]] = None,
      setup: Option[Code[Unit]] = None,
      close0: Option[Code[Unit]] = None,
      close: Option[Code[Unit]] = None
    ): Stream[B] = CodeStream.mapCPS(this)(f, setup0, setup, close0, close)

    def map[B](
      f: A => B,
      setup0: Option[Code[Unit]] = None,
      setup: Option[Code[Unit]] = None,
      close0: Option[Code[Unit]] = None,
      close: Option[Code[Unit]] = None
    ): Stream[B] = CodeStream.map(this)(f, setup0, setup, close0, close)

    def flatMap[B](f: A => Stream[B]): Stream[B] =
      CodeStream.flatMap(map(f))
  }

  def range(mb: EmitMethodBuilder[_], start: Code[Int], step: Code[Int], len: Code[Int]): Stream[Code[Int]] = {
    val lstep = mb.newLocal[Int]("sr_lstep")
    val cur = mb.newLocal[Int]("sr_cur")
    val t = mb.newLocal[Int]("sr_t")
    val rem = mb.newLocal[Int]("sr_rem")

    unfold[Code[Int]](
      init0 = Code._empty,
      init = Code(lstep := step, cur := start, rem := len),
      f = {
        case (_ctx, k) =>
          implicit val ctx = _ctx
          k(COption(rem <= 0,
            Code(
              t := cur,
              rem := rem - 1,
              cur := cur + lstep,
              t)))
      })
  }

  def unfold[A](
    init0: Code[Unit],
    init: Code[Unit],
    f: (EmitStreamContext, COption[A] => Code[Ctrl]) => Code[Ctrl]
  ): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      Source[A](
        setup0 = init0,
        close0 = Code._empty,
        setup = init,
        close = Code._empty,
        pull = f(ctx, _.apply(
          none = eos,
          some = a => push(a))))
    }
  }

  def forEachCPS[A](mb: EmitMethodBuilder[_], stream: Stream[A], f: (A, Code[Ctrl]) => Code[Ctrl]): Code[Unit] =
    run(mb, stream.mapCPS[Unit]((_, a, k) => f(a, k(()))))

  def forEach[A](mb: EmitMethodBuilder[_], stream: Stream[A], f: A => Code[Unit]): Code[Unit] =
    run(mb, stream.mapCPS((_, a, k) => Code(f(a), k(()))))

  def run(mb: EmitMethodBuilder[_], stream: Stream[Unit]): Code[Unit] = {
    implicit val ctx = EmitStreamContext(mb)
    val Leos = CodeLabel()
    val Lpull = CodeLabel()
    val source = stream(eos = Leos.goto, push = _ => Lpull.goto)
    Code(
      source.setup0,
      source.setup,
      // fall through
      Lpull, source.pull,
      Leos, source.close, source.close0
      // fall off
    )
  }

  def mapCPS[A, B](stream: Stream[A])(
    f: (EmitStreamContext, A, B => Code[Ctrl]) => Code[Ctrl],
    setup0: Option[Code[Unit]] = None,
    setup:  Option[Code[Unit]] = None,
    close0: Option[Code[Unit]] = None,
    close:  Option[Code[Unit]] = None
  ): Stream[B] = new Stream[B] {
    def apply(eos: Code[Ctrl], push: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[B] = {
      val source = stream(
        eos = eos,
        push = f(ctx, _, b => push(b)))
      Source[B](
        setup0 = setup0.map(Code(_, source.setup0)).getOrElse(source.setup0),
        close0 = close0.map(Code(_, source.close0)).getOrElse(source.close0),
        setup = setup.map(Code(_, source.setup)).getOrElse(source.setup),
        close = close.map(Code(_, source.close)).getOrElse(source.close),
        pull = source.pull)
    }
  }

  def map[A, B](stream: Stream[A])(
    f: A => B,
    setup0: Option[Code[Unit]] = None,
    setup:  Option[Code[Unit]] = None,
    close0: Option[Code[Unit]] = None,
    close:  Option[Code[Unit]] = None
  ): Stream[B] = mapCPS(stream)((_, a, k) => k(f(a)), setup0, setup, close0, close)

  def flatMap[A](outer: Stream[Stream[A]]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val closing = ctx.mb.newLocal[Boolean]("sfm_closing")
      val LouterPull = CodeLabel()
      var innerSource: Source[A] = null
      val LinnerPull = CodeLabel()
      val LinnerEos = CodeLabel()
      val LcloseOuter = CodeLabel()
      val inInnerStream = ctx.mb.newLocal[Boolean]("sfm_in_innner")
      val outerSource = outer(
        eos = eos,
        push = inner => {
          innerSource = inner(
            eos = LinnerEos.goto,
            push = push)
          Code(FastIndexedSeq[Code[Unit]](
            innerSource.setup, inInnerStream := true, LinnerPull, innerSource.pull,
              // for layout
              LinnerEos, innerSource.close, inInnerStream := false, closing.mux(LcloseOuter.goto, LouterPull.goto)))
        })
      Source[A](
        setup0 = Code(outerSource.setup0, innerSource.setup0),
        close0 = Code(innerSource.close0, outerSource.close0),
        setup = Code(closing := false, inInnerStream := false, outerSource.setup),
        close = Code(inInnerStream.mux(Code(closing := true, LinnerEos.goto), Code._empty), LcloseOuter, outerSource.close),
        pull = inInnerStream.mux(LinnerPull.goto, Code(LouterPull, outerSource.pull)))
    }
  }

  def filter[A](stream: Stream[COption[A]]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val Lpull = CodeLabel()
      val source = stream(
        eos = eos,
        push = _.apply(none = Lpull.goto, some = push))
      source.copy(pull = Code(Lpull, source.pull))
    }
  }

  def take[A](stream: Stream[COption[A]]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val Leos = CodeLabel()
      stream(
        eos = Code(Leos, eos),
        push = _.apply(none = Leos.goto, some = push)).asInstanceOf[Source[A]]
    }
  }

  def zip[A, B](left: Stream[A], right: Stream[B]): Stream[(A, B)] = new Stream[(A, B)] {
    def apply(eos: Code[Ctrl], push: ((A, B)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(A, B)] = {
      val Leos = CodeLabel()
      var rightSource: Source[B] = null
      val leftSource = left(
        eos = Code(Leos, eos),
        push = a => {
          rightSource = right(
            eos = Leos.goto,
            push = b => push((a, b)))
          rightSource.pull
        })

      Source[(A, B)](
        setup0 = Code(leftSource.setup0, rightSource.setup0),
        close0 = Code(leftSource.close0, rightSource.close0),
        setup = Code(leftSource.setup, rightSource.setup),
        close = Code(leftSource.close, rightSource.close),
        pull = leftSource.pull)
    }
  }

  def multiZip[A](streams: IndexedSeq[Stream[A]]): Stream[IndexedSeq[A]] = new Stream[IndexedSeq[A]] {
    def apply(eos: Code[Ctrl], push: IndexedSeq[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[IndexedSeq[A]] = {
      val Leos = CodeLabel()

      def nthSource(n: Int, acc: IndexedSeq[A]): Source[A] = {
        if (n == streams.length - 1) {
          streams(n)(Code(Leos, eos), c => push(acc :+ c))
        } else {
          var rest: Source[A] = null
          val src = streams(n)(
            Leos.goto,
            c => {
              rest = nthSource(n + 1, acc :+ c)
              rest.pull
            })
          Source[A](
            setup0 = Code(src.setup0, rest.setup0),
            close0 = Code(rest.close0, src.close0),
            setup = Code(src.setup, rest.setup),
            close = Code(rest.close, src.close),
            pull = src.pull)
        }
      }

      nthSource(0, IndexedSeq.empty).asInstanceOf[Source[IndexedSeq[A]]]
    }
  }

  def leftJoinRightDistinct(
    mb: EmitMethodBuilder[_],
    lElemType: PType, left: Stream[EmitCode],
    rElemType: PType, right: Stream[EmitCode],
    comp: (EmitCode, EmitCode) => Code[Int]
  ): Stream[(EmitCode, EmitCode)] = new Stream[(EmitCode, EmitCode)] {
    def apply(eos: Code[Ctrl], push: ((EmitCode, EmitCode)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(EmitCode, EmitCode)] = {
      val pulledRight = mb.newLocal[Boolean]()
      val rightEOS = mb.newLocal[Boolean]()
      val lx = mb.newEmitLocal(lElemType) // last value received from left
      val rx = mb.newEmitLocal(rElemType) // last value received from right
      val rxOut = mb.newEmitLocal(rElemType) // B value to push (may be rNil while xB is not)

      var rightSource: Source[EmitCode] = null
      val leftSource = left(
        eos = eos,
        push = a => {
          val Lpush = CodeLabel()
          val LpullRight = CodeLabel()
          val Lcompare = CodeLabel()

          val compareCode = Code(Lcompare, {
            val c = mb.newLocal[Int]()
            Code(
              c := comp(lx, rx),
              (c > 0).mux(
                LpullRight.goto,
                (c < 0).mux(
                  Code(rxOut := EmitCode.missing(rElemType), Lpush.goto),
                  Code(rxOut := rx, Lpush.goto))))
          })

          rightSource = right(
            eos = Code(rightEOS := true, rxOut := EmitCode.missing(rElemType), Lpush.goto),
            push = b => Code(rx := b, Lcompare.goto))

          Code(
            lx := a,
            pulledRight.mux[Unit](
              rightEOS.mux[Ctrl](Code(Lpush, push((lx, rxOut))), compareCode),
              Code(pulledRight := true, Code(LpullRight, rightSource.pull))))
        })

      Source[(EmitCode, EmitCode)](
        setup0 = Code(leftSource.setup0, rightSource.setup0),
        close0 = Code(leftSource.close0, rightSource.close0),
        setup = Code(pulledRight := false, rightEOS := false, leftSource.setup, rightSource.setup),
        close = Code(leftSource.close, rightSource.close),
        pull = leftSource.pull)
    }
  }
}

object EmitStream {

  import CodeStream._

  def write(mb: EmitMethodBuilder[_], sstream: SizedStream, ab: StagedArrayBuilder): Code[Unit] = {
    val SizedStream(stream, optLen) = sstream
    Code(
      ab.clear,
      optLen match {
        case Some((setupLen, len)) => Code(setupLen, ab.ensureCapacity(len))
        case None => ab.ensureCapacity(16)
      },
      stream.forEach(mb) { et =>
        Code(et.setup, et.m.mux(ab.addMissing(), ab.add(et.v)))
      })
  }

  def toArray(mb: EmitMethodBuilder[_], aTyp: PArray, optStream: COption[SizedStream]): EmitCode = {
    val srvb = new StagedRegionValueBuilder(mb, aTyp)
    val result = optStream.map { ss =>
      ss.length match {
        case None =>
          val xLen = mb.newLocal[Int]("sta_len")
          val i = mb.newLocal[Int]("sta_i")
          val vab = new StagedArrayBuilder(aTyp.elementType, mb, 0)
          PCode(aTyp, Code(
            write(mb, ss, vab),
            xLen := vab.size,
            srvb.start(xLen),
            i := const(0),
            Code.whileLoop(i < xLen,
              vab.isMissing(i).mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(aTyp.elementType)(vab(i))),
              i := i + 1,
              srvb.advance()),
            srvb.offset))

        case Some((setupLen, len)) =>
          PCode(aTyp, Code(
            setupLen,
            srvb.start(len),
            ss.stream.forEach(mb) { et =>
              Code(
                et.setup,
                et.m.mux(srvb.setMissing(), srvb.addIRIntermediate(aTyp.elementType)(et.v)),
                srvb.advance())
            },
            srvb.offset))
      }
    }

    COption.toEmitCode(result, aTyp, mb)
  }

  def sequence(mb: EmitMethodBuilder[_], elemPType: PType, elements: IndexedSeq[EmitCode]): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val i = mb.newLocal[Int]()
      val t = mb.newEmitLocal("ss_t", elemPType)
      val Leos = CodeLabel()
      val Lpush = CodeLabel()

      Source[EmitCode](
        setup0 = Code._empty,
        close0 = Code._empty,
        setup = i := const(0),
        close = Code._empty,
        pull = (i.get < elements.length).mux(
          Code(
            Code.switch(i, Leos.goto, elements.map(elem => Code(t := elem, Lpush.goto))),
            Lpush,
            i += 1,
            push(t)),
          Code(Leos, eos)))
    }
  }

  // length is required to be a variable reference
  case class SizedStream(stream: Stream[EmitCode], length: Option[(Code[Unit], Settable[Int])])

  def mux(mb: EmitMethodBuilder[_], eltType: PType, cond: Code[Boolean], left: Stream[EmitCode], right: Stream[EmitCode]): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val b = mb.newLocal[Boolean]()
      val Leos = CodeLabel()
      val elt = mb.newEmitLocal("stream_mux_elt", eltType)
      val Lpush = CodeLabel()

      val l = left(Code(Leos, eos), (a) => Code(elt := a, Lpush, push(elt)))
      val r = right(Leos.goto, (a) => Code(elt := a, Lpush.goto))

      Source[EmitCode](
        setup0 = Code(l.setup0, r.setup0),
        close0 = Code(l.close0, r.close0),
        setup = Code(b := cond, b.get.mux(l.setup, r.setup)),
        close = b.get.mux(l.close, r.close),
        pull = b.get.mux(l.pull, r.pull))
    }
  }

  def extendNA(mb: EmitMethodBuilder[_], eltType: PType, stream: Stream[EmitCode]): Stream[COption[EmitCode]] = new Stream[COption[EmitCode]] {
    def apply(eos: Code[Ctrl], push: COption[EmitCode] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[COption[EmitCode]] = {
      val atEnd = mb.newLocal[Boolean]()
      val x = mb.newEmitLocal(eltType)
      val Lpush = CodeLabel()
      val source = stream(Code(atEnd := true, Lpush.goto), a => Code(x := a, Lpush, push(COption(atEnd.get, x.get))))
      Source[COption[EmitCode]](
        setup0 = source.setup0,
        close0 = source.close0,
        setup = Code(atEnd := false, source.setup),
        close = source.close,
        pull = atEnd.get.mux(Lpush.goto, source.pull))
    }
  }

  private[ir] def apply[C](
    emitter: Emit[C],
    mb: EmitMethodBuilder[C],
    streamIR0: IR,
    env0: Emit.E,
    er: EmitRegion,
    container: Option[AggContainer]
  ): COption[SizedStream] = {
    assert(mb eq er.mb)

    def emitStream(streamIR: IR, env: Emit.E): COption[SizedStream] = {

      def emitIR(ir: IR, mb:  EmitMethodBuilder[C] = mb, env: Emit.E = env, container: Option[AggContainer] = container): EmitCode =
        emitter.emit(ir, mb, env, er, container)

      streamIR match {
        case NA(_) =>
          COption.None

        case x@StreamRange(startIR, stopIR, stepIR) =>
          val eltType = coerce[PStream](x.pType).elementType
          val step = mb.genFieldThisRef[Int]("sr_step")
          val start = mb.genFieldThisRef[Int]("sr_start")
          val stop = mb.genFieldThisRef[Int]("sr_stop")
          val llen = mb.genFieldThisRef[Long]("sr_llen")
          val len = mb.newLocal[Int]("sr_len")

          val startt = emitIR(startIR)
          val stopt = emitIR(stopIR)
          val stept = emitIR(stepIR)

          new COption[SizedStream] {
            def apply(none: Code[Ctrl], some: SizedStream => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
              Code(
                startt.setup,
                stopt.setup,
                stept.setup,
                (startt.m || stopt.m || stept.m).mux[Unit](
                  none,
                  Code(
                    start := startt.value,
                    stop := stopt.value,
                    step := stept.value,
                    (step ceq const(0)).orEmpty(Code._fatal[Unit]("Array range cannot have step size 0.")),
                    llen := (step < const(0)).mux(
                      (start <= stop).mux(const(0L), (start.toL - stop.toL - const(1L)) / (-step).toL + const(1L)),
                      (start >= stop).mux(const(0L), (stop.toL - start.toL - const(1L)) / step.toL + const(1L))),
                    (llen > const(Int.MaxValue.toLong)).mux[Unit](
                      Code._fatal[Unit]("Array range cannot have more than MAXINT elements."),
                      some(SizedStream(
                        range(mb, start, step, llen.toI)
                          .map(i => EmitCode(Code._empty, const(false), PCode(eltType, i))),
                        Some((len := llen.toI, len))))))))
            }
          }

        case ToStream(containerIR) =>
          val aType = coerce[PContainer](containerIR.pType)

          COption.fromEmitCode(emitIR(containerIR)).mapCPS { (containerAddr, k) =>
            val (asetup, a) = EmitCodeBuilder.scoped(mb) { cb =>
              containerAddr.asIndexable.memoize(cb, "ts_a")
            }

            val len = mb.newLocal[Int]("ts_len")
            val i = mb.newLocal[Int]("ts_i")
            val newStream = new Stream[EmitCode] {
              def apply(eos: Code[Ctrl], push: (EmitCode) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] =
                new Source[EmitCode](
                  setup0 = Code._empty,
                  setup = i := 0,
                  close = Code._empty,
                  close0 = Code._empty,
                  pull = (i < len).mux(
                    Code(i += 1,
                      push(
                        EmitCode.fromI(mb) { cb =>
                          a.loadElement(cb, i - 1)
                        })),
                    eos))
            }

            val sslen = mb.newLocal[Int]("ts_sslen")
            Code(
              asetup,
              len := a.loadLength(),
              k(SizedStream(
                newStream,
                Some((sslen := len, sslen)))))
          }

        case x@MakeStream(elements, t) =>
          val eltType = coerce[PStream](x.pType).elementType
          val stream = sequence(mb, eltType, elements.toFastIndexedSeq.map { ir =>
              val et = emitIR(ir)
              EmitCode(et.setup, et.m, PCode(eltType, eltType.copyFromTypeAndStackValue(er.mb, er.region, ir.pType, et.value)))
          })

          val len = mb.newLocal[Int]()

          COption.present(SizedStream(stream, Some((len := elements.length, len))))

        case x@ReadPartition(pathIR, spec, requestedType) =>
          val eltType = coerce[PStream](x.pType).elementType
          val strType = coerce[PString](pathIR.pType)

          val (_, dec) = spec.buildEmitDecoderF[Long](requestedType, mb.ecb)

          COption.fromEmitCode(emitIR(pathIR)).map { path =>
            val pathString = strType.loadString(path.tcode[Long])
            val xRowBuf = mb.newLocal[InputBuffer]()
            val stream = unfold[Code[Long]](
              Code._empty,
              Code._empty,
              (_, k) =>
                k(COption(
                  !xRowBuf.load().readByte().toZ,
                  dec(er.region, xRowBuf))))
            .map(
              EmitCode.present(eltType, _),
              setup0 = None,
              setup = Some(xRowBuf := spec
                .buildCodeInputBuffer(mb.getUnsafeReader(pathString, true))))

            SizedStream(stream, None)
          }

        case In(n, PStream(eltType, _)) =>
          val xIter = mb.newLocal[Iterator[RegionValue]]()

          new COption[Code[Iterator[RegionValue]]] {
            def apply(none: Code[Ctrl], some: (Code[Iterator[RegionValue]]) => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
              mb.getArg[Boolean](2 + 2 * n + 1).mux(
                none,
                some(mb.getArg[Iterator[RegionValue]](2 + 2 * n)))
            }
          }.map { iter =>
            val stream = unfold[Code[RegionValue]](
              Code._empty,
              Code._empty,
              (_, k) => k(COption(
                !xIter.load().hasNext,
                xIter.load().next()))
            ).map(
              rv => EmitCode.present(eltType, Region.loadIRIntermediate(eltType)(rv.invoke[Long]("getOffset"))),
              setup0 = None,
              setup = Some(xIter := iter)
            )

            SizedStream(stream, None)
          }

        case StreamMap(childIR, name, bodyIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR, env)
          optStream.map { case SizedStream(stream, len) =>
            val newStream = stream.map { eltt =>
              val xElt = mb.newEmitField(name, childEltType)
              val bodyenv = env.bind(name -> xElt)
              val bodyt = emitIR(bodyIR, env = bodyenv)

              EmitCode(
                Code(xElt := eltt,
                     bodyt.setup),
                bodyt.m,
                bodyt.pv)
            }

            SizedStream(newStream, len)
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR, env)

          optStream.map { case SizedStream(stream, len) =>
            val newStream = filter(stream
              .map { elt =>
                val xElt = mb.newEmitField(name, childEltType)
                val condEnv = env.bind(name -> xElt)
                val cond = emitIR(condIR, env = condEnv)

                new COption[EmitCode] {
                  def apply(none: Code[Ctrl], some: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
                    Code(
                      xElt := elt,
                      cond.setup,
                      (cond.m || !cond.value[Boolean]).mux(
                        none,
                        some(EmitCode(Code._empty, xElt.load.m, xElt.load.pv))
                      )
                    )
                  }
                }
              })

            SizedStream(newStream, None)
          }

        case StreamZip(as, names, bodyIR, behavior) =>
          val eltTypes = {
            val types = as.map(ir => coerce[PStream](ir.pType).elementType)
            behavior match {
              case ArrayZipBehavior.ExtendNA => types.map(_.setRequired(false))
              case _ => types
            }
          }
          val eltVars = (names, eltTypes).zipped.map(mb.newEmitField)

          val optStreams = COption.lift(as.map(emitStream(_, env)))

          optStreams.map { emitStreams =>
            val streams = emitStreams.map(_.stream)
            val lengths = emitStreams.map(_.length)

            behavior match {

              case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
                val newStream = multiZip(streams)
                  .map { elts =>
                    val bodyEnv = env.bind(names.zip(eltVars): _*)
                    val body = emitIR(bodyIR, env = bodyEnv)
                    EmitCode(Code(Code((eltVars, elts).zipped.map { (v, x) => v := x }), body.setup), body.m, body.pv)
                  }
                val newLength = behavior match {
                  case ArrayZipBehavior.TakeMinLength =>
                    lengths.reduceLeft(_.liftedZip(_).map {
                      case ((s1, l1), (s2, l2)) =>
                        (Code(s1, s2, (l1 > l2).orEmpty(l1 := l2)), l1)
                    })
                  case ArrayZipBehavior.AssumeSameLength =>
                    lengths.flatten.headOption
                }

                SizedStream(newStream, newLength)

              case ArrayZipBehavior.AssertSameLength =>
                // extend to infinite streams, where the COption becomes missing after EOS
                val extended: IndexedSeq[Stream[COption[EmitCode]]] =
                  streams.zipWithIndex.map { case (stream, i) =>
                    extendNA(mb, eltTypes(i), stream)
                  }

                // zip to an infinite stream, where the COption is missing when all streams are EOS
                val flagged: Stream[COption[EmitCode]] = multiZip(extended)
                  .mapCPS { (_, elts, k) =>
                    val allEOS = mb.newLocal[Boolean]("zip_stream_all_eos")
                    val anyEOS = mb.newLocal[Boolean]("zip_stream_any_eos")
                    // convert COption[TypedTriplet[_]] to TypedTriplet[_]
                    // where COption encodes if the stream has ended; update
                    // allEOS and anyEOS
                    val checkedElts: IndexedSeq[Code[Unit]] =
                      elts.zip(eltVars).map { case (optEC, eltVar) =>
                        optEC.cases(mb)(
                          anyEOS := true,
                          ec => Code(
                            allEOS := false,
                            eltVar := ec))
                      }

                    val bodyEnv = env.bind(names.zip(eltVars): _*)
                    val body = emitIR(bodyIR, env = bodyEnv)

                    Code(
                      allEOS := true,
                      anyEOS := false,
                      Code(checkedElts),
                      (anyEOS & !allEOS).mux[Unit](
                        Code._fatal[Unit]("zip: length mismatch"),
                        k(COption(allEOS, body))): Code[Ctrl])
                  }

                // termininate the stream when all streams are EOS
                val newStream = take(flagged)

                val newLength =
                    lengths.flatten.reduceLeftOption[(Code[Unit], Settable[Int])] {
                      case ((s1, l1), (s2, l2)) =>
                        (Code(s1,
                          s2,
                          l1.cne(l2).orEmpty(Code._fatal[Unit](
                            const("zip: length mismatch: ").concat(l1.toS).concat(", ").concat(l2.toS)))),
                          l1)
                    }

                SizedStream(newStream, newLength)

              case ArrayZipBehavior.ExtendNA =>
                // extend to infinite streams, where the COption becomes missing after EOS
                val extended: IndexedSeq[Stream[COption[EmitCode]]] =
                  streams.zipWithIndex.map { case (stream, i) =>
                    extendNA(mb, eltTypes(i), stream)
                  }

                // zip to an infinite stream, where the COption is missing when all streams are EOS
                val flagged: Stream[COption[EmitCode]] = multiZip(extended)
                  .mapCPS { (_, elts, k) =>
                    val allEOS = mb.newLocal[Boolean]()
                    // convert COption[TypedTriplet[_]] to TypedTriplet[_]
                    // where COption encodes if the stream has ended; update
                    // allEOS and anyEOS
                    val checkedElts: IndexedSeq[EmitCode] =
                      elts.zip(eltTypes).map { case (optET, t) =>
                        val optElt =
                          optET
                            .flatMapCPS[PCode] { (elt, _, k) =>
                              Code(allEOS := false,
                                   k(COption.fromEmitCode(elt)))
                            }

                        COption.toEmitCode(optElt, t, mb)
                      }
                    val bodyEnv = env.bind(names.zip(eltVars): _*)
                    val body = emitIR(bodyIR, env = bodyEnv)

                    Code(
                      allEOS := true,
                      Code((eltVars, checkedElts).zipped.map { (v, x) => v := x }),
                      k(COption(allEOS, body)))
                  }

                // termininate the stream when all streams are EOS
                val newStream = take(flagged)

                val newLength = behavior match {
                  case ArrayZipBehavior.ExtendNA =>
                    lengths.reduceLeft(_.liftedZip(_).map {
                      case ((s1, l1), (s2, l2)) =>
                        (Code(s1, s2, (l1 < l2).orEmpty(l1 := l2)), l1)
                    })
                  case ArrayZipBehavior.AssertSameLength =>
                    lengths.flatten.reduceLeftOption[(Code[Unit], Settable[Int])] {
                      case ((s1, l1), (s2, l2)) =>
                        (Code(s1,
                              s2,
                              l1.cne(l2).orEmpty(Code._fatal[Unit](
                                const("zip: length mismatch: ").concat(l1.toS).concat(", ").concat(l2.toS)))),
                          l1)
                    }
                }

                SizedStream(newStream, newLength)
            }
          }

        case StreamFlatMap(outerIR, name, innerIR) =>
          val outerEltType = coerce[PStream](outerIR.pType).elementType

          val optOuter = emitStream(outerIR, env)

          optOuter.map { outer =>
            val nested = outer.stream.mapCPS[COption[Stream[EmitCode]]] { (ctx, elt, k) =>
              val xElt = mb.newEmitField(name, outerEltType)
              val innerEnv = env.bind(name -> xElt)
              val optInner = emitStream(innerIR, innerEnv).map(_.stream)

              Code(
                xElt := elt,
                k(optInner))
            }

            SizedStream(flatMap(filter(nested)), None)
          }

        case If(condIR, thn, els) =>
          val eltType = coerce[PStream](thn.pType).elementType
          val xCond = mb.genFieldThisRef[Boolean]("stream_if_cond")

          val condT = COption.fromEmitCode(emitIR(condIR))
          val optLeftStream = emitStream(thn, env)
          val optRightStream = emitStream(els, env)

          // TODO: set xCond in setup of choose, don't need CPS
          condT.flatMapCPS[SizedStream] { (cond, _, k) =>
            val newOptStream = COption.choose[SizedStream](
              xCond,
              optLeftStream,
              optRightStream,
              { case (SizedStream(leftStream, lLen), SizedStream(rightStream, rLen)) =>
                  val newStream = mux(mb, eltType,
                    xCond,
                    leftStream,
                    rightStream)
                  val newLen = lLen.liftedZip(rLen).map { case ((s1, l1), (s2, l2)) =>
                    (Code(s1, s2, xCond.orEmpty(l2 := l1)), l2)
                  }

                SizedStream(newStream, newLen)
              })

            Code(xCond := cond.tcode[Boolean], k(newOptStream))
          }

        case Let(name, valueIR, bodyIR) =>
          val valueType = valueIR.pType
          val xValue = mb.newEmitField(name, valueType)

          val valuet = emitIR(valueIR)
          val bodyEnv = env.bind(name -> xValue)

          emitStream(bodyIR, bodyEnv).addSetup(xValue := valuet)

        case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = x.accPType

          val streamOpt = emitStream(childIR, env)
          streamOpt.map { case SizedStream(stream, len) =>
            val Lpush = CodeLabel()
            val hasPulled = mb.newLocal[Boolean]()

            val xElt = mb.newEmitField(eltName, eltType)
            val xAcc = mb.newEmitField(accName, accType)
            val tmpAcc = mb.newEmitField(accName, accType)

            val zero = emitIR(zeroIR).map(accType.copyFromPValue(mb, er.region, _))
            val bodyEnv = env.bind(accName -> tmpAcc, eltName -> xElt)

            val body = emitIR(bodyIR, env = bodyEnv).map(accType.copyFromPValue(mb, er.region, _))

            val newStream = new Stream[EmitCode] {
              def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
                val source = stream(
                  eos = eos,
                  push = a => Code(xElt := a, tmpAcc := xAcc, xAcc := body, Lpush, push(xAcc)))

                Source[EmitCode](
                  setup0 = source.setup0,
                  setup = Code(hasPulled := false, xAcc := zero, source.setup),
                  close = source.close,
                  close0 = source.close0,
                  pull = hasPulled.mux(
                    source.pull,
                    Code(hasPulled := true, Lpush.goto)))
              }
            }

            val newLen = len.map { case (s, l) => (Code(s, l := l + 1), l) }
            SizedStream(newStream, newLen)
          }

        case x@RunAggScan(array, name, init, seqs, result, _) =>
          val aggs = x.physicalSignatures
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromClassBuilder(aggs, mb.ecb, "array_agg_scan")

          val eltType = coerce[PStream](array.pType).elementType

          val xElt = mb.newEmitField("aggscan_elt", eltType)
          val xResult = mb.newEmitField("aggscan_result", result.pType)

          val bodyEnv = env.bind(name -> xElt)
          val cInit = emitIR(init, container = Some(newContainer))
          val seqPerElt = emitIR(seqs, env = bodyEnv, container = Some(newContainer))
          val postt = emitIR(result, env = bodyEnv, container = Some(newContainer))

          val optStream = emitStream(array, env)

          optStream.map { case SizedStream(stream, len) =>
            val newStream = stream.map[EmitCode](
              { eltt =>
                EmitCode(
                  Code(
                    xElt := eltt,
                    xResult := postt,
                    seqPerElt.setup),
                  xResult.get)
              },
              setup0 = Some(aggSetup),
              close0 = Some(aggCleanup),
              setup = Some(cInit.setup))

            SizedStream(newStream, len)
          }

        case StreamLeftJoinDistinct(leftIR, rightIR, leftName, rightName, compIR, joinIR) =>
          val lEltType = coerce[PStream](leftIR.pType).elementType
          val rEltType = coerce[PStream](rightIR.pType).elementType.setRequired(false)
          val xLElt = mb.newEmitField("join_lelt", lEltType)
          val xRElt = mb.newEmitField("join_relt", rEltType)

          val env2 = env.bind(leftName -> xLElt, rightName -> xRElt)

          def compare(lelt: EmitCode, relt: EmitCode): Code[Int] = {
            val compt = emitIR(compIR, env = env2)
            Code(
              xLElt := lelt,
              xRElt := relt,
              compt.setup,
              compt.m.orEmpty(Code._fatal[Unit]("StreamLeftJoinDistinct: comp can't be missing")),
              coerce[Int](compt.v))
          }

          emitStream(leftIR, env).flatMap { case SizedStream(leftStream, leftLen) =>
            emitStream(rightIR, env).map { case SizedStream(rightStream, _) =>
              val newStream = leftJoinRightDistinct(
                mb,
                lEltType, leftStream,
                rEltType, rightStream,
                compare)
                .map { case (lelt, relt) =>
                  val joint = emitIR(joinIR, env = env2)
                  EmitCode(
                    Code(xLElt := lelt, xRElt := relt, joint.setup),
                    joint.m,
                    joint.pv)
                }

              SizedStream(newStream, leftLen)
            }
          }

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }
    }

    emitStream(streamIR0, env0)
  }
}
