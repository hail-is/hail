package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValue, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.types.physical._
import is.hail.io.{AbstractTypedCodecSpec, InputBuffer}
import is.hail.utils._

import scala.language.{existentials, higherKinds}
import scala.reflect.ClassTag

case class EmitStreamContext(mb: MethodBuilder, jb: JoinPointBuilder)

abstract class COption[+A] { self =>
  def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl]

  def cases[B: TypeInfo](mb: MethodBuilder)(none: Code[B], some: A => Code[B]): Code[B] =
    JoinPoint.CallCC[Code[B]]((jb, ret) => apply(ret(none), a => ret(some(a)))(EmitStreamContext(mb, jb)))

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
      val noneJP = ctx.jb.joinPoint()
      noneJP.define(_ => none)
      self.apply(noneJP(()), f(_)(noneJP(()), some))
    }
  }

  def filter(cond: Code[Boolean]): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val noneJP = ctx.jb.joinPoint()
      noneJP.define(_ => none)
      cond.mux(noneJP(()), self.apply(noneJP(()), some))
    }
  }

  def flatMapCPS[B](f: (A, EmitStreamContext, COption[B] => Code[Ctrl]) => Code[Ctrl]): COption[B] = new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val noneJP = ctx.jb.joinPoint()
      noneJP.define(_ => none)
      self.apply(noneJP(()), a => f(a, ctx, optB => optB(noneJP(()), some)))
    }
  }
}

object COption {
  def apply[A](missing: Code[Boolean], value: A): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      missing.mux(none, some(value))
  }

  // None is the only COption allowed to not call `some` at compile time
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
      val noneJP = ctx.jb.joinPoint()
      noneJP.define(_ => none)

      def nthOpt(i: Int, acc: IndexedSeq[A]): Code[Ctrl] =
        if (i == opts.length - 1)
          opts(i)(noneJP(()), a => some(acc :+ a))
        else
          opts(i)(noneJP(()), a => nthOpt(i+1, acc :+ a))

      nthOpt(0, FastIndexedSeq())
    }
  }

  // Returns a COption value equivalent to 'left' when 'useLeft' is true,
  // otherwise returns a value equivalent to 'right'. In the case where neither
  // 'left' nor 'right' are missing, uses 'fuse' to combine the values.
  // Presumably 'fuse' dynamically chooses one or the other based on the same
  // boolean passed in 'useLeft. 'fuse' is needed because we don't require
  // a ParameterPack[A]
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
          val noneJP = ctx.jb.joinPoint()
          noneJP.define(_ => none)
          val k = ctx.jb.joinPoint()
          val runLeft = left(noneJP(()), a => {l = Some(a); k(())})
          val runRight = right(noneJP(()), a => {r = Some(a); k(())})

          k.define(_ => some(fuse(l.get, r.get)))

          useLeft.mux(runLeft, runRight)
        }
      }
    }

  def fromEmitTriplet[A](et: EmitCode): COption[Code[A]] = new COption[Code[A]] {
    def apply(none: Code[Ctrl], some: Code[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      Code(et.setup, et.m.mux(none, some(coerce[A](et.v))))
    }
  }

  def fromTypedTriplet(et: EmitCode): COption[Code[_]] = fromEmitTriplet(et)

  def toEmitTriplet(opt: COption[Code[_]], t: PType, mb: MethodBuilder): EmitCode = {
    val ti: TypeInfo[_] = typeToTypeInfo(t)
    val m = mb.newLocal[Boolean]
    val v = mb.newLocal(ti)
    val setup = JoinPoint.CallCC[Unit] { (jb, ret) =>
      opt(Code(m := true, v.storeAny(defaultValue(ti)), ret(())),
          a => Code(m := false, v.storeAny(a), ret(())))(EmitStreamContext(mb, jb))
    }
    EmitCode(setup, m, PCode(t, v.load()))
  }

  def toTypedTriplet(t: PType, mb: MethodBuilder, opt: COption[Code[_]]): TypedTriplet[t.type] =
    TypedTriplet(t, toEmitTriplet(opt, t, mb))
}

object CodeStream { self =>
  import is.hail.asm4s.joinpoint.JoinPoint.CallCC
  import is.hail.asm4s.joinpoint._
  def newLocal[T: ParameterPack](implicit ctx: EmitStreamContext): ParameterStore[T] = implicitly[ParameterPack[T]].newLocals(ctx.mb)
  def joinPoint()(implicit ctx: EmitStreamContext): DefinableJoinPoint[Unit] = ctx.jb.joinPoint()
  def joinPoint[T: ParameterPack](implicit ctx: EmitStreamContext): DefinableJoinPoint[T] = ctx.jb.joinPoint[T](ctx.mb)

  private case class Source[+A](setup0: Code[Unit], close0: Code[Unit], setup: Code[Unit], close: Code[Unit], pull: Code[Ctrl])

  abstract class Stream[+A] {
    private[CodeStream] def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A]

    def fold[S: ParameterPack](mb: MethodBuilder)(s0: S, f: (A, S) => S, ret: S => Code[Ctrl]): Code[Ctrl] =
      CodeStream.fold(mb, this, s0, f, ret)
    def foldCPS[S: ParameterPack](mb: MethodBuilder)(s0: S, f: (A, S, S => Code[Ctrl]) => Code[Ctrl], ret: S => Code[Ctrl]): Code[Ctrl] =
      CodeStream.foldCPS(mb, this, s0, f, ret)
    def forEach(mb: MethodBuilder)(f: A => Code[Unit]): Code[Unit] =
      CodeStream.forEach(mb, this, f)
    def mapCPS[B](
      f: (EmitStreamContext, A, B => Code[Ctrl]) => Code[Ctrl],
      setup0: Option[Code[Unit]] = None,
      setup:  Option[Code[Unit]] = None,
      close0: Option[Code[Unit]] = None,
      close:  Option[Code[Unit]] = None
    ): Stream[B] = CodeStream.mapCPS(this)(f, setup0, setup, close0, close)
    def map[B](
      f: A => B,
      setup0: Option[Code[Unit]] = None,
      setup:  Option[Code[Unit]] = None,
      close0: Option[Code[Unit]] = None,
      close:  Option[Code[Unit]] = None
    ): Stream[B] = CodeStream.map(this)(f, setup0, setup, close0, close)
    def flatMap[B](f: A => Stream[B]): Stream[B] =
      CodeStream.flatMap(map(f))
    def scanCPS[S: ParameterPack](
      mb: MethodBuilder, s0: S
    )(f: (A, S, S => Code[Ctrl]) => Code[Ctrl]
    ): Stream[S] = {
      val (res, _) = CodeStream.scanCPS(mb, this, s0, f)
      res
    }
    def scan[S: ParameterPack](mb: MethodBuilder, s0: S)(f: (A, S) => S): Stream[S] =
      scanCPS(mb, s0)((a, s, k) => k(f(a, s)))
    def longScanCPS[S: ParameterPack](
      mb: MethodBuilder, s0: S
    )(f: (A, S, S => Code[Ctrl]) => Code[Ctrl]
    ): Stream[S] =
      CodeStream.longScanCPS(mb, this, s0, f)
    def longScan[S: ParameterPack](
      mb: MethodBuilder, s0: S
    )(f: (A, S) => S
    ): Stream[S] =
      longScanCPS(mb, s0)((a, s, k) => k(f(a, s)))
  }

  implicit class StreamPP[A](val stream: Stream[A]) extends AnyVal {
    def filter(cond: A => Code[Boolean])(implicit pp: ParameterPack[A]): Stream[A] =
      CodeStream.filter(stream, cond)
  }

  def unfold[A, S: ParameterPack](
    s0: S,
    f: (S, EmitStreamContext, COption[(A, S)] => Code[Ctrl]) => Code[Ctrl]
  ): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val s = newLocal[S]
      Source[A](
        setup0 = s.init,
        close0 = Code._empty,
        setup = s := s0,
        close = Code._empty,
        pull = f(s.load, ctx, _.apply(
          none = eos,
          // Warning: `a` should not depend on `s`
          some = { case (a, s1) => Code(s := s1, push(a)) })))
    }
  }

  def range(start: Code[Int], step: Code[Int], len: Code[Int]): Stream[Code[Int]] =
    unfold[Code[Int], (Code[Int], Code[Int])](
      s0 = (start, len),
      f = { case ((cur, rem), _ctx, k) =>
        implicit val ctx = _ctx
        val xCur = newLocal[Code[Int]]
        val xRem = newLocal[Code[Int]]
        Code(
          xCur := cur,
          xRem := rem - 1,
          k(COption(xRem.load < 0,
                    (xCur.load, (xCur.load + step, xRem.load)))))
      })

  def foldCPS[A, S: ParameterPack](
    mb: MethodBuilder,
    stream: Stream[A],
    s0: S,
    f: (A, S, S => Code[Ctrl]) => Code[Ctrl],
    ret: S => Code[Ctrl]
  ): Code[Ctrl] = {
    val (scan, s) = scanCPS(mb, stream, s0, f)
    Code(run(mb, scan.map(_ => ())), ret(s.load))
  }

  def fold[A, S: ParameterPack](mb: MethodBuilder, stream: Stream[A], s0: S, f: (A, S) => S, ret: S => Code[Ctrl]): Code[Ctrl] =
    foldCPS[A, S](mb, stream, s0, (a, s, k) => k(f(a, s)), ret)

  def forEachCPS[A](mb: MethodBuilder, stream: Stream[A], f: (A, Code[Ctrl]) => Code[Ctrl]): Code[Unit] =
    run(mb, stream.mapCPS[Unit]((_, a, k) => f(a, k(()))))

  def forEach[A](mb: MethodBuilder, stream: Stream[A], f: A => Code[Unit]): Code[Unit] =
    run(mb, stream.mapCPS((_, a, k) => Code(f(a), k(()))))

  def run(mb: MethodBuilder, stream: Stream[Unit]): Code[Unit] = {
    CallCC[Unit] { (jb, ret) =>
      implicit val ctx = EmitStreamContext(mb, jb)
      val pullJP = joinPoint()
      val eosJP = joinPoint()
      val source = stream(eos = eosJP(()), push = _ => pullJP(()))
      pullJP.define(_ => source.pull)
      eosJP.define(_ => Code(source.close, source.close0, ret(())))
      Code(source.setup0, source.setup, pullJP(()))
    }
  }

  // Inclusive scan: s0 is not first element, last element is the total fold
  def scanCPS[A, S: ParameterPack](
    mb: MethodBuilder,
    stream: Stream[A],
    s0: S,
    f: (A, S, S => Code[Ctrl]) => Code[Ctrl]
  ): (Stream[S], ParameterStore[S]) = {
    val s = implicitly[ParameterPack[S]].newLocals(mb)
    val res = mapCPS[A, S](stream)(
      (_, a, k) => f(a, s.load, s1 => Code(s := s1, k(s.load))),
      setup0 = Some(s.init),
      setup = Some(s := s0))

    (res, s)
  }

  // the length+1 scan
  def longScanCPS[A, S: ParameterPack](
    mb: MethodBuilder,
    stream: Stream[A],
    s0: S,
    f: (A, S, S => Code[Ctrl]) => Code[Ctrl]
  ): Stream[S] = new Stream[S] {
    def apply(eos: Code[Ctrl], push: S => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[S] = {
      val hasPulled = newLocal[Code[Boolean]]
      val s = newLocal[S]
      val pushJP = joinPoint()
      pushJP.define(_ => push(s.load))
      val source = stream(
        eos = eos,
        push = a => f(a, s.load, s1 => Code(s := s1, pushJP(()))))
      Source[S](
        setup0 = Code(hasPulled := false, s.init, source.setup0),
        close0 = source.close0,
        setup = Code(hasPulled := false, s := s0, source.setup),
        close = source.close,
        pull = hasPulled.load.mux(source.pull, Code(hasPulled := true, pushJP(())))
      )
    }
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
      val outerPullJP = joinPoint()
      var innerSource: Source[A] = null
      val innerPullJP = joinPoint()
      val innerEosJP = joinPoint()
      val inInnerStream = newLocal[Code[Boolean]]
      val outerSource = outer(
        eos = eos,
        push = inner => {
          innerSource = inner(
            eos = innerEosJP(()),
            push = push)
          innerPullJP.define(_ => innerSource.pull)
          innerEosJP.define(_ => Code(innerSource.close, outerPullJP(())))
          Code(innerSource.setup, inInnerStream := true, innerPullJP(()))
        })
      outerPullJP.define(_ => Code(inInnerStream := false, outerSource.pull))
      Source[A](
        setup0 = Code(inInnerStream := const(false), outerSource.setup0, innerSource.setup0),
        close0 = Code(innerSource.close0, outerSource.close0),
        setup = Code(inInnerStream := false, outerSource.setup),
        close = Code(inInnerStream.load.mux(innerSource.close, Code._empty), outerSource.close),
        pull = JoinPoint.mux(inInnerStream.load, innerPullJP, outerPullJP))
    }
  }

  def filter[A](stream: Stream[COption[A]]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val pullJP = joinPoint()
      val source = stream(
        eos = eos,
        push = _.apply(none = pullJP(()), some = push))
      pullJP.define(_ => source.pull)
      source.copy(pull = pullJP(()))
    }
  }

  def filter[A: ParameterPack](stream: Stream[A], cond: A => Code[Boolean]): Stream[A] =
    filter(mapCPS[A, COption[A]](stream)((_ctx, a, k) => {
      implicit val ctx = _ctx
      val as = newLocal[A]
      Code(as := a, k(COption(!cond(as.load), as.load)))
    }))

  def take[A](stream: Stream[COption[A]]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val eosJP = joinPoint()
      eosJP.define(_ => eos)
      stream(
        eos = eosJP(()),
        push = _.apply(none = eosJP(()), some = push)).asInstanceOf[Source[A]]
    }
  }

  def zip[A, B](left: Stream[A], right: Stream[B]): Stream[(A, B)] = new Stream[(A, B)] {
    def apply(eos: Code[Ctrl], push: ((A, B)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(A, B)] = {
      val eosJP = joinPoint()
      var rightSource: Source[B] = null
      val leftSource = left(
        eos = eosJP(()),
        push = a => {
          rightSource = right(
            eos = eosJP(()),
            push = b => push((a, b)))
          rightSource.pull
        })
      eosJP.define(_ => eos)

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
      val eosJP = joinPoint()

      def nthSource(n: Int, acc: IndexedSeq[A]): Source[A] = {
        if (n == streams.length - 1) {
          streams(n)(eosJP(()), c => push(acc :+ c))
        } else {
          var rest: Source[A] = null
          val src = streams(n)(
            eosJP(()),
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

      val source = nthSource(0, IndexedSeq.empty)
      eosJP.define(_ => eos)

      source.asInstanceOf[Source[IndexedSeq[A]]]
    }
  }

  def leftJoinRightDistinct[A: ParameterPack, B: ParameterPack](
    left: Stream[A],
    right: Stream[B],
    rNil: B,
    comp: (A, B) => Code[Int]
  ): Stream[(A, B)] = new Stream[(A, B)] {
    def apply(eos: Code[Ctrl], push: ((A, B)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(A, B)] = {
      val pulledRight = newLocal[Code[Boolean]]
      val rightEOS = newLocal[Code[Boolean]]
      val xA = newLocal[A] // last value received from left
      val xB = newLocal[B] // last value received from right
      val xOutB = newLocal[B] // B value to push (may be rNil while xB is not)
      val xNilB = newLocal[B] // saved rNil

      var rightSource: Source[B] = null
      val leftSource = left(
        eos = eos,
        push = a => {
          val pushJP = joinPoint()
          val pullRightJP = joinPoint()
          val compareJP = joinPoint()

          pushJP.define(_ => push((xA.load, xOutB.load)))

          compareJP.define(_ => {
            val c = newLocal[Code[Int]]
            Code(
              c := comp(xA.load, xB.load),
              (c.load > 0).mux(
                pullRightJP(()),
                (c.load < 0).mux(
                  Code(xOutB := xNilB.load, pushJP(())),
                  Code(xOutB := xB.load, pushJP(())))))
          })

          rightSource = right(
            eos = Code(rightEOS := true, xOutB := xNilB.load, pushJP(())),
            push = b => Code(xB := b, compareJP(())))

          pullRightJP.define(_ => rightSource.pull)

          Code(
            xA := a,
            pulledRight.load.mux(
              rightEOS.load.mux(pushJP(()), compareJP(())),
              Code(pulledRight := true, pullRightJP(()))))
        })

      Source[(A, B)](
        setup0 = Code(pulledRight.init, rightEOS.init, xA.init, xB.init, xOutB.init, xNilB.init, leftSource.setup0, rightSource.setup0),
        close0 = Code(leftSource.close0, rightSource.close0),
        setup = Code(pulledRight := false, rightEOS := false, xNilB := rNil, leftSource.setup, rightSource.setup),
        close = Code(leftSource.close, rightSource.close),
        pull = leftSource.pull)
    }
  }

  def extendNA[A: ParameterPack](stream: Stream[A]): Stream[COption[A]] = new Stream[COption[A]] {
    def apply(eos: Code[Ctrl], push: COption[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[COption[A]] = {
      val atEnd = newLocal[Code[Boolean]]
      val x = newLocal[A]
      val pushJP = joinPoint()
      val source = stream(Code(atEnd := true, pushJP(())), a => Code(x := a, pushJP(())))
      pushJP.define(_ => push(COption(atEnd.load, x.load)))
      Source[COption[A]](
        setup0 = Code(atEnd := false, x.init, source.setup0),
        close0 = source.close0,
        setup = Code(atEnd := false, x.init, source.setup),
        close = source.close,
        pull = atEnd.load.mux(pushJP(()), source.pull))
    }
  }

  def mux[A: ParameterPack](cond: Code[Boolean], left: Stream[A], right: Stream[A]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val b = newLocal[Code[Boolean]]
      val eosJP = joinPoint()
      val pushJP = joinPoint[A]

      eosJP.define(_ => eos)
      pushJP.define(push)

      val l = left(eosJP(()), pushJP(_))
      val r = right(eosJP(()), pushJP(_))

      val lPullJP = joinPoint()
      val rPullJP = joinPoint()

      lPullJP.define(_ => l.pull)
      rPullJP.define(_ => r.pull)
      Source[A](
        setup0 = Code(b := false, l.setup0, r.setup0),
        close0 = Code(l.close0, r.close0),
        setup = Code(b := cond, b.load.mux(l.setup, r.setup)),
        close = b.load.mux(l.close, r.close),
        pull = JoinPoint.mux(b.load, lPullJP, rPullJP))
    }
  }

  def sequence[A: ParameterPack](elements: Seq[A]): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val i = newLocal[Code[Int]]
      val eosJP = joinPoint()
      val pushJP = joinPoint[A]

      eosJP.define(_ => eos)
      pushJP.define(a => Code(i := i.load + 1, push(a)))

      Source[A](
        setup0 = i := 0,
        close0 = Code._empty,
        setup = i := 0,
        close = Code._empty,
        pull = JoinPoint.switch(i.load, eosJP, elements.map { elt =>
          val j = joinPoint()
          j.define(_ => pushJP(elt))
          j
        }))
    }
  }
}

object EmitStream {
  import CodeStream._

  def write(mb: MethodBuilder, sstream: SizedStream, ab: StagedArrayBuilder): Code[Unit] = {
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

  def toArray(mb: MethodBuilder, aTyp: PArray, optStream: COption[SizedStream]): EmitCode = {
    val srvb = new StagedRegionValueBuilder(mb, aTyp)
    val result = optStream.map { ss =>
      ss.length match {
        case None =>
          val xLen = mb.newLocal[Int]
          val i = mb.newLocal[Int]
          val vab = new StagedArrayBuilder(aTyp.elementType, mb, 0)
          Code(
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
            srvb.offset)

        case Some((setupLen, len)) =>
          Code(
            setupLen,
            srvb.start(len),
            ss.stream.forEach(mb) { et =>
              Code(
                et.setup,
                et.m.mux(srvb.setMissing(), srvb.addIRIntermediate(aTyp.elementType)(et.v)),
                srvb.advance())
            },
            srvb.offset)
      }
    }

    COption.toEmitTriplet(result, aTyp, mb)
  }

  // length is required to be a variable reference
  case class SizedStream(stream: Stream[EmitCode], length: Option[(Code[Unit], Settable[Int])])

  private[ir] def apply(
    emitter: Emit,
    mb: EmitMethodBuilder,
    streamIR0: IR,
    env0: Emit.E,
    er: EmitRegion,
    container: Option[AggContainer]
  ): COption[SizedStream] = {
    assert(emitter.fb eq mb.fb)
    assert(mb eq er.mb)
    val fb = mb.fb

    def emitStream(streamIR: IR, env: Emit.E): COption[SizedStream] = {

      def emitIR(ir: IR, mb:  EmitMethodBuilder = mb, env: Emit.E = env, container: Option[AggContainer] = container): EmitCode =
        emitter.emit(ir, mb, env, er, container)

      streamIR match {

        case NA(_) =>
          COption.None

        case x@StreamRange(startIR, stopIR, stepIR) =>
          val eltType = coerce[PStream](x.pType).elementType
          val step = fb.newField[Int]("sr_step")
          val start = fb.newField[Int]("sr_start")
          val stop = fb.newField[Int]("sr_stop")
          val llen = fb.newField[Long]("sr_llen")
          val len = mb.newLocal[Int]

          val startt = emitIR(startIR)
          val stopt = emitIR(stopIR)
          val stept = emitIR(stepIR)

          new COption[SizedStream] {
            def apply(none: Code[Ctrl], some: SizedStream => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
              Code(
                startt.setup,
                stopt.setup,
                stept.setup,
                (startt.m || stopt.m || stept.m).mux(
                  none,
                  Code(
                    start := startt.value,
                    stop := stopt.value,
                    step := stept.value,
                    (step ceq const(0)).orEmpty(Code._fatal[Unit]("Array range cannot have step size 0.")),
                    llen := (step < const(0)).mux(
                      (start <= stop).mux(const(0L), (start.toL - stop.toL - const(1L)) / (-step).toL + const(1L)),
                      (start >= stop).mux(const(0L), (stop.toL - start.toL - const(1L)) / step.toL + const(1L))),
                    (llen > const(Int.MaxValue.toLong)).mux(
                      Code._fatal[Ctrl]("Array range cannot have more than MAXINT elements."),
                      some(SizedStream(
                        range(start, step, llen.toI)
                          .map(i => EmitCode(Code._empty, const(false), PCode(eltType, i))),
                        Some((len := llen.toI, len))))))))
            }
          }

        case ToStream(containerIR) =>
          val aType = coerce[PContainer](containerIR.pType)
          val eltType = aType.elementType

          COption.fromEmitTriplet[Long](emitIR(containerIR)).mapCPS { (containerAddr, k) =>
            val xAddr = fb.newField[Long]("a_off")
            val newStream = range(0, 1, aType.loadLength(xAddr)).map { i =>
              EmitCode(
                Code._empty,
                aType.isElementMissing(xAddr, i),
                PCode(eltType, Region.loadIRIntermediate(eltType)(aType.elementOffset(xAddr, i))))
            }
            val len = mb.newLocal[Int]

            Code(
              xAddr := containerAddr,
              k(SizedStream(
                newStream,
                Some((len := aType.loadLength(xAddr), len)))))
          }

        case x@MakeStream(elements, t) =>
          val eltType = coerce[PStream](x.pType).elementType
          implicit val eltPack = TypedTriplet.pack(eltType)

          val stream = sequence(elements.map {
            ir => TypedTriplet(eltType, {
              val et = emitIR(ir)
              EmitCode(et.setup, et.m, PCode(eltType, eltType.copyFromTypeAndStackValue(er.mb, er.region, ir.pType, et.value)))
            })
          }).map(_.untyped)

          val len = mb.newLocal[Int]

          COption.present(SizedStream(stream, Some((len := elements.length, len))))

        case x@ReadPartition(pathIR, spec, requestedType) =>
          val eltType = coerce[PStream](x.pType).elementType
          val strType = coerce[PString](pathIR.pType)

          val (_, dec) = spec.buildEmitDecoderF[Long](requestedType, fb)

          COption.fromEmitTriplet[Long](emitIR(pathIR)).map { path =>
            val pathString = strType.loadString(path)
            val xRowBuf = mb.newLocal[InputBuffer]
            val stream = unfold[Code[Long], Unit](
              (),
              (_, _, k) =>
                k(COption(
                  !xRowBuf.load().readByte().toZ,
                  (dec(er.region, xRowBuf), ())))
            ).map(
              EmitCode.present(eltType, _),
              setup0 = Some(xRowBuf := Code._null),
              setup = Some(xRowBuf := spec
                .buildCodeInputBuffer(fb.getUnsafeReader(pathString, true))))

            SizedStream(stream, None)
          }

        case In(n, streamType@PStream(eltType, _)) =>
          val xIter = mb.newLocal[Iterator[RegionValue]]

          COption.fromEmitTriplet[Iterator[RegionValue]](
            emitter.normalArgument(mb, n, streamType)
          ).map { iter =>
            val stream = unfold[Code[RegionValue], Unit](
              (),
              (_, _, k) => k(COption(
                !xIter.load().hasNext,
                (xIter.load().next(), ())))
            ).map(
              rv => EmitCode.present(eltType, Region.loadIRIntermediate(eltType)(rv.invoke[Long]("getOffset"))),
              setup0 = Some(xIter := Code._null),
              setup = Some(xIter := iter)
            )

            SizedStream(stream, None)
          }

        case StreamMap(childIR, name, bodyIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType
          implicit val childEltPack = TypedTriplet.pack(childEltType)

          val optStream = emitStream(childIR, env)
          optStream.map { case SizedStream(stream, len) =>
            val newStream = stream.map { eltt =>
              val xElt = childEltPack.newFields(mb.fb, name)
              val bodyenv = Emit.bindEnv(env, name -> xElt)
              val bodyt = emitIR(bodyIR, env = bodyenv)

              EmitCode(
                Code(xElt := TypedTriplet(childEltType, eltt),
                     bodyt.setup),
                bodyt.m,
                bodyt.pv)
            }

            SizedStream(newStream, len)
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType
          implicit val childEltPack = TypedTriplet.pack(childEltType)

          val optStream = emitStream(childIR, env)

          optStream.map { case SizedStream(stream, len) =>
            val newStream = filter(stream
              .map { elt =>
                val xElt = childEltPack.newFields(mb.fb, name)
                val condEnv = Emit.bindEnv(env, name -> xElt)
                val cond = emitIR(condIR, env = condEnv)

                new COption[EmitCode] {
                  def apply(none: Code[Ctrl], some: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
                    Code(
                      xElt := TypedTriplet(childEltType, elt),
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
          val eltsPack = ParameterPack.array(eltTypes.map(TypedTriplet.pack(_)))
          val eltVars = eltsPack.newFields(mb.fb, names)

          val optStreams = COption.lift(as.map(emitStream(_, env)))

          optStreams.map { emitStreams =>
            val streams = emitStreams.map(_.stream)
            val lengths = emitStreams.map(_.length)

            behavior match {

              case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
                val newStream = multiZip(streams)
                  .map { elts =>
                    val bodyEnv = Emit.bindEnv(env, names.zip(eltVars.pss.asInstanceOf[IndexedSeq[ParameterStoreTriplet[_]]]): _*)
                    val body = emitIR(bodyIR, env = bodyEnv)
                    val typedElts = eltTypes.zip(elts).map { case (t, v) => TypedTriplet(t, v) }
                    EmitCode(Code(eltVars := typedElts, body.setup), body.m, body.pv)
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

              case behavior@(ArrayZipBehavior.ExtendNA | ArrayZipBehavior.AssertSameLength) =>
                // extend to infinite streams, where the COption becomes missing after EOS
                val extended: IndexedSeq[Stream[COption[TypedTriplet[_]]]] =
                  streams.zipWithIndex.map { case (stream, i) =>
                    val t = eltTypes(i)

                    extendNA[TypedTriplet[_]](stream.map(TypedTriplet(t, _)))(eltsPack.pps(i).asInstanceOf[ParameterPack[TypedTriplet[_]]])
                  }

                // zip to an infinite stream, where the COption is missing when all streams are EOS
                val flagged: Stream[COption[EmitCode]] = multiZip(extended)
                  .mapCPS { (_, elts, k) =>
                    val assert = behavior == ArrayZipBehavior.AssertSameLength
                    val allEOS = mb.newLocal[Boolean]
                    val anyEOS = if (assert) mb.newLocal[Boolean] else null
                    // convert COption[TypedTriplet[_]] to TypedTriplet[_]
                    // where COption encodes if the stream has ended; update
                    // allEOS and anyEOS
                    val checkedElts: IndexedSeq[TypedTriplet[_]] =
                      elts.zip(eltTypes).map { case (optET, t) =>
                        val optElt =
                          (if (assert) optET.doIfNone(anyEOS := true) else optET)
                            .flatMapCPS[Code[_]] { (elt, _, k) =>
                              Code(allEOS := false,
                                   k(COption.fromEmitTriplet(elt.untyped)))
                            }

                        COption.toTypedTriplet(t, mb, optElt)
                      }
                    val bodyEnv = Emit.bindEnv(env, names.zip(eltVars.pss.asInstanceOf[IndexedSeq[ParameterStoreTriplet[_]]]): _*)
                    val body = emitIR(bodyIR, env = bodyEnv)

                    Code(
                      allEOS := true,
                      if (assert) anyEOS := false else Code._empty,
                      eltVars := checkedElts,
                      if (assert)
                        (anyEOS & !allEOS).mux(
                          Code._fatal[Ctrl]("zip: length mismatch"),
                          k(COption(allEOS, body)))
                      else
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
          val outerEltPack = TypedTriplet.pack(outerEltType)

          val optOuter = emitStream(outerIR, env)

          optOuter.map { outer =>
            val nested = outer.stream.mapCPS[COption[Stream[EmitCode]]] { (ctx, elt, k) =>
              val xElt = outerEltPack.newFields(ctx.mb.fb, name)
              val innerEnv = Emit.bindEnv(env, name -> xElt)
              val optInner = emitStream(innerIR, innerEnv).map(_.stream)

              Code(
                xElt := TypedTriplet(outerEltType, elt),
                k(optInner))
            }

            SizedStream(flatMap(filter(nested)), None)
          }

        case If(condIR, thn, els) =>
          val eltType = coerce[PStream](thn.pType).elementType
          implicit val eltPack: ParameterPack[TypedTriplet[eltType.type]] = TypedTriplet.pack(eltType)
          val xCond = mb.newField[Boolean]

          val condT = COption.fromEmitTriplet[Boolean](emitIR(condIR))
          val optLeftStream = emitStream(thn, env)
          val optRightStream = emitStream(els, env)

          // TODO: set xCond in setup of choose, don't need CPS
          condT.flatMapCPS[SizedStream] { (cond, _, k) =>
            val newOptStream = COption.choose[SizedStream](
              xCond,
              optLeftStream,
              optRightStream,
              { case (SizedStream(leftStream, lLen), SizedStream(rightStream, rLen)) =>
                  val newStream = mux(
                    xCond,
                    leftStream.map(TypedTriplet(eltType, _)),
                    rightStream.map(TypedTriplet(eltType, _))
                    ).map(_.untyped)
                  val newLen = lLen.liftedZip(rLen).map { case ((s1, l1), (s2, l2)) =>
                    (Code(s1, s2, xCond.orEmpty(l2 := l1)), l2)
                  }

                SizedStream(newStream, newLen)
              })

            Code(xCond := cond, k(newOptStream))
          }

        case Let(name, valueIR, bodyIR) =>
          val valueType = valueIR.pType
          val valuePack = TypedTriplet.pack(valueType)
          val xValue = valuePack.newFields(mb.fb, name)

          val valuet = TypedTriplet(valueType, emitIR(valueIR))
          val bodyEnv = Emit.bindEnv(env, name -> xValue)

          emitStream(bodyIR, bodyEnv).addSetup(xValue := valuet)

        case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = x.accPType
          val eltPack = TypedTriplet.pack(eltType)
          implicit val accPack = TypedTriplet.pack(accType)

          def scanBody(elt: TypedTriplet[eltType.type], acc: TypedTriplet[accType.type]): TypedTriplet[accType.type] = {
            val xElt = eltPack.newFields(fb, eltName)
            val xAcc = accPack.newFields(fb, accName)
            val bodyEnv = Emit.bindEnv(env, accName -> xAcc, eltName -> xElt)

            val bodyT = TypedTriplet(accType, emitIR(bodyIR, env = bodyEnv).map(accType.copyFromPValue(mb, er.region, _)))
            TypedTriplet(accType, EmitCode(Code(xElt := elt, xAcc := acc, bodyT.setup), bodyT.m, bodyT.pv))
          }

          val zerot = TypedTriplet(accType, emitIR(zeroIR).map(accType.copyFromPValue(mb, er.region, _)))
          val streamOpt = emitStream(childIR, env)

          streamOpt.map { case SizedStream(stream, len) =>
            val newStream =
              stream.map(TypedTriplet(eltType, _))
                    .longScan(mb, zerot)(scanBody)
                    .map(_.untyped)
            val newLen = len.map { case (s, l) => (Code(s, l := l + 1), l)}
            SizedStream(newStream, newLen)
          }

        case x@RunAggScan(array, name, init, seqs, result, _) =>
          val aggs = x.physicalSignatures
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromFunctionBuilder(aggs, fb, "array_agg_scan")

          val eltType = coerce[PStream](array.pType).elementType
          implicit val eltPack = TypedTriplet.pack(eltType)

          val xElt = eltPack.newFields(fb, "aggscan_elt")

          val bodyEnv = Emit.bindEnv(env, name -> xElt)
          val cInit = emitIR(init, container = Some(newContainer))
          val seqPerElt = emitIR(seqs, env = bodyEnv, container = Some(newContainer))
          val postt = emitIR(result, env = bodyEnv, container = Some(newContainer))

          val optStream = emitStream(array, env)

          optStream.map { case SizedStream(stream, len) =>
            val newStream = stream.map[EmitCode](
              { eltt =>
                EmitCode(
                  Code(
                    xElt := TypedTriplet(eltType, eltt),
                    postt.setup,
                    seqPerElt.setup),
                  postt.m,
                  postt.pv)
              },
              setup0 = Some(Code(xElt.init, aggSetup)),
              close0 = Some(aggCleanup),
              setup = Some(cInit.setup))

            SizedStream(newStream, len)
          }

        case StreamLeftJoinDistinct(leftIR, rightIR, leftName, rightName, compIR, joinIR) =>
          val lEltType = coerce[PStream](leftIR.pType).elementType
          val rEltType = coerce[PStream](rightIR.pType).elementType.setRequired(false)
          implicit val lEltPack = TypedTriplet.pack(lEltType)
          implicit val rEltPack = TypedTriplet.pack(rEltType)
          val xLElt = lEltPack.newFields(fb, "join_lelt")
          val xRElt = rEltPack.newFields(fb, "join_relt")

          val env2 = Emit.bindEnv(env, leftName -> xLElt, rightName -> xRElt)

          def compare(lelt: TypedTriplet[lEltType.type], relt: TypedTriplet[rEltType.type]): Code[Int] = {
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
                leftStream.map(TypedTriplet(lEltType, _)),
                rightStream.map(TypedTriplet(rEltType, _)),
                TypedTriplet.missing(rEltType),
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
