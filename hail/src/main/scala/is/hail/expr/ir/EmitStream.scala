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
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = missing.mux(none, some(value))
  }

  // None is the only COption allowed to not call `some` at compile time
  object None extends COption[Nothing] {
    def apply(none: Code[Ctrl], some: Nothing => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = none
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

  def fromEmitTriplet[A](et: EmitTriplet): COption[Code[A]] = new COption[Code[A]] {
    def apply(none: Code[Ctrl], some: Code[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      Code(et.setup, et.m.mux(none, some(coerce[A](et.v))))
    }
  }

  def fromTypedTriplet(et: EmitTriplet): COption[Code[_]] = fromEmitTriplet(et)

  def toEmitTriplet(opt: COption[Code[_]], t: PType, mb: MethodBuilder): EmitTriplet = {
    val ti: TypeInfo[_] = typeToTypeInfo(t)
    val m = mb.newLocal[Boolean]
    val v = mb.newLocal(ti)
    val setup = JoinPoint.CallCC[Unit] { (jb, ret) =>
      opt(Code(m := true, v.storeAny(defaultValue(ti)), ret(())),
          a => Code(m := false, v.storeAny(a), ret(())))(EmitStreamContext(mb, jb))
    }
    EmitTriplet(setup, m, PValue(t, v.load()))
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
        setup0 = Code(inInnerStream := false, outerSource.setup0, innerSource.setup0),
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
        pull = JoinPoint.mux(b.load, lPullJP, rPullJP)
      )
    }
  }

  def fromParameterized[P, A](
    stream: EmitStream.Parameterized[P, A]
  ): P => COption[Stream[A]] = p =>
    if (stream == EmitStream.missing)
      COption.None
    else {
      new COption[Stream[A]] {
        def apply(none: Code[Ctrl], some: Stream[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
          import EmitStream.{Missing, Start, EOS, Yield}
          implicit val sP = stream.stateP
          val s = newLocal[stream.S]
          val sNew = newLocal[stream.S]

          def src(s0: stream.S): Stream[A] = new Stream[A] {
            def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
              Source[A](
                setup0 = Code(s.init, sNew.init),
                close0 = Code._empty,
                setup = sNew := s0,
                close = Code._empty,
                pull = Code(s := sNew.load, stream.step(s.load) {
                  case EOS => eos
                  case Yield(elt, s1) => Code(sNew := s1, push(elt))
                }))
            }
          }

          stream.init(p) {
            case Missing => none
            case Start(s0) => some(src(s0))
          }
        }
      }
    }
}

object EmitStream2 {
  import CodeStream._

  def write(mb: MethodBuilder, stream: Stream[EmitTriplet], ab: StagedArrayBuilder): Code[Unit] =
    Code(
      ab.clear,
      stream.forEach(mb) { et =>
        Code(et.setup, et.m.mux(ab.addMissing(), ab.add(et.v)))
      })

  def toArray(mb: MethodBuilder, aTyp: PArray, optStream: COption[Stream[EmitTriplet]]): EmitTriplet = {
    // FIXME: add fast path when stream length is known
    val srvb = new StagedRegionValueBuilder(mb, aTyp)
    val len = mb.newLocal[Int]
    val i = mb.newLocal[Int]
    val vab = new StagedArrayBuilder(aTyp.elementType, mb, 16)
    val result = optStream.map { stream =>
      Code(
        write(mb, stream, vab),
        len := vab.size,
        srvb.start(len, init = true),
        i := 0,
        Code.whileLoop(i < len,
                       vab.isMissing(i).mux(
                         srvb.setMissing(),
                         srvb.addIRIntermediate(aTyp.elementType)(vab(i))),
                       i := i + 1,
                       srvb.advance()),
        srvb.offset)
    }

    COption.toEmitTriplet(result, aTyp, mb)
  }

  private[ir] def apply(
    emitter: Emit,
    streamIR0: IR,
    env0: Emit.E,
    er: EmitRegion,
    container: Option[AggContainer]
  ): COption[Stream[EmitTriplet]] = {
    assert(emitter.mb eq er.mb)
    val mb = emitter.mb
    val fb = mb.fb

    def emitStream(streamIR: IR, env: Emit.E): COption[Stream[EmitTriplet]] = {

      def emitIR(ir: IR, env: Emit.E = env, container: Option[AggContainer] = container): EmitTriplet =
        emitter.emit(ir, env, er, container)

      streamIR match {

        case StreamMap(childIR, name, bodyIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType
          implicit val childEltPack = TypedTriplet.pack(childEltType)

          val optStream = emitStream(childIR, env)
          optStream.map { stream =>
            stream.map { eltt =>
              val xElt = childEltPack.newFields(mb.fb, name)
              val bodyenv = Emit.bindEnv(env, name -> xElt)
              val bodyt = emitIR(bodyIR, bodyenv)

              EmitTriplet(
                Code(xElt := TypedTriplet(childEltType, eltt),
                     bodyt.setup),
                bodyt.m,
                bodyt.pv)
            }
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType
          implicit val childEltPack = TypedTriplet.pack(childEltType)

          val optStream = emitStream(childIR, env)

          optStream.map { stream =>
            filter(stream
              .map { elt =>
                val xElt = childEltPack.newFields(mb.fb, name)
                val condEnv = Emit.bindEnv(env, name -> xElt)
                val cond = emitIR(condIR, condEnv)

                new COption[EmitTriplet] {
                  def apply(none: Code[Ctrl], some: EmitTriplet => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
                    Code(
                      xElt := TypedTriplet(childEltType, elt),
                      cond.setup,
                      (cond.m || !cond.value[Boolean]).mux(
                        none,
                        some(EmitTriplet(Code._empty, xElt.load.m, xElt.load.pv))
                      )
                    )
                  }
                }
              })
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

          optStreams.map { streams =>
            behavior match {

              case ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength =>
                multiZip(streams)
                  .map { elts =>
                    val bodyEnv = Emit.bindEnv(env, names.zip(eltVars.pss.asInstanceOf[IndexedSeq[ParameterStoreTriplet[_]]]): _*)
                    val body = emitIR(bodyIR, bodyEnv)
                    val typedElts = eltTypes.zip(elts).map { case (t, v) => TypedTriplet(t, v) }
                    EmitTriplet(Code(eltVars := typedElts, body.setup), body.m, body.pv)
                  }

              case ArrayZipBehavior.ExtendNA | ArrayZipBehavior.AssertSameLength =>
                // extend to infinite streams, where the COption becomes missing after EOS
                val extended: IndexedSeq[Stream[COption[TypedTriplet[_]]]] =
                  streams.zipWithIndex.map { case (stream, i) =>
                    val t = eltTypes(i)

                    extendNA[TypedTriplet[_]](stream.map(TypedTriplet(t, _)))(eltsPack.pps(i).asInstanceOf[ParameterPack[TypedTriplet[_]]])
                  }

                // zip to an infinite stream, where the COption is missing when all streams are EOS
                val flagged: Stream[COption[EmitTriplet]] = multiZip(extended)
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
                    val body = emitIR(bodyIR, bodyEnv)

                    Code(
                      allEOS := true,
                      if (assert) anyEOS := false else Code._empty,
                      eltVars := checkedElts,
                      if (assert)
                        (anyEOS & !allEOS).mux(
                          Code._fatal("zip: length mismatch"),
                          k(COption(allEOS, body)))
                      else
                        k(COption(allEOS, body)))
                  }

                // termininate the stream when all streams are EOS
                take(flagged)
            }
          }

        case StreamFlatMap(outerIR, name, innerIR) =>
          val outerEltType = coerce[PStream](outerIR.pType).elementType
          val outerEltPack = TypedTriplet.pack(outerEltType)

          val optOuter = emitStream(outerIR, env)

          optOuter.map { outer =>
            val nested = outer.mapCPS[COption[Stream[EmitTriplet]]] { (ctx, elt, k) =>
              val xElt = outerEltPack.newFields(ctx.mb.fb, name)
              val innerEnv = Emit.bindEnv(env, name -> xElt)
              val optInner = emitStream(innerIR, innerEnv)

              Code(
                xElt := TypedTriplet(outerEltType, elt),
                k(optInner))
            }

            flatMap(filter(nested))
          }

        case If(condIR, thn, els) =>
          val eltType = coerce[PStream](thn.pType).elementType
          implicit val eltPack: ParameterPack[TypedTriplet[eltType.type]] = TypedTriplet.pack(eltType)
          val xCond = mb.newField[Boolean]

          val condT = COption.fromEmitTriplet[Boolean](emitIR(condIR))
          val optLeftStream = emitStream(thn, env)
          val optRightStream = emitStream(els, env)

          // TODO: set xCond in setup of choose, don't need CPS
          condT.flatMapCPS[Stream[EmitTriplet]] { (cond, _, k) =>
            Code(
              xCond := cond,
              k(COption.choose[Stream[EmitTriplet]](
                xCond,
                optLeftStream,
                optRightStream,
                (leftStream, rightStream) =>
                  mux(xCond, leftStream.map(TypedTriplet(eltType, _)), rightStream.map(TypedTriplet(eltType, _))).map(_.untyped)))
              )
          }

        case Let(name, valueIR, bodyIR) =>
          val valueType = valueIR.pType
          val valuePack = TypedTriplet.pack(valueType)
          val xValue = valuePack.newFields(mb.fb, name)

          val valuet = TypedTriplet(valueType, emitIR(valueIR))
          val bodyEnv = Emit.bindEnv(env, name -> xValue)

          emitStream(bodyIR, bodyEnv).addSetup(xValue := valuet)

        case StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = zeroIR.pType
          val eltPack = TypedTriplet.pack(eltType)
          implicit val accPack = TypedTriplet.pack(accType)

          def scanBody(elt: TypedTriplet[eltType.type], acc: TypedTriplet[accType.type]): TypedTriplet[accType.type] = {
            val xElt = eltPack.newFields(fb, eltName)
            val xAcc = accPack.newFields(fb, accName)
            val bodyEnv = Emit.bindEnv(env, accName -> xAcc, eltName -> xElt)

            val bodyT = TypedTriplet(accType, emitIR(bodyIR, bodyEnv))
            TypedTriplet(accType, EmitTriplet(Code(xElt := elt, xAcc := acc, bodyT.setup), bodyT.m, bodyT.pv))
          }

          val zerot = TypedTriplet(accType, emitIR(zeroIR))
          val streamOpt = emitStream(childIR, env)
          streamOpt.map { stream =>
            stream.map(TypedTriplet(eltType, _))
                  .longScan(mb, zerot)(scanBody)
                  .map(_.untyped)
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
          optStream.map { stream =>
            stream.map[EmitTriplet](
              { eltt =>
                EmitTriplet(
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
          }

        case StreamLeftJoinDistinct(leftIR, rightIR, leftName, rightName, compIR, joinIR) =>
          val lEltType = coerce[PStream](leftIR.pType).elementType
          val rEltType = coerce[PStream](rightIR.pType).elementType
          implicit val lEltPack = TypedTriplet.pack(lEltType)
          implicit val rEltPack = TypedTriplet.pack(rEltType)
          val xLElt = lEltPack.newFields(fb, "join_lelt")
          val xRElt = rEltPack.newFields(fb, "join_relt")

          val env2 = Emit.bindEnv(env, leftName -> xLElt, rightName -> xRElt)

          def compare(lelt: TypedTriplet[lEltType.type], relt: TypedTriplet[rEltType.type]): Code[Int] = {
            val compt = emitIR(compIR, env2)
            Code(
              xLElt := lelt,
              xRElt := relt,
              compt.setup,
              compt.m.orEmpty(Code._fatal("StreamLeftJoinDistinct: comp can't be missing")),
              coerce[Int](compt.v))
          }

          emitStream(leftIR, env).flatMap { leftStream =>
            emitStream(rightIR, env).map { rightStream =>
              leftJoinRightDistinct(
                leftStream.map(TypedTriplet(lEltType, _)),
                rightStream.map(TypedTriplet(rEltType, _)),
                TypedTriplet.missing(rEltType),
                compare)
                .map { case (lelt, relt) =>
                  val joint = emitIR(joinIR, env2)
                  EmitTriplet(
                    Code(xLElt := lelt, xRElt := relt, joint.setup),
                    joint.m,
                    joint.pv)
                }
            }
          }

        case _ =>
          val EmitStream(parameterized, eltType) =
            EmitStream.apply(emitter, streamIR, env, er, container)
          fromParameterized(parameterized)(())
      }
    }

    emitStream(streamIR0, env0)
  }
}


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

    def init(param: P)(
      k: Init[S] => Code[Ctrl]
    )(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val missing = ctx.jb.joinPoint()
      missing.define { _ => k(Missing) }

      def loop(i: Int, ab: ArrayBuilder[Any]): Code[Ctrl] = {
        if (i == streams.length)
          k(Start(ab.result(): IndexedSeq[_]))
        else
          streams(i).init(param) {
            case Missing => missing(())
            case Start(s) =>
              ab += s
              loop(i + 1, ab)
          }
      }

      loop(0, new ArrayBuilder)
    }

    def step(state: IndexedSeq[_])(k: Step[EmitTriplet, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val eos = ctx.jb.joinPoint()
      eos.define(_ => k(EOS))
      behavior match {
        case ArrayZipBehavior.AssertSameLength =>
          val anyEOS = ctx.mb.newLocal[Boolean]
          val allEOS = ctx.mb.newLocal[Boolean]
          val labels = (0 to streams.size).map(_ => ctx.jb.joinPoint())

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
              labels(i).define(_ => streamI.step(state(i).asInstanceOf[streamI.S]) {
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
              streamI.step(state(i).asInstanceOf[streamI.S]) {
                case Yield(elt, s) =>
                  ab += ((elt, s))
                  loop(i + 1, ab)
                case EOS => eos(())
              }
            }
          }

          loop(0, new ArrayBuilder)

        case ArrayZipBehavior.ExtendNA =>
          val allEOS = ctx.mb.newLocal[Boolean]
          val missing = streams.map(_ => ctx.mb.newLocal[Boolean])
          val labels = (0 to streams.size).map(_ => ctx.jb.joinPoint())

          val ab = new ArrayBuilder[(PValue, Any)]
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
              labels(i).define(_ => streamI.step(state(i).asInstanceOf[streamI.S]) {
                case EOS =>
                  Code(missing(i) := true, labels(i + 1)(()))
                case Yield(elt, s) =>
                  ab += ((elt.pv, s))
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

    def init(param: P)(
      k: Init[S] => Code[Ctrl]
    )(implicit ctx: EmitStreamContext
    ): Code[Ctrl]

    def step(state: S)(k: Step[A, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl]

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
      def init(param: Q)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
        val missing = ctx.jb.joinPoint()
        missing.define { _ => k(Missing) }
        f(param, {
          case Some(newParam) => self.init(newParam) {
            case Missing => missing(())
            case Start(s) => k(Start(s))
          }
          case None => missing(())
        })
      }
      def step(state: S)(k: Step[A, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        self.step(state) {
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
      def init(param: P)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        self.init(param) {
          case Missing => k(Missing)
          case Start(s) => k(Start(s))
        }
      def step(state: S)(k: Step[B, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        self.step(state) {
          case EOS => k(EOS)
          case Yield(a, s) => f(a, b => k(Yield(b, s)))
        }
    }

    def filterMap[B](f: (A, Option[B] => Code[Ctrl]) => Code[Ctrl]): Parameterized[P, B] = new Parameterized[P, B] {
      type S = self.S
      implicit val stateP: ParameterPack[S] = self.stateP
      def emptyState: S = self.emptyState
      def length(s0: S): Option[Code[Int]] = None
      def init(param: P)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        self.init(param)(k)
      def step(s0: S)(k: Step[B, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
        val pull = ctx.jb.joinPoint[S](ctx.mb)
        pull.define(self.step(_) {
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

      def init(param: P)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        self.init(param) {
          case Missing => k(Missing)
          case Start(s0) => k(Start((s0, zero, true)))
        }

      def step(state: S)(k: Step[B, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
        val yield_ = ctx.jb.joinPoint[(B, self.S)](ctx.mb)
        yield_.define { case (b, s1) => k(Yield(b, (s1, b, false))) }
        val (s, b, isFirstStep) = state
        isFirstStep.mux(
          yield_((b, s)),
          self.step(s) {
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
    def init(param: Any)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      k(Missing)
    def step(s: S)(k: Step[Nothing, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      k(EOS)
  }

  def read[P](dec: Code[InputBuffer] => Code[Long]): Parameterized[Code[InputBuffer], Code[Long]] = new Parameterized[Code[InputBuffer], Code[Long]] {
    type S = Code[InputBuffer]
    val stateP: ParameterPack[S] = implicitly

    def emptyState: S = Code._null

    def length(s0: S): Option[Code[Int]] = None

    def init(buf: Code[InputBuffer])(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      k(Start(buf))

    def step(state: S)(k: Step[Code[Long], S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
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

    def init(len: Code[Int])(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      k(Start((len, start)))

    def step(state: S)(k: Step[Code[Int], S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val (nLeft, acc) = state
      stepIf(k, nLeft > 0, acc, (nLeft - 1, acc + incr))
    }
  }

  def sequence[A: ParameterPack](elements: Seq[A]): Parameterized[Any, A] = new Parameterized[Any, A] {
    type S = Code[Int]
    val stateP: ParameterPack[S] = implicitly
    def emptyState: S = elements.length
    def length(s0: S): Option[Code[Int]] = Some(const(elements.length) - s0)

    def init(param: Any)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      k(Start(0))

    def step(idx: S)(k: Step[A, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val eos = ctx.jb.joinPoint()
      val yld = ctx.jb.joinPoint[A](ctx.mb)
      eos.define { _ => k(EOS) }
      yld.define { a => k(Yield(a, idx + 1)) }
      JoinPoint.switch(idx, eos, elements.map { elt =>
        val j = ctx.jb.joinPoint()
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

    def init(param: A)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      outer.init(param) {
        case Missing => k(Missing)
        case Start(outS0) => k(Start((outS0, inner.emptyState)))
      }

    def step(state: S)(k: Step[C, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val stepInner = ctx.jb.joinPoint[S](ctx.mb)
      val stepOuter = ctx.jb.joinPoint[outer.S](ctx.mb)
      stepInner.define { case (outS, innS) =>
        inner.step(innS) {
          case EOS => stepOuter(outS)
          case Yield(innElt, innS1) => k(Yield(innElt, (outS, innS1)))
        }
      }
      stepOuter.define(outer.step(_) {
        case EOS => k(EOS)
        case Yield(outElt, outS) => inner.init(outElt) {
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

    def init(param: P)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val missing = ctx.jb.joinPoint()
      missing.define { _ => k(Missing) }
      left.init(param) {
        case Missing => missing(())
        case Start(lS) => right.init(param) {
          case Missing => missing(())
          case Start(rS) => k(Start((lS, rS, (rNil, false))))
        }
      }
    }

    def step(state: S)(k: Step[(A, B), S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val (lS0, rS0, (rPrev, somePrev)) = state
      left.step(lS0) {
        case EOS => k(EOS)
        case Yield(lElt, lS) =>
          val push = ctx.jb.joinPoint[(B, right.S, (B, Code[Boolean]))](ctx.mb)
          val pull = ctx.jb.joinPoint[right.S](ctx.mb)
          val compare = ctx.jb.joinPoint[(B, right.S)](ctx.mb)
          push.define { case (rElt, rS, rPrevOpt) =>
            k(Yield((lElt, rElt), (lS, rS, rPrevOpt)))
          }
          pull.define(right.step(_) {
            case EOS => push((rNil, right.emptyState, (rNil, false)))
            case Yield(rElt, rS) => compare((rElt, rS))
          })
          compare.define { case (rElt, rS) =>
            ParameterPack.let(ctx.mb, comp(lElt, rElt)) { c =>
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

      def init(iter: S)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        k(Start(iter))

      def step(iter: S)(k: Step[Code[A], S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
        iter.hasNext.mux(
          ParameterPack.let(ctx.mb, iter.next()) { elt => k(Yield(elt, iter)) },
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

    def init(param: P)(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val missing = ctx.jb.joinPoint()
      val start = ctx.jb.joinPoint[S](ctx.mb)
      missing.define { _ => k(Missing) }
      start.define { s => k(Start(s)) }
      cond.mux(
        left.init(param) {
          case Start(s0) => start((true, s0, right.emptyState))
          case Missing => missing(())
        },
        right.init(param) {
          case Start(s0) => start((false, left.emptyState, s0))
          case Missing => missing(())
        })
    }

    def step(state: S)(k: Step[A, S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val (useLeft, lS, rS) = state
      val eos = ctx.jb.joinPoint()
      val push = ctx.jb.joinPoint[(A, left.S, right.S)](ctx.mb)
      eos.define { _ => k(EOS) }
      push.define { case (elt, lS, rS) => k(Yield(elt, (useLeft, lS, rS))) }
      useLeft.mux(
        left.step(lS) {
          case Yield(a, lS1) => push((a, lS1, rS))
          case EOS => eos(())
        },
        right.step(rS) {
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

    def init(ib: Code[InputBuffer])(k: Init[S] => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      k(Start(ib))

    def step(ib: Code[InputBuffer])(
      k: Step[Code[T], S] => Code[Ctrl]
    )(implicit ctx: EmitStreamContext): Code[Ctrl] =
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
    def present(pt: PType, v: Code[_]): EmitTriplet = EmitTriplet(Code._empty, false, PValue(pt, v))

    def emitIR(ir: IR, env: Emit.E): EmitTriplet =
      emitter.emit(ir, env, er, container)

    def emitStream(streamIR: IR, env: Emit.E): Parameterized[Any, EmitTriplet] = {
      streamIR match {
        case NA(_) =>
          missing

        case In(i, t@PStream(eltPType, _)) =>
          val EmitTriplet(_, m, v) = emitter.normalArgument(i, t)
          fromIterator[RegionValue]
            .map { (rv: Code[RegionValue]) =>
              present(eltPType, Region.loadIRIntermediate(eltPType)(rv.invoke[Long]("getOffset")))
            }
            .guardParam { (_, k) =>
              m.mux(k(None), k(Some(coerce[Iterator[RegionValue]](v.code))))
            }

        case ReadPartition(path, spec, requestedType) =>
          val p = emitIR(path, env)
          val pathString = path.pType.asInstanceOf[PString].loadString(p.value)

          val (pt, dec) = spec.buildEmitDecoderF(requestedType, fb)

          read(dec(er.region, _)).map(present(pt, _)).guardParam { (_, k) =>
            val rowBuf = spec.buildCodeInputBuffer(fb.getUnsafeReader(pathString, true))
            Code(p.setup, p.m.mux(k(None), k(Some(rowBuf))))
          }

        case x@MakeStream(elements, t) =>
          val e = coerce[PStream](x.pType).elementType
          implicit val eP = TypedTriplet.pack(e)
          sequence(elements.map {
            ir => TypedTriplet(e, {
              val et = emitIR(ir, env)
              EmitTriplet(et.setup, et.m, e.copyFromPValue(er.mb, er.region, et.pv))
            })
          }).map(_.untyped)

        case x@StreamRange(startIR, stopIR, stepIR) =>
          val step = fb.newField[Int]("sr_step")
          val start = fb.newField[Int]("sr_start")
          val stop = fb.newField[Int]("sr_stop")
          val llen = fb.newField[Long]("sr_llen")

          range(start, step)
            .map(present(x.pType.asInstanceOf[PStream].elementType, _))
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
                PValue(eltPType, Region.loadIRIntermediate(eltPType)(pType.elementOffset(aoff, i))))
            }
            .guardParam { (_, k) =>
              val arrt = emitIR(containerIR, env)
              val len = pType.loadLength(aoff)
              Code(arrt.setup,
                arrt.m.mux(
                  k(None),
                  Code(aoff := arrt.value, k(Some(len)))))
            }

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }
    }

    EmitStream(
      emitStream(streamIR0, env0),
      streamIR0.pType.asInstanceOf[PStream].elementType)
  }
}

case class EmitStream(
  stream: EmitStream.Parameterized[Any, EmitTriplet],
  elementType: PType)
