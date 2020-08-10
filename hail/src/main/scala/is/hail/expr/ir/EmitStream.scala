package is.hail.expr.ir

import is.hail.io._
import is.hail.services.shuffler._
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.types.virtual._
import is.hail.types.physical._
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual.TStream
import is.hail.utils._
import java.io.{DataOutputStream, InputStream, OutputStream}
import java.net.Socket
import java.util.Base64

import org.apache.log4j.Logger

import scala.language.{existentials, higherKinds}

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

  def none[A](dummy: A): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val _ = some(dummy)
      none
    }
  }

  def present[A](value: A): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      some(value)
  }

  def lift[A](opts: IndexedSeq[COption[A]]): COption[IndexedSeq[A]] =
    if (opts.length == 0)
      COption.present(FastIndexedSeq())
    else if (opts.length == 1)
      opts.head.map(a => IndexedSeq(a))
    else
      new COption[IndexedSeq[A]] {
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
  def choose[A](useLeft: Code[Boolean], left: COption[A], right: COption[A], fuse: (A, A) => A): COption[A] = new COption[A] {
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

  def fromEmitCode(et: EmitCode): COption[PCode] = new COption[PCode] {
    def apply(none: Code[Ctrl], some: PCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      Code(et.setup, et.m.mux(none, some(et.pv)))
    }
  }

  def toEmitCode(opt: COption[PCode], mb: EmitMethodBuilder[_]): EmitCode = {
    implicit val ctx = EmitStreamContext(mb)
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    var value: PCode = null
    val setup = opt(Lmissing.goto, pc => { value = pc; Lpresent.goto })

    assert(value != null)
    EmitCode(
      Code._empty,
      new CCode(setup.start, Lmissing.start, Lpresent.start),
      value)
  }
}

abstract class Stream[+A] { self =>
  import Stream._

  def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A]

  def fold(mb: EmitMethodBuilder[_], init: => Code[Unit], f: (A) => Code[Unit], ret: => Code[Ctrl]): Code[Ctrl] = {
    implicit val ctx = EmitStreamContext(mb)
    val Ltop = CodeLabel()
    val Lafter = CodeLabel()
    val s = self(Lafter.goto, (a) => Code(f(a), Ltop.goto: Code[Ctrl]))
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

  def forEachCPS(mb: EmitMethodBuilder[_], f: (A, Code[Ctrl]) => Code[Ctrl]): Code[Unit] =
    mapCPS[Unit]((_, a, k) => f(a, k(()))).run(mb)

  def forEach(mb: EmitMethodBuilder[_], f: A => Code[Unit]): Code[Unit] =
    mapCPS[Unit]((_, a, k) => Code(f(a), k(()))).run(mb)

  def forEachI(cb: EmitCodeBuilder, f: A => Unit): Unit = {
    val savedCode = cb.code
    cb.code = Code._empty
    val streamCode = forEach(cb.emb, a => { f(a); cb.code })
    cb.code = Code(savedCode, streamCode)
  }

  def run(mb: EmitMethodBuilder[_]): Code[Unit] = {
    implicit val ctx = EmitStreamContext(mb)
    val Leos = CodeLabel()
    val Lpull = CodeLabel()
    val source = self(eos = Leos.goto, push = _ => Lpull.goto)
    Code(
      source.setup0,
      source.setup,
      // fall through
      Lpull, source.pull,
      Leos, source.close, source.close0
      // fall off
      )
  }

  def mapCPS[B](
    f: (EmitStreamContext, A, B => Code[Ctrl]) => Code[Ctrl],
    setup0: Option[Code[Unit]] = None,
    setup:  Option[Code[Unit]] = None,
    close0: Option[Code[Unit]] = None,
    close:  Option[Code[Unit]] = None
  ): Stream[B] = new Stream[B] {
    def apply(eos: Code[Ctrl], push: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[B] = {
      val source = self(
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

  def map[B](
    f: A => B,
    setup0: Option[Code[Unit]] = None,
    setup:  Option[Code[Unit]] = None,
    close0: Option[Code[Unit]] = None,
    close:  Option[Code[Unit]] = None
  ): Stream[B] = mapCPS((_, a, k) => k(f(a)), setup0, setup, close0, close)

  def addSetup(setup: Code[Unit]) = map(x => x, setup = Some(setup))

  def flatMap[B](f: A => Stream[B]): Stream[B] =
    map(f).flatten
}

object Stream {
  case class Source[+A](setup0: Code[Unit], close0: Code[Unit], setup: Code[Unit], close: Code[Unit], pull: Code[Ctrl])

  def empty[A](dummy: A): Stream[A] = new Stream[A] {
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      val _ = push(dummy)
      Source[A](
        setup0 = Code._empty,
        close0 = Code._empty,
        setup = Code._empty,
        close = Code._empty,
        pull = eos)
    }
  }

  def iota(mb: EmitMethodBuilder[_], start: Code[Int], step: Code[Int]): Stream[Code[Int]] = {
    val lstep = mb.genFieldThisRef[Int]("sr_lstep")
    val cur = mb.genFieldThisRef[Int]("sr_cur")

    unfold[Code[Int]](
      f = {
        case (_ctx, k) =>
          implicit val ctx = _ctx
          Code(cur := cur + lstep, k(COption.present(cur)))
      },
      setup = Some(Code(lstep := step, cur := start - lstep)))
  }

  def iotaL(mb: EmitMethodBuilder[_], start: Code[Long], step: Code[Int]): Stream[Code[Long]] = {
    val lstep = mb.genFieldThisRef[Int]("sr_lstep")
    val cur = mb.genFieldThisRef[Long]("sr_cur")

    unfold[Code[Long]](
      f = {
        case (_ctx, k) =>
          implicit val ctx = _ctx
          Code(cur := cur + lstep.toL, k(COption.present(cur)))
      },
      setup = Some(Code(lstep := step, cur := start - lstep.toL)))
  }

  def range(mb: EmitMethodBuilder[_], start: Code[Int], step: Code[Int], len: Code[Int]): Stream[Code[Int]] =
    zip(iota(mb, start, step),
        iota(mb, len, -1))
      .map[COption[Code[Int]]] { case (cur, rem) =>
        COption(rem <= 0, cur)
      }
      .take

  def unfold[A](
    f: (EmitStreamContext, COption[A] => Code[Ctrl]) => Code[Ctrl],
    setup0: Option[Code[Unit]] = None,
    setup:  Option[Code[Unit]] = None,
    close0: Option[Code[Unit]] = None,
    close:  Option[Code[Unit]] = None
  ): Stream[A] = new Stream[A] {
    assert(f != null)
    def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
      Source[A](
        setup0 = setup0.getOrElse(Code._empty),
        close0 = close0.getOrElse(Code._empty),
        setup = setup.getOrElse(Code._empty),
        close = close.getOrElse(Code._empty),
        pull = f(ctx, _.apply(
          none = eos,
          some = a => push(a))))
    }
  }

  def grouped[A](mb: EmitMethodBuilder[_], childStream: StagedRegion => Stream[A], size: Code[Int], backupRegion: StagedRegion): Stream[StagedRegion => Stream[A]] = new Stream[StagedRegion => Stream[A]] {
    def apply(outerEos: Code[Ctrl], outerPush: (StagedRegion => Stream[A]) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[StagedRegion => Stream[A]] = {
      val xCounter = ctx.mb.genFieldThisRef[Int]("st_grp_ctr")
      val xInOuter = ctx.mb.genFieldThisRef[Boolean]("st_grp_io")
      val xSize = ctx.mb.genFieldThisRef[Int]("st_grp_sz")
      val LchildPull = CodeLabel()
      val LouterPush = CodeLabel()

      var childEltRegion = backupRegion.createDummyChildRegion

      var childSource: Source[A] = null
      val inner = (innerEltRegion: StagedRegion) => new Stream[A] {
        def apply(innerEos: Code[Ctrl], innerPush: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
          val LinnerEos = CodeLabel()
          val LinnerPush = CodeLabel()

          // Need to be able to free the memory used by a child stream element
          // when the outer stream advances before all inner stream elements
          // are consumed.
          childEltRegion = innerEltRegion.createChildRegion(mb)

          childSource = childStream(childEltRegion)(
            xInOuter.mux(outerEos, LinnerEos.goto),
            { a =>
              Code(LinnerPush, childEltRegion.giveToParent(), innerPush(a))

              Code(
                // xCounter takes values in [1, xSize + 1]
                xCounter := xCounter + 1,
                // !xInOuter iff this element was requested by an inner stream.
                // Else we are stepping to the beginning of the next group.
                xInOuter.mux(
                  (xCounter > xSize).mux(
                    // first of a group
                    Code(xCounter := 1, LouterPush.goto),
                    Code(childEltRegion.clear(), LchildPull.goto)),
                  LinnerPush.goto))
            })

          Code(LinnerEos, innerEos)

          Source[A](
            setup0 = Code._empty,
            close0 = Code._empty,
            setup = Code._empty,
            close = Code._empty,
            pull = xInOuter.mux(
              // xInOuter iff this is the first pull from inner stream,
              // in which case the element has already been produced
              Code(
                xInOuter := false,
                xCounter.cne(1).orEmpty(Code._fatal[Unit](const("expected counter = 1"))),
                LinnerPush.goto),
              (xCounter < xSize).mux(
                LchildPull.goto,
                LinnerEos.goto)))
        }
      }

      Code(LouterPush, outerPush(inner))

      if (childSource == null) {
        // inner stream is unused
        val Lunreachable = CodeLabel()
        Code(Lunreachable, Code._fatal[Unit]("unreachable"))
        // because LinnerPush is never executed, nothing is ever given to backupRegion;
        // childEltRegion is cleared every element.
        val unusedInnerSource = inner(childEltRegion)(Lunreachable.goto, _ => Lunreachable.goto)
      }

      Code(LchildPull, childSource.pull)

      Source[StagedRegion => Stream[A]](
        setup0 = Code(childSource.setup0, childEltRegion.allocateRegion(Region.REGULAR)),
        close0 = Code(childEltRegion.allocateRegion(Region.REGULAR), childSource.close0),
        setup = Code(
          childSource.setup,
          xSize := size,
          xCounter := xSize),
        close = childSource.close,
        pull = Code(xInOuter := true, LchildPull.goto))
    }
  }

  implicit class StreamStream[A](val outer: Stream[Stream[A]]) extends AnyVal {
    def flatten: Stream[A] = new Stream[A] {
      def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
        val closing = ctx.mb.genFieldThisRef[Boolean]("sfm_closing")
        val LouterPull = CodeLabel()
        var innerSource: Source[A] = null
        val LinnerPull = CodeLabel()
        val LinnerEos = CodeLabel()
        val LcloseOuter = CodeLabel()
        val inInnerStream = ctx.mb.genFieldThisRef[Boolean]("sfm_in_innner")
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
  }

  implicit class StreamCOpt[A](val stream: Stream[COption[A]]) extends AnyVal {
    def flatten: Stream[A] = new Stream[A] {
      def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
        val Lpull = CodeLabel()
        val source = stream(
          eos = eos,
          push = _.apply(none = Lpull.goto, some = push))
        source.copy(pull = Code(Lpull, source.pull))
      }
    }

    def take: Stream[A] = new Stream[A] {
      def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
        val Leos = CodeLabel()
        stream(
          eos = Code(Leos, eos),
          push = _.apply(none = Leos.goto, some = push)).asInstanceOf[Source[A]]
      }
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
    comp: (EmitValue, EmitValue) => Code[Int]
  ): Stream[(EmitCode, EmitCode)] = new Stream[(EmitCode, EmitCode)] {
    def apply(eos: Code[Ctrl], push: ((EmitCode, EmitCode)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(EmitCode, EmitCode)] = {
      val pulledRight = mb.genFieldThisRef[Boolean]()
      val rightEOS = mb.genFieldThisRef[Boolean]()
      val lx = mb.newEmitField(lElemType) // last value received from left
      val rx = mb.newEmitField(rElemType) // last value received from right
      val rxOut = mb.newEmitField(rElemType.setRequired(false)) // right value to push (may be missing while rx is not)

      var rightSource: Source[EmitCode] = null
      val leftSource = left(
        eos = eos,
        push = a => {
          val Lpush = CodeLabel()
          val LpullRight = CodeLabel()
          val Lcompare = CodeLabel()

          val compareCode = Code(Lcompare, {
            val c = mb.genFieldThisRef[Int]()
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

  def merge(
    mb: EmitMethodBuilder[_],
    lElemType: PType, mkLeft: StagedRegion => Stream[EmitCode],
    rElemType: PType, mkRight: StagedRegion => Stream[EmitCode],
    outElemType: PType, destRegion: StagedRegion,
    comp: (EmitValue, EmitValue) => Code[Int]
  ): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val pulledRight = mb.genFieldThisRef[Boolean]()
      val rightEOS = mb.genFieldThisRef[Boolean]()
      val leftEOS = mb.genFieldThisRef[Boolean]()
      val lx = mb.newEmitField(lElemType) // last value received from left
      val rx = mb.newEmitField(rElemType) // last value received from right
      val leftRegion = destRegion.createChildRegion(mb)
      val rightRegion = destRegion.createChildRegion(mb)
      val left = mkLeft(leftRegion)
      val right = mkRight(rightRegion)
      val outx = mb.newEmitField(outElemType) // value to push
      val c = mb.genFieldThisRef[Int]()

      // Invariants:
      // * 'pulledRight' is false until after the first pull from 'left',
      //   before the first pull from 'right'
      // * 'rightEOS'/'leftEOS' are true iff 'left'/'right' have reached EOS
      // * 'lx'/'rx' contain the most recent value received from 'left'/'right',
      //   and are uninitialized before the first pull
      // * 'c' contains the result of the most recent comparison, unless
      //   'left'/'right' has reached EOS, in which case it is permanently set
      //   to 1/-1.

      val Leos = CodeLabel()
      Code(Leos, eos)
      val LpullRight = CodeLabel()
      val Lpush = CodeLabel()

      var rightSource: Source[EmitCode] = null
      val leftSource = left(
        eos = rightEOS.mux(
          Leos.goto,
          Code(
            leftEOS := true,
            c := 1, // 'c' will not change again
            pulledRight.mux(
              Lpush.goto,
              Code(pulledRight := true, LpullRight.goto)))),

        push = a => {
          val Lcompare = CodeLabel()

          Code(Lcompare, c := comp(lx, rx), Lpush.goto)

          rightSource = right(
            eos = leftEOS.mux(
              Leos.goto,
              Code(rightEOS := true, c := -1, Lpush.goto)), // 'c' will not change again
            push = b => Code(
              rx := b,
              // If 'left' has ended, we know 'c' == 1, so jumping to 'Lpush'
              // will push 'rx'. If 'right' has not ended, compare 'lx' and 'rx'
              // and push smaller.
              leftEOS.mux(Lpush.goto, Lcompare.goto)))

          Code(Lpush,
               // Push smaller of 'lx' and 'rx', with 'lx' breaking ties.
               (c <= 0).mux(
                 Code(outx := lx.castTo(mb, destRegion.code, outElemType),
                      leftRegion.giveToParent()),
                 Code(outx := rx.castTo(mb, destRegion.code, outElemType),
                      rightRegion.giveToParent())),
               push(outx))
          Code(LpullRight, rightSource.pull)

          Code(
            lx := a,
            // If this was the first pull, still need to pull from 'right.
            // Otherwise, if 'right' has ended, we know 'c' == -1, so jumping
            // to 'Lpush' will push 'lx'. If 'right' has not ended, compare 'lx'
            // and 'rx' and push smaller.
            pulledRight.mux(
              rightEOS.mux(Lpush.goto, Lcompare.goto),
              Code(pulledRight := true, LpullRight.goto)))
        })

      Source[EmitCode](
        setup0 = Code(leftSource.setup0,
                      rightSource.setup0,
                      leftRegion.allocateRegion(Region.REGULAR),
                      rightRegion.allocateRegion(Region.REGULAR)),
        close0 = Code(leftRegion.free(),
                      rightRegion.free(),
                      leftSource.close0,
                      rightSource.close0),
        setup = Code(pulledRight := false,
                     leftEOS := false,
                     rightEOS := false,
                     c := 0,
                     leftSource.setup,
                     rightSource.setup),
        close = Code(leftSource.close,
                     rightSource.close,
                     leftRegion.clear(),
                     rightRegion.clear()),
        // On first pull, pull from 'left', then 'right', then compare.
        // Subsequently, look at 'c' to pull from whichever side was last pushed.
        pull = leftEOS.mux(
          LpullRight.goto,
          (rightEOS || c <= 0).mux(leftSource.pull, LpullRight.goto)))
    }
  }

  def outerJoinRightDistinct(
    mb: EmitMethodBuilder[_],
    lElemType: PType, left: Stream[EmitCode],
    rElemType: PType, right: Stream[EmitCode],
    comp: (EmitValue, EmitValue) => Code[Int]
  ): Stream[(EmitCode, EmitCode)] = new Stream[(EmitCode, EmitCode)] {
    def apply(eos: Code[Ctrl], push: ((EmitCode, EmitCode)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(EmitCode, EmitCode)] = {
      val pulledRight = mb.genFieldThisRef[Boolean]()
      val rightEOS = mb.genFieldThisRef[Boolean]()
      val leftEOS = mb.genFieldThisRef[Boolean]()
      val lx = mb.newEmitField(lElemType) // last value received from left
      val rx = mb.newEmitField(rElemType) // last value received from right
      val lOutMissing = mb.genFieldThisRef[Boolean]("ojrd_lom")
      val rOutMissing = mb.genFieldThisRef[Boolean]("ojrd_rom")
      val c = mb.genFieldThisRef[Int]()

      val Leos = CodeLabel()
      Code(Leos, eos)
      val LpullRight = CodeLabel()
      val LpullLeft = CodeLabel()
      val Lpush = CodeLabel()

      var rightSource: Source[EmitCode] = null
      val leftSource = left(
        eos = rightEOS.mux(
          Leos.goto,
          Code(
            leftEOS := true,
            lOutMissing := true,
            rOutMissing := false,
            (pulledRight && c.cne(0)).mux(
              Lpush.goto,
              Code(
                pulledRight := true,
                LpullRight.goto)))),
        push = a => {
          val Lcompare = CodeLabel()

          Code(Lcompare,
            c := comp(lx, rx),
            lOutMissing := false,
            rOutMissing := false,
            (c > 0).mux(
              pulledRight.mux(
                Code(lOutMissing := true, Lpush.goto),
                Code(pulledRight := true, LpullRight.goto)
              ),
              (c < 0).mux(
                Code(rOutMissing := true, Lpush.goto),
                Code(
                  (lOutMissing || rOutMissing).orEmpty(Code._fatal[Unit]("")),
                  pulledRight := true,
                  Lpush.goto)))
          )

          rightSource = right(
            eos = leftEOS.mux(
              Leos.goto,
              Code(rightEOS := true, lOutMissing := false, rOutMissing := true, Lpush.goto)
            ),
            push = b => Code(
              rx := b,
              leftEOS.mux(
                Code((!lOutMissing || rOutMissing).orEmpty(Code._fatal[Unit]("")), Lpush.goto),
                Lcompare.goto
              )))

          Code(Lpush, push((lx.missingIf(mb, lOutMissing), rx.missingIf(mb, rOutMissing))))
          Code(LpullRight, rightSource.pull)

          Code(
            lx := a,
            pulledRight.mux[Unit](
              rightEOS.mux[Ctrl](
                Code((lOutMissing || !rOutMissing).orEmpty(Code._fatal[Unit]("")), Lpush.goto),
                Code(
                  c.ceq(0).orEmpty(pulledRight := false),
                  Lcompare.goto)),
              Code(pulledRight := true, LpullRight.goto)))
        })

      Code(LpullLeft, leftSource.pull)

      Source[(EmitCode, EmitCode)](
        setup0 = Code(leftSource.setup0, rightSource.setup0),
        close0 = Code(leftSource.close0, rightSource.close0),
        setup = Code(pulledRight := false, leftEOS := false, rightEOS := false, c := 0, leftSource.setup, rightSource.setup),
        close = Code(leftSource.close, rightSource.close),
        pull = leftEOS.mux(LpullRight.goto, rightEOS.mux(LpullLeft.goto, (c <= 0).mux(LpullLeft.goto, LpullRight.goto))))
    }
  }

  def kWayMerge[A: TypeInfo](
    mb: EmitMethodBuilder[_],
    streams: IndexedSeq[StagedRegion => Stream[Code[A]]],
    destRegion: StagedRegion,
    // compare two (idx, value) pairs, where 'value' is a value from the 'idx'th
    // stream
    lt: (Code[Int], Code[A], Code[Int], Code[A]) => Code[Boolean]
  ): Stream[(Code[Int], Code[A])] = new Stream[(Code[Int], Code[A])] {
    def apply(eos: Code[Ctrl], push: ((Code[Int], Code[A])) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(Code[Int], Code[A])] = {
      // The algorithm maintains a tournament tree of comparisons between the
      // current values of the k streams. The tournament tree is a complete
      // binary tree with k leaves. The leaves of the tree are the streams,
      // and each internal node represents the "contest" between the "winners"
      // of the two subtrees, where the winner is the stream with the smaller
      // current key. Each internal node stores the index of the stream which
      // *lost* that contest.
      // Each time we remove the overall winner, and replace that stream's
      // leaf with its next value, we only need to rerun the contests on the
      // path from that leaf to the root, comparing the new value with what
      // previously lost that contest to the previous overall winner.
      val k = streams.length
      // The leaf nodes of the tournament tree, each of which holds a pointer
      // to the current value of that stream.
      val heads = mb.genFieldThisRef[Array[A]]("merge_heads")
      // The internal nodes of the tournament tree, laid out in breadth-first
      // order, each of which holds the index of the stream which lost that
      // contest.
      val bracket = mb.genFieldThisRef[Array[Int]]("merge_bracket")
      // When updating the tournament tree, holds the winner of the subtree
      // containing the updated leaf. Otherwise, holds the overall winner, i.e.
      // the current least element.
      val winner = mb.genFieldThisRef[Int]("merge_winner")
      val i = mb.genFieldThisRef[Int]("merge_i")
      val challenger = mb.genFieldThisRef[Int]("merge_challenger")
      val eltRegions = destRegion.createChildRegionArray(mb, k)

      val runMatch = CodeLabel()
      val LpullChild = CodeLabel()
      val LloopEnd = CodeLabel()
      val Leos = CodeLabel()
      Code(Leos, eos)

      val matchIdx = mb.genFieldThisRef[Int]("merge_match_idx")
      // Compare 'winner' with value in 'matchIdx', loser goes in 'matchIdx',
      // winner goes on to next round. A contestant '-1' beats everything
      // (negative infinity), a contestant 'k' loses to everything
      // (positive infinity), and values in between are indices into 'heads'.
      Code(runMatch,
        challenger := bracket(matchIdx),
        (matchIdx.ceq(0) || challenger.ceq(-1)).orEmpty(LloopEnd.goto),
        (challenger.cne(k) && (winner.ceq(k) || lt(challenger, heads(challenger), winner, heads(winner)))).orEmpty(Code(
          bracket(matchIdx) = winner,
          winner := challenger)),
        matchIdx := matchIdx >>> 1,
        runMatch.goto,

        LloopEnd,
        matchIdx.ceq(0).mux(
          // 'winner' is smallest of all k heads. If 'winner' = k, all heads
          // must be k, and all streams are exhausted.
          winner.ceq(k).mux(
            Leos.goto,
            Code(eltRegions(winner).giveToParent(), push((winner, heads(winner))))),
          // We're still in the setup phase
          Code(bracket(matchIdx) = winner, i := i + 1, winner := i, LpullChild.goto)))

      val sources = streams.zipWithIndex.map { case (stream, idx) =>
        stream(eltRegions(idx))(
          eos = Code(winner := k, matchIdx := (idx + k) >>> 1, runMatch.goto),
          push = elt => Code(heads(idx) = elt, matchIdx := (idx + k) >>> 1, runMatch.goto))
      }

      Code(LpullChild,
        Code.switch(winner,
          Leos.goto, // can only happen if k=0
          sources.map(_.pull.asInstanceOf[Code[Unit]])))

      Source[(Code[Int], Code[A])](
        setup0 = Code(Code(sources.map(_.setup0)), eltRegions.allocateRegions(mb, Region.REGULAR)),
        close0 = Code(eltRegions.freeAll(mb), Code(sources.map(_.close0))),
        setup = Code(
          Code(sources.map(_.setup)),
          bracket := Code.newArray[Int](k),
          heads := Code.newArray[A](k),
          Code.forLoop(i := 0, i < k, i := i + 1, bracket(i) = -1),
          i := 0,
          winner := 0),
        close = Code(Code(sources.map(_.close)), bracket := Code._null, heads := Code._null),
        pull = LpullChild.goto)
    }
  }
}

object EmitStream {

  import Stream._

  def write(
    mb: EmitMethodBuilder[_],
    pcStream: PCanonicalStreamCode,
    ab: StagedArrayBuilder,
    destRegion: StagedRegion
  ): Code[Unit] = {
    _write(mb, pcStream.stream, ab, destRegion)
  }

  private def _write(
    mb: EmitMethodBuilder[_],
    sstream: SizedStream,
    ab: StagedArrayBuilder,
    destRegion: StagedRegion
  ): Code[Unit] = {
    val SizedStream(ssSetup, stream, optLen) = sstream
    val eltRegion = destRegion.createChildRegion(mb)
    Code(FastSeq(
      eltRegion.allocateRegion(Region.REGULAR),
      ssSetup,
      ab.clear,
      ab.ensureCapacity(optLen.getOrElse(16)),
      stream(eltRegion).forEach(mb, { elt => Code(
        elt.setup,
        elt.m.mux(
          ab.addMissing(),
          ab.add(eltRegion.copyToParent(mb, elt.pv).code)),
        eltRegion.clear())
      }),
      eltRegion.free()))
  }

  def toArray(
    mb: EmitMethodBuilder[_],
    aTyp: PArray,
    pcStream: PCanonicalStreamCode,
    destRegion: StagedRegion
  ): PCode = {
    val srvb = new StagedRegionValueBuilder(mb, aTyp, destRegion.code)
    val ss = pcStream.stream
    ss.length match {
      case None =>
        val xLen = mb.newLocal[Int]("sta_len")
        val i = mb.newLocal[Int]("sta_i")
        val vab = new StagedArrayBuilder(aTyp.elementType, mb, 0)
        val ptr = Code(
          _write(mb, ss, vab, destRegion),
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
        PCode(aTyp, ptr)

      case Some(len) =>
        val eltRegion = destRegion.createChildRegion(mb)
        val ptr = Code.sequence1(FastIndexedSeq(
            eltRegion.allocateRegion(Region.REGULAR),
            ss.setup,
            srvb.start(len),
            ss.stream(eltRegion).forEach(mb, { et =>
              Code(FastSeq(
                et.setup,
                et.m.mux(
                  srvb.setMissing(),
                  eltRegion.addToParentRVB(srvb, et.pv)),
                eltRegion.clear(),
                srvb.advance()))
            }),
            eltRegion.clear()),
          srvb.offset)
        PCode(aTyp, ptr)
    }
  }

  def sequence(mb: EmitMethodBuilder[_], elemPType: PType, elements: IndexedSeq[EmitCode]): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val i = mb.genFieldThisRef[Int]()
      val t = mb.newEmitField("ss_t", elemPType)
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
            i := i + 1,
            push(t)),
          Code(Leos, eos)))
    }
  }

  case class SizedStream(setup: Code[Unit], stream: StagedRegion => Stream[EmitCode], length: Option[Code[Int]]) {
    def getStream(eltRegion: StagedRegion): Stream[EmitCode] = stream(eltRegion).addSetup(setup)
  }

  object SizedStream {
    def unsized(stream: StagedRegion => Stream[EmitCode]): SizedStream =
      SizedStream(Code._empty, stream, None)
  }

  def mux(mb: EmitMethodBuilder[_], eltType: PType, cond: Code[Boolean], left: Stream[EmitCode], right: Stream[EmitCode]): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val b = mb.genFieldThisRef[Boolean]()
      val Leos = CodeLabel()
      val elt = mb.newEmitField("stream_mux_elt", eltType)
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

  // Assumes distinct keys in each input stream.
  def kWayZipJoin(
    mb: EmitMethodBuilder[_],
    streams: IndexedSeq[StagedRegion => Stream[PCode]],
    destRegion: StagedRegion,
    resultType: PArray,
    key: IndexedSeq[String]
  ): Stream[(PCode, PCode)] = new Stream[(PCode, PCode)] {
    def apply(eos: Code[Ctrl], push: ((PCode, PCode)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(PCode, PCode)] = {
      // The algorithm maintains a tournament tree of comparisons between the
      // current values of the k streams. The tournament tree is a complete
      // binary tree with k leaves. The leaves of the tree are the streams,
      // and each internal node represents the "contest" between the "winners"
      // of the two subtrees, where the winner is the stream with the smaller
      // current key. Each internal node stores the index of the stream which
      // *lost* that contest.
      // Each time we remove the overall winner, and replace that stream's
      // leaf with its next value, we only need to rerun the contests on the
      // path from that leaf to the root, comparing the new value with what
      // previously lost that contest to the previous overall winner.

      val k = streams.length
      // The leaf nodes of the tournament tree, each of which holds a pointer
      // to the current value of that stream.
      val heads = mb.genFieldThisRef[Array[Long]]("merge_heads")
      // The internal nodes of the tournament tree, laid out in breadth-first
      // order, each of which holds the index of the stream which lost that
      // contest.
      val bracket = mb.genFieldThisRef[Array[Int]]("merge_bracket")
      // When updating the tournament tree, holds the winner of the subtree
      // containing the updated leaf. Otherwise, holds the overall winner, i.e.
      // the current least element.
      val winner = mb.genFieldThisRef[Int]("merge_winner")
      val result = mb.genFieldThisRef[Array[Long]]("merge_result")
      val i = mb.genFieldThisRef[Int]("merge_i")

      val eltType = resultType.elementType.asInstanceOf[PStruct]
      val keyType = eltType.selectFields(key)
      val curKey = ctx.mb.newPField("st_grpby_curkey", keyType)
      val eltRegions = destRegion.createChildRegionArray(mb, k)

      val keyViewType = PSubsetStruct(eltType, key: _*)
      val lt: (Code[Long], Code[Long]) => Code[Boolean] = keyViewType
        .codeOrdering(mb, keyViewType, missingFieldsEqual = false)
        .asInstanceOf[CodeOrdering { type T = Long }]
        .lteqNonnull
      val hasKey: (Code[Long], Code[Long]) => Code[Boolean] = keyViewType
        .codeOrdering(mb, keyType, missingFieldsEqual = false)
        .asInstanceOf[CodeOrdering { type T = Long }]
        .equivNonnull

      val srvb = new StagedRegionValueBuilder(mb, resultType, destRegion.code)

      val runMatch = CodeLabel()
      val LpullChild = CodeLabel()
      val LloopEnd = CodeLabel()
      val LaddToResult = CodeLabel()
      val LstartNewKey = CodeLabel()
      val Leos = CodeLabel()
      val Lpush = CodeLabel()

      Code(Leos, eos)

      Code(Lpush,
        srvb.start(k),
        Code.forLoop(i := 0, i < k, i := i + 1,
          Code(
            result(i).ceq(0L).mux(
              srvb.setMissing(),
              srvb.addIRIntermediate(eltType)(result(i))),
            srvb.advance())),
        push((curKey, PCode(resultType, srvb.offset))))

      val winnerPc = new PSubsetStructCode(keyViewType, heads(winner))

      Code(LstartNewKey,
        Code.forLoop(i := 0, i < k, i := i + 1, result(i) = 0L),
        curKey := eltRegions(winner).copyToParent(mb, winnerPc, keyType),
        LaddToResult.goto)

      Code(LaddToResult,
        result(winner) = heads(winner),
        eltRegions(winner).giveToParent(),
        LpullChild.goto)

      def inSetup: Code[Boolean] = result.isNull

      val matchIdx = mb.genFieldThisRef[Int]("merge_match_idx")
      val challenger = mb.genFieldThisRef[Int]("merge_challenger")
      // Compare 'winner' with value in 'matchIdx', loser goes in 'matchIdx',
      // winner goes on to next round. A contestant '-1' beats everything
      // (negative infinity), a contestant 'k' loses to everything
      // (positive infinity), and values in between are indices into 'heads'.
      Code(runMatch,
        challenger := bracket(matchIdx),
        (matchIdx.ceq(0) || challenger.ceq(-1)).orEmpty(LloopEnd.goto),
        (challenger.cne(k) && (winner.ceq(k) || lt(heads(challenger), heads(winner)))).orEmpty(Code(
          bracket(matchIdx) = winner,
          winner := challenger)),
        matchIdx := matchIdx >>> 1,
        runMatch.goto,

        LloopEnd,
        matchIdx.ceq(0).mux(
          // 'winner' is smallest of all k heads. If 'winner' = k, all heads
          // must be k, and all streams are exhausted.
          inSetup.mux(
            winner.ceq(k).mux(
              Leos.goto,
              Code(result := Code.newArray[Long](k), LstartNewKey.goto)),
            (winner.cne(k) && hasKey(heads(winner), curKey.tcode[Long])).mux(
              LaddToResult.goto,
              Lpush.goto)),
          // We're still in the setup phase
          Code(bracket(matchIdx) = winner, i := i + 1, winner := i, LpullChild.goto)))

      val sources = streams.zipWithIndex.map { case (stream, idx) =>
        stream(eltRegions(idx))(
          eos = Code(winner := k, matchIdx := (idx + k) >>> 1,  runMatch.goto),
          push = elt => Code(
            heads(idx) = elt.castTo(mb, eltRegions(idx).code, eltType).tcode[Long],
            matchIdx := (idx + k) >>> 1,
            runMatch.goto))
      }

      Code(LpullChild,
        Code.switch(winner,
          Leos.goto, // can only happen if k=0
          sources.map(_.pull.asInstanceOf[Code[Unit]])))

      Source[(PCode, PCode)](
        setup0 = Code(Code(sources.map(_.setup0)), eltRegions.allocateRegions(mb, Region.REGULAR)),
        close0 = Code(eltRegions.freeAll(mb), Code(sources.map(_.close0))),
        setup = Code(
          Code(sources.map(_.setup)),
          bracket := Code.newArray[Int](k),
          heads := Code.newArray[Long](k),
          result := Code._null,
          Code.forLoop(i := 0, i < k, i := i + 1, bracket(i) = -1),
          winner := 0),
        close = Code(Code(sources.map(_.close)), bracket := Code._null, heads := Code._null),
        pull = inSetup.mux(
          Code(i := 0, LpullChild.goto),
          winner.ceq(k).mux(
            Leos.goto,
            LstartNewKey.goto)))
    }
  }

  def groupBy(
    mb: EmitMethodBuilder[_],
    stream: StagedRegion => Stream[PCode],
    eltType: PStruct,
    key: Array[String],
    backupRegion: StagedRegion // used when the inner stream is unused
  ): Stream[StagedRegion => Stream[PCode]] = new Stream[StagedRegion => Stream[PCode]] {
    def apply(outerEos: Code[Ctrl], outerPush: (StagedRegion => Stream[PCode]) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[StagedRegion => Stream[PCode]] = {
      val keyType = eltType.selectFields(key)
      val keyViewType = PSubsetStruct(eltType, key)
      val ordering = keyType.codeOrdering(mb, keyViewType, missingFieldsEqual = false).asInstanceOf[CodeOrdering { type T = Long }]

      val xCurKey = ctx.mb.newPField("st_grpby_curkey", keyType)
      val xCurElt = ctx.mb.newPField("st_grpby_curelt", eltType)
      val xInOuter = ctx.mb.genFieldThisRef[Boolean]("st_grpby_io")
      val xEOS = ctx.mb.genFieldThisRef[Boolean]("st_grpby_eos")
      val xNextGrpReady = ctx.mb.genFieldThisRef[Boolean]("st_grpby_ngr")

      var holdingRegion = backupRegion.createDummyChildRegion
      var keyRegion = backupRegion.createDummyChildRegion

      val LchildPull = CodeLabel()
      val LouterPush = CodeLabel()
      val LouterEos = CodeLabel()

      var childSource: Source[PCode] = null
      val inner = (innerEltRegion: StagedRegion) => new Stream[PCode] {
        def apply(innerEos: Code[Ctrl], innerPush: PCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[PCode] = {
          holdingRegion = innerEltRegion.createChildRegion(mb)
          keyRegion = innerEltRegion.createChildRegion(mb)
          val LinnerEos = CodeLabel()
          val LinnerPush = CodeLabel()

          childSource = stream(holdingRegion)(
            xInOuter.mux(LouterEos.goto, Code(xEOS := true, LinnerEos.goto)),
            { a: PCode =>
              Code(
                xCurElt := a,
                // !xInOuter iff this element was requested by an inner stream.
                // Else we are stepping to the beginning of the next group.
                (xCurKey.tcode[Long].cne(0L) && ordering.equivNonnull(xCurKey.tcode[Long], xCurElt.tcode[Long])).mux(
                  xInOuter.mux(
                    Code(holdingRegion.clear(), LchildPull.goto),
                    LinnerPush.goto),
                  Code(
                    keyRegion.clear(),
                    xCurKey := {
                      val pc = new PSubsetStructCode(keyViewType, xCurElt.tcode[Long])
                      pc.castTo(mb, keyRegion.code, keyType)
                    },
                    xInOuter.mux(
                      LouterPush.goto,
                      Code(xNextGrpReady := true, LinnerEos.goto)))))
            })

          Code(LinnerPush, holdingRegion.giveToParent(), innerPush(xCurElt))
          Code(LinnerEos, innerEos)

          Source[PCode](
            setup0 = Code._empty,
            close0 = Code._empty,
            setup = Code._empty,
            close = Code._empty,
            pull = xInOuter.mux(
              // xInOuter iff this is the first pull from inner stream,
              // in which case the element has already been produced.
              // Otherwise holdingRegion is empty, because we were just in LinnerPush
              Code(xInOuter := false, LinnerPush.goto),
              LchildPull.goto))
        }
      }

      Code(LouterPush, outerPush(inner))

      if (childSource == null) {
        // inner stream is unused
        val Lunreachable = CodeLabel()
        Code(Lunreachable, Code._fatal[Unit]("unreachable"))
        // because LinnerPush is never executed, nothing is ever given to backupRegion;
        // holdingRegion will be cleared every element, and keyRegion cleared every new key.
        val unusedInnerSource = inner(backupRegion)(Lunreachable.goto, _ => Lunreachable.goto)
      }

      // Precondition: holdingRegion is empty
      Code(LchildPull, childSource.pull)

      Code(LouterEos, outerEos)

      Source[StagedRegion => Stream[PCode]](
        setup0 = Code(childSource.setup0, holdingRegion.allocateRegion(Region.REGULAR), keyRegion.allocateRegion(Region.TINIER)),
        close0 = Code(holdingRegion.free(), keyRegion.free(), childSource.close0),
        setup = Code(
          childSource.setup,
          xCurKey := keyType.defaultValue,
          xEOS := false,
          xNextGrpReady := false),
        close = childSource.close,
        pull = Code(
          xInOuter := true,
          xEOS.mux(
            LouterEos.goto,
            xNextGrpReady.mux(
              Code(xNextGrpReady := false, LouterPush.goto),
              LchildPull.goto))))
    }
  }

  def extendNA(mb: EmitMethodBuilder[_], eltType: PType, stream: Stream[EmitCode]): Stream[COption[EmitCode]] = new Stream[COption[EmitCode]] {
    def apply(eos: Code[Ctrl], push: COption[EmitCode] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[COption[EmitCode]] = {
      val atEnd = mb.genFieldThisRef[Boolean]()
      val x = mb.newEmitField(eltType)
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

  private[ir] def emit[C](
    emitter: Emit[C],
    streamIR0: IR,
    mb: EmitMethodBuilder[C],
    outerRegion: StagedRegion,
    env0: Emit.E,
    container: Option[AggContainer]
  ): EmitCode = {

    def _emitStream(streamIR: IR, outerRegion: StagedRegion, env: Emit.E): COption[SizedStream] = {

      def emitStream(streamIR: IR, outerRegion: StagedRegion = outerRegion, env: Emit.E = env): COption[SizedStream] =
        _emitStream(streamIR, outerRegion, env)

      def emitStreamToEmitCode(streamIR: IR, outerRegion: StagedRegion = outerRegion, env: Emit.E = env): EmitCode =
        COption.toEmitCode(
          _emitStream(streamIR, outerRegion, env).map { stream =>
            PCanonicalStreamCode(streamIR.pType.asInstanceOf[PCanonicalStream], stream)
          }, mb)

      def emitIR(ir: IR, env: Emit.E = env, region: StagedRegion = outerRegion, container: Option[AggContainer] = container): EmitCode =
        emitter.emitWithRegion(ir, mb, region, env, container)

      def emitVoidIR(ir: IR, env: Emit.E = env, container: Option[AggContainer] = container): Code[Unit] = {
        EmitCodeBuilder.scopedVoid(mb) { cb =>
          emitter.emitVoid(cb, ir, mb, outerRegion, env, container, None)
        }
      }

      streamIR match {
        case x@NA(_) =>
          COption.none(coerce[PCanonicalStream](x.pType).defaultValue.stream)

        case x@Ref(name, _) =>
          val typ = coerce[PStream](x.pType)
          val ev = env.lookup(name)
          if (ev.pt != typ)
            throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $typ")
          COption.fromEmitCode(ev.get).map(_.asStream.stream)

        case x@StreamRange(startIR, stopIR, stepIR) =>
          val eltType = coerce[PStream](x.pType).elementType
          val step = mb.genFieldThisRef[Int]("sr_step")
          val start = mb.genFieldThisRef[Int]("sr_start")
          val stop = mb.genFieldThisRef[Int]("sr_stop")
          val llen = mb.genFieldThisRef[Long]("sr_llen")
          val len = mb.genFieldThisRef[Int]("sr_len")

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
                        len := llen.toI,
                        eltRegion => range(mb, start, step, len)
                          .map(i => EmitCode(Code._empty, const(false), PCode(eltType, i))),
                        Some(len)))))))
            }
          }

        case ToStream(containerIR) =>
          COption.fromEmitCode(emitIR(containerIR)).mapCPS { (containerAddr, k) =>
            val (asetup, a) = EmitCodeBuilder.scoped(mb) { cb =>
              containerAddr.asIndexable.memoize(cb, "ts_a")
            }

            val len = mb.genFieldThisRef[Int]("ts_len")
            val i = mb.genFieldThisRef[Int]("ts_i")
            val newStream = new Stream[EmitCode] {
              def apply(eos: Code[Ctrl], push: (EmitCode) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] =
                new Source[EmitCode](
                  setup0 = Code._empty,
                  setup = i := 0,
                  close = Code._empty,
                  close0 = Code._empty,
                  pull = (i < len).mux(
                    Code(i := i + 1,
                      push(
                        EmitCode.fromI(mb) { cb =>
                          a.loadElement(cb, i - 1)
                        })),
                    eos))
            }

            Code(
              asetup,
              len := a.loadLength(),
              k(SizedStream(Code._empty, eltRegion => newStream, Some(len))))
          }

        case x@MakeStream(elements, _) =>
          val eltType = coerce[PStream](x.pType).elementType
          val stream = (eltRegion: StagedRegion) =>
            sequence(mb, eltType, elements.toFastIndexedSeq.map { ir =>
              emitIR(ir, region = eltRegion)
                .map(_.castTo(mb, eltRegion.code, eltType))
            })

          COption.present(SizedStream(Code._empty, stream, Some(elements.length)))

        case x@ReadPartition(context, rowType, reader) =>
          reader.emitStream(context, rowType, emitter, mb, outerRegion, env, container)

        case In(n, PCanonicalStream(eltType, _)) =>
          val xIter = mb.genFieldThisRef[Iterator[java.lang.Long]]("streamInIterator")
          val hasNext = mb.genFieldThisRef[Boolean]("streamInHasNext")
          val next = mb.genFieldThisRef[Long]("streamInNext")

          // this, Region, ...
          mb.getStreamEmitParam(2 + n).map { iter =>
            val stream = unfold[Code[Long]](
              (_, k) => Code(
                hasNext := xIter.load().hasNext,
                hasNext.orEmpty(next := xIter.load().next().invoke[Long]("longValue")),
                k(COption(!hasNext, next)))
            ).map(
              rv => EmitCode.present(eltType, Region.loadIRIntermediate(eltType)(rv)),
              setup0 = None,
              setup = Some(xIter := iter)
            )

            SizedStream.unsized(eltRegion => stream)
          }

        case StreamTake(a, num) =>
          val optStream = emitStream(a)
          val optN = COption.fromEmitCode(emitIR(num))
          val xN = mb.genFieldThisRef[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optN.map { n => SizedStream(
              Code(setup,
                   xN := n.tcode[Int],
                   (xN < 0).orEmpty(Code._fatal[Unit](const("StreamTake: negative length")))),
              eltRegion => zip(stream(eltRegion), range(mb, 0, 1, xN))
                .map({ case (elt, count) => elt }),
              len.map(_.min(xN)))
            }
          }

        case StreamDrop(a, num) =>
          val optStream = emitStream(a)
          val optN = COption.fromEmitCode(emitIR(num))
          val xN = mb.genFieldThisRef[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optN.map { n => SizedStream(
              Code(setup,
                   xN := n.tcode[Int],
                   (xN < 0).orEmpty(Code._fatal[Unit](const("StreamDrop: negative num")))),
              eltRegion => zip(stream(eltRegion), iota(mb, 0, 1))
                .map({ case (elt, count) => COption(count < xN, elt) })
                .flatten,
              len.map(l => (l - xN).max(0)))
            }
          }

        case x@StreamGrouped(a, size) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          val optStream = emitStream(a)
          val optSize = COption.fromEmitCode(emitIR(size))
          val xS = mb.genFieldThisRef[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optSize.map { s =>
              SizedStream(
                Code(setup, xS := s.tcode[Int], (xS <= 0).orEmpty(Code._fatal[Unit](const("StreamGrouped: nonpositive size")))),
                eltRegion => Stream.grouped(mb, stream, xS, eltRegion)
                  .map { inner =>
                    EmitCode(
                      Code._empty,
                      false,
                      PCanonicalStreamCode(
                        innerType,
                        SizedStream.unsized(inner)))
                  },
                len.map(l => ((l.toL + xS.toL - 1L) / xS.toL).toI)) // rounding up integer division
            }
          }

        case x@StreamGroupByKey(a, key) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          val eltType = coerce[PStruct](innerType.elementType)
          val optStream = emitStream(a)
          optStream.map { ss =>
            val nonMissingStream = (eltRegion: StagedRegion) => ss.getStream(eltRegion).mapCPS[PCode] { (_, ec, k) =>
              Code(ec.setup, ec.m.orEmpty(Code._fatal[Unit](const("expected non-missing"))), k(ec.pv))
            }
            SizedStream.unsized { eltRegion =>
              groupBy(mb, nonMissingStream, eltType, key.toArray, eltRegion).map { inner =>
                EmitCode.present(
                  PCanonicalStreamCode(innerType,
                    SizedStream.unsized { innerEltRegion =>
                      inner(innerEltRegion).map(EmitCode.present)
                    }))
              }
            }
          }

        case StreamMap(childIR, name, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR)
          optStream.map { case SizedStream(setup, stream, len) =>
            def newStream(eltRegion: StagedRegion) = stream(eltRegion).map { eltt => (eltType, bodyIR.pType) match {
              case (eltType: PCanonicalStream, _: PCanonicalStream) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))

                emitStreamToEmitCode(bodyIR, outerRegion = eltRegion, env = bodyenv)
              case (eltType: PCanonicalStream, _) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))

                emitIR(bodyIR, region = eltRegion, env = bodyenv)
              case (_, _: PCanonicalStream) =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)

                EmitCode(
                  xElt := eltt,
                  emitStreamToEmitCode(bodyIR, outerRegion = eltRegion, env = bodyenv))
              case _ =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)
                val bodyt = emitIR(bodyIR, region = eltRegion, env = bodyenv)

                EmitCode(xElt := eltt, bodyt)
            }}

            SizedStream(setup, newStream, len)
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR)

          optStream.map { ss =>
            val newStream = (eltRegion: StagedRegion) => {
              val tmpRegion = eltRegion.createChildRegion(mb)
              ss.getStream(tmpRegion)
                .map (
                  { elt =>
                    val xElt = mb.newEmitField(name, childEltType)
                    val cond = emitIR(condIR, env = env.bind(name -> xElt), region = tmpRegion)

                    new COption[EmitCode] {
                      def apply(none: Code[Ctrl], some: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
                        Code(
                          xElt := elt,
                          cond.setup,
                          (cond.m || !cond.value[Boolean]).mux(
                            Code(tmpRegion.clear(), none),
                            some(EmitCode.fromI(mb) { cb =>
                              xElt.toI(cb)
                                .mapMissing(cb) { cb += tmpRegion.clear() }
                                .map(cb) { pc =>
                                  cb += tmpRegion.giveToParent()
                                  pc
                                }
                            })))
                      }
                    }
                  },
                  setup0 = Some(tmpRegion.allocateRegion(Region.REGULAR)),
                  close0 = Some(tmpRegion.free()))
                .flatten
            }

            SizedStream.unsized(newStream)
          }

        case x@StreamMerge(leftIR, rightIR, key) =>
          val lElemType = coerce[PStruct](coerce[PStream](leftIR.pType).elementType)
          val rElemType = coerce[PStruct](coerce[PStream](rightIR.pType).elementType)
          val outElemType = coerce[PStream](x.pType).elementType

          val lKeyViewType = PSubsetStruct(lElemType, key: _*)
          val rKeyViewType = PSubsetStruct(rElemType, key: _*)
          val ordering = lKeyViewType.codeOrdering(mb, rKeyViewType, missingFieldsEqual = false).asInstanceOf[CodeOrdering { type T = Long }]

          def compare(lelt: EmitValue, relt: EmitValue): Code[Int] = {
            assert(lelt.pt == lElemType)
            assert(relt.pt == rElemType)
            ordering.compare((lelt.m, lelt.value[Long]), (relt.m, relt.value[Long]))
          }

          emitStream(leftIR).flatMap { case SizedStream(leftSetup, leftStream, leftLen) =>
            emitStream(rightIR).map { case SizedStream(rightSetup, rightStream, rightLen) =>
              SizedStream(
                Code(leftSetup, rightSetup),
                eltRegion => merge(mb,
                  lElemType, leftStream,
                  rElemType, rightStream,
                  outElemType, eltRegion, compare),
                for (l <- leftLen; r <- rightLen) yield l + r)
            }
          }

        case StreamZip(as, names, bodyIR, behavior) =>
          // FIXME: should make StreamZip support unrealizable element types
          val eltTypes = {
            val types = as.map(ir => coerce[PStream](ir.pType).elementType)
            behavior match {
              case ArrayZipBehavior.ExtendNA => types.map(_.setRequired(false))
              case _ => types
            }
          }
          val eltVars = (names, eltTypes).zipped.map(mb.newEmitField)

          val optStreams = COption.lift(as.map(emitStream(_)))

          optStreams.map { emitStreams =>
            val lenSetup = Code(emitStreams.map(_.setup))
            val streams = emitStreams.map(_.stream(outerRegion))
            val lengths = emitStreams.map(_.length)

            behavior match {

              case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
                val newStream = (eltRegion: StagedRegion) =>
                  multiZip(emitStreams.map(_.stream(eltRegion)))
                    .map { elts =>
                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      val body = emitIR(bodyIR, env = bodyEnv, region = eltRegion)
                      EmitCode(Code(Code((eltVars, elts).zipped.map { (v, x) => v := x }), body.setup), body.m, body.pv)
                    }
                val newLength = behavior match {
                  case ArrayZipBehavior.TakeMinLength =>
                    lengths.reduceLeft(_.liftedZip(_).map {
                      case (l1, l2) => l1.min(l2)
                    })
                  case ArrayZipBehavior.AssumeSameLength =>
                    lengths.flatten.headOption
                }

                SizedStream(lenSetup, newStream, newLength)

              case ArrayZipBehavior.AssertSameLength =>
                val newStream = (eltRegion: StagedRegion) => {
                  // extend to infinite streams, where the COption becomes missing after EOS
                  val extended: IndexedSeq[Stream[COption[EmitCode]]] =
                    emitStreams.map(_.stream(eltRegion)).zipWithIndex.map { case (stream, i) =>
                      extendNA(mb, eltTypes(i), stream)
                    }

                  // zip to an infinite stream, where the COption is missing when all streams are EOS
                  val flagged: Stream[COption[EmitCode]] = multiZip(extended)
                    .mapCPS { (_, elts, k) =>
                      val allEOS = mb.genFieldThisRef[Boolean]("zip_stream_all_eos")
                      val anyEOS = mb.genFieldThisRef[Boolean]("zip_stream_any_eos")
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
                      val body = emitIR(bodyIR, env = bodyEnv, region = eltRegion)

                      Code(
                        allEOS := true,
                        anyEOS := false,
                        Code(checkedElts),
                        (anyEOS & !allEOS).mux[Unit](
                          Code._fatal[Unit]("zip: length mismatch"),
                          k(COption(allEOS, body))): Code[Ctrl])
                    }

                  // termininate the stream when all streams are EOS
                  flagged.take
                }

                val newLength = lengths.flatten match {
                  case Seq() => None
                  case ls =>
                    val len = mb.genFieldThisRef[Int]("zip_asl_len")
                    val lenTemp = mb.genFieldThisRef[Int]("zip_asl_len_temp")
                    Some(Code(
                      len := ls.head,
                      ls.tail.foldLeft(Code._empty) { (acc, l) =>
                        Code(acc,
                          lenTemp := l,
                          len.cne(lenTemp).orEmpty(Code._fatal[Unit](
                            const("zip: length mismatch: ").concat(len.toS).concat(", ").concat(lenTemp.toS))))
                      },
                      len))
                }

                SizedStream(lenSetup, newStream, newLength)

              case ArrayZipBehavior.ExtendNA =>
                val newStream = (eltRegion: StagedRegion) => {
                  // extend to infinite streams, where the COption becomes missing after EOS
                  val extended: IndexedSeq[Stream[COption[EmitCode]]] =
                    emitStreams.map(_.stream(eltRegion)).zipWithIndex.map { case (stream, i) =>
                      extendNA(mb, eltTypes(i), stream)
                    }

                  // zip to an infinite stream, where the COption is missing when all streams are EOS
                  val flagged: Stream[COption[EmitCode]] = multiZip(extended)
                    .mapCPS { (_, elts, k) =>
                      val allEOS = mb.genFieldThisRef[Boolean]()
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

                          COption.toEmitCode(optElt, mb)
                        }
                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      val body = emitIR(bodyIR, env = bodyEnv, region = eltRegion)

                      Code(
                        allEOS := true,
                        Code((eltVars, checkedElts).zipped.map { (v, x) => v := x }),
                        k(COption(allEOS, body)))
                    }

                  // termininate the stream when all streams are EOS
                  flagged.take
                }

                val newLength = lengths.reduceLeft(_.liftedZip(_).map {
                  case (l1, l2) => l1.max(l2)
                })

                SizedStream(lenSetup, newStream, newLength)
            }
          }

        case x@StreamMultiMerge(as, key) =>
          val eltType = x.pType.elementType.asInstanceOf[PStruct]

          val keyViewType = PSubsetStruct(eltType, key: _*)
          val ord = keyViewType
            .codeOrdering(mb, keyViewType)
            .asInstanceOf[CodeOrdering { type T = Long }]
          def comp(li: Code[Int], lv: Code[Long], ri: Code[Int], rv: Code[Long]): Code[Boolean] =
            Code.memoize(ord.compareNonnull(lv, rv), "stream_merge_comp") { c =>
              c < 0 || (c.ceq(0) && li < ri)
            }

          COption.lift(as.map(emitStream(_))).map { sss =>
            val streams = sss.map { ss => (eltRegion: StagedRegion) =>
              ss.stream(eltRegion).map { ec =>
                ec.get().castTo(mb, outerRegion.code, eltType).tcode[Long]
              }
            }
            SizedStream(
              Code(sss.map(_.setup)),
              eltRegion => kWayMerge[Long](mb, streams, eltRegion, comp).map { case (i, elt) =>
                EmitCode.present(PCode(eltType, elt))
              },
              sss.map(_.length).reduce(_.liftedZip(_).map {
                case (l, r) => l + r
              }))
          }

        case x@StreamZipJoin(as, key, curKey, curVals, joinIR) =>
          val curValsType = x.curValsType
          val eltType = curValsType.elementType.setRequired(true).asInstanceOf[PStruct]
          val keyType = eltType.selectFields(key)

          def joinF(eltRegion: StagedRegion): ((PCode, PCode)) => EmitCode = { case (k, vs) =>
            val xKey = mb.newPresentEmitField("zipjoin_key", keyType)
            val xElts = mb.newPresentEmitField("zipjoin_elts", curValsType)
            val newEnv = env.bind(curKey -> xKey, curVals -> xElts)
            val joint = joinIR.pType match {
              case _: PCanonicalStream =>
                emit(emitter, joinIR, mb, outerRegion, newEnv, container)
              case _ =>
                emitIR(joinIR, env = newEnv, region = eltRegion)
            }

            EmitCode(Code(xKey := k, xElts := vs), joint)
          }

          COption.lift(as.map(emitStream(_))).map { sss =>
            val streams = sss.map { ss => (eltRegion: StagedRegion) =>
              ss.getStream(eltRegion).map(_.get())
            }
            SizedStream.unsized { eltRegion =>
              kWayZipJoin(mb, streams, eltRegion, curValsType, key)
                .map(joinF(eltRegion))
            }
          }

        case StreamFlatMap(outerIR, name, innerIR) =>
          val outerEltType = coerce[PStream](outerIR.pType).elementType

          val optOuter = emitStream(outerIR)

          optOuter.map { outer =>
            val nested = (eltRegion: StagedRegion, outerEltRegion: StagedOwnedRegion) => {
              outer.getStream(outerEltRegion).map[COption[Stream[EmitCode]]] { elt =>
                val optInner = if (outerEltType.isRealizable) {
                  val xElt = mb.newEmitField(name, outerEltType)
                  val innerEnv = env.bind(name -> xElt)
                  val optInner = emitStream(innerIR, outerRegion = outerEltRegion, env = innerEnv)

                  optInner.addSetup(xElt := elt)
                } else {
                  val innerEnv = env.bind(name -> new EmitUnrealizableValue(outerEltType, elt))

                  emitStream(innerIR, outerRegion = outerEltRegion, env = innerEnv)
                }

                optInner.map { inner =>
                  inner.getStream(eltRegion)
                    .map(x => EmitCode(outerEltRegion.shareWithParent(), x),
                         close = Some(outerEltRegion.clear()))
                }
              }
            }

            SizedStream.unsized { eltRegion =>
              val outerEltRegion = eltRegion.createChildRegion(mb)
              nested(eltRegion, outerEltRegion)
                .flatten.flatten
                .map(x => x,
                     setup0 = Some(outerEltRegion.allocateRegion(Region.REGULAR)),
                     close0 = Some(outerEltRegion.free()))
            }
          }

        case If(condIR, thn, els) =>
          val eltType = coerce[PStream](thn.pType).elementType
          val xCond = mb.genFieldThisRef[Boolean]("stream_if_cond")

          val condT = COption.fromEmitCode(emitIR(condIR))
          val optLeftStream = emitStream(thn)
          val optRightStream = emitStream(els)

          condT.flatMap[SizedStream] { cond =>
            val newOptStream = COption.choose[SizedStream](
              xCond,
              optLeftStream,
              optRightStream,
              { case (SizedStream(leftSetup, leftStream, lLen), SizedStream(rightSetup, rightStream, rLen)) =>
                  val newStream = mux(mb, eltType,
                    xCond,
                    leftStream(outerRegion),
                    rightStream(outerRegion))
                  val newLen = lLen.liftedZip(rLen).map { case (l1, l2) =>
                    xCond.mux(l1, l2)
                  }
                  val newSetup = xCond.mux(leftSetup, rightSetup)

                  SizedStream(newSetup, eltRegion => newStream, newLen)
              })

            newOptStream.addSetup(xCond := cond.tcode[Boolean])
          }

        case Let(name, valueIR, bodyIR) =>
          val valueType = valueIR.pType

          valueType match {
            case _: PCanonicalStream =>
              val valuet = emit(emitter, valueIR, mb, outerRegion, env, container)
              val bodyEnv = env.bind(name -> new EmitUnrealizableValue(valueType, valuet))

              emitStream(bodyIR, env = bodyEnv)

            case _ =>
              val xValue = mb.newEmitField(name, valueType)
              val bodyEnv = env.bind(name -> xValue)
              val valuet = emitIR(valueIR)

              emitStream(bodyIR, env = bodyEnv).addSetup(xValue := valuet)
          }

        case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = x.accPType

          val streamOpt = emitStream(childIR)
          streamOpt.map { case SizedStream(setup, stream, len) =>
            val Lpush = CodeLabel()
            val hasPulled = mb.genFieldThisRef[Boolean]()

            val xElt = mb.newEmitField(eltName, eltType)
            val xAcc = mb.newEmitField(accName, accType)
            val tmpAcc = mb.newEmitField(accName, accType)

            val zero = emitIR(zeroIR).map(_.castTo(mb, outerRegion.code, accType))
            val bodyEnv = env.bind(accName -> tmpAcc, eltName -> xElt)

            val body = emitIR(bodyIR, env = bodyEnv).map(_.castTo(mb, outerRegion.code, accType))

            val newStream = new Stream[EmitCode] {
              def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
                val source = stream(outerRegion)(
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

            val newLen = len.map(l => l + 1)
            SizedStream(setup, eltRegion => newStream, newLen)
          }

        case x@RunAggScan(array, name, init, seqs, result, states) =>
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb, "array_agg_scan")

          val eltType = coerce[PStream](array.pType).elementType

          val xElt = mb.newEmitField("aggscan_elt", eltType)
          val xResult = mb.newEmitField("aggscan_result", result.pType)

          val bodyEnv = env.bind(name -> xElt)
          val cInit = emitVoidIR(init, container = Some(newContainer))
          val seqPerElt = emitVoidIR(seqs, env = bodyEnv, container = Some(newContainer))
          val postt = emitIR(result, env = bodyEnv, container = Some(newContainer))

          val optStream = emitStream(array)

          optStream.map { case SizedStream(setup, stream, len) =>
            val newStream = stream(outerRegion).map[EmitCode](
              { eltt =>
                EmitCode(
                  Code(
                    xElt := eltt,
                    xResult := postt,
                    seqPerElt),
                  xResult.get)
              },
              setup0 = Some(aggSetup),
              close0 = Some(aggCleanup),
              setup = Some(cInit))

            SizedStream(setup, eltRegion => newStream, len)
          }

        case StreamJoinRightDistinct(leftIR, rightIR, lKey, rKey, leftName, rightName, joinIR, joinType) =>
          assert(joinType == "left" || joinType == "outer")
          val lEltType = coerce[PStruct](coerce[PStream](leftIR.pType).elementType)
          val rEltType = coerce[PStruct](coerce[PStream](rightIR.pType).elementType)
          val xLElt = mb.newEmitField("join_lelt", lEltType.orMissing(joinType == "left"))
          val xRElt = mb.newEmitField("join_relt", rEltType.setRequired(false))
          val newEnv = env.bind(leftName -> xLElt, rightName -> xRElt)

          val lKeyViewType = PSubsetStruct(lEltType, lKey: _*)
          val rKeyViewType = PSubsetStruct(rEltType, rKey: _*)
          val ordering = lKeyViewType.codeOrdering(mb, rKeyViewType, missingFieldsEqual = false).asInstanceOf[CodeOrdering { type T = Long }]

          def compare(lelt: EmitValue, relt: EmitValue): Code[Int] = {
            assert(lelt.pt == lEltType)
            assert(relt.pt == rEltType)
            ordering.compare((lelt.m, lelt.value[Long]), (relt.m, relt.value[Long]))
          }

          def joinF: ((EmitCode, EmitCode)) => EmitCode = { case (lelt, relt) =>
            val joint = joinIR.pType match {
              case _: PCanonicalStream =>
                emit(emitter, joinIR, mb, outerRegion, newEnv, container)
              case _ =>
                emitIR (joinIR, newEnv)
            }

            EmitCode(Code(xLElt := lelt, xRElt := relt), joint)
          }

          emitStream(leftIR).flatMap { case SizedStream(leftSetup, leftStream, leftLen) =>
            emitStream(rightIR).map { rightSS =>
              val rightStream = rightSS.getStream(outerRegion)
              val newStream = if (joinType == "left")
                leftJoinRightDistinct(
                  mb,
                  lEltType, leftStream(outerRegion),
                  rEltType, rightStream,
                  compare)
                  .map(joinF)
              else
                outerJoinRightDistinct(
                  mb,
                  lEltType, leftStream(outerRegion),
                  rEltType, rightStream,
                  compare)
                  .map(joinF)

              SizedStream(leftSetup, eltRegion => newStream, if (joinType == "left") leftLen else None)
            }
          }

        case x@ShuffleRead(idIR, keyRangeIR) =>
          val shuffleType = coerce[TShuffle](idIR.typ)
          assert(shuffleType.rowDecodedPType == coerce[PStream](x.pType).elementType)
          val keyType = coerce[TInterval](keyRangeIR.typ).pointType
          val keyPType = coerce[PInterval](keyRangeIR.pType).pointType
          assert(keyType == shuffleType.keyType)
          assert(keyPType == shuffleType.keyDecodedPType)

          COption.fromEmitCode(emitIR(idIR)).doIfNone(
            Code._fatal[Unit]("ShuffleRead cannot have null ID")
          ).flatMap { case (idt: PCanonicalShuffleCode) =>
            COption.fromEmitCode(emitIR(keyRangeIR)).doIfNone(
              Code._fatal[Unit]("ShuffleRead cannot have null key range")
            ).flatMap { case (keyRanget: PIntervalCode) =>
              val intervalPhysicalType = keyRanget.pt

              val uuidLocal = mb.newLocal[Long]("shuffleUUID")
              val uuid = new PCanonicalShuffleSettable(idt.pt.asInstanceOf[PCanonicalShuffle], uuidLocal)
              val keyRange = mb.newLocal[Long]("shuffleClientKeyRange")
              COption(
                Code(
                  uuidLocal := idt.tcode[Long],
                  keyRange := keyRanget.tcode[Long],
                  !intervalPhysicalType.startDefined(keyRange) || !intervalPhysicalType.endDefined(keyRange)),
                keyRange
              ).doIfNone(
                Code._fatal[Unit]("ShuffleRead cannot have null start or end points of key range")
              ).map { (keyRange: LocalRef[Long]) =>
                val startt = intervalPhysicalType.loadStart(keyRange)
                val startInclusivet = intervalPhysicalType.includesStart(keyRange)
                val endt = intervalPhysicalType.loadEnd(keyRange)
                val endInclusivet = intervalPhysicalType.includesEnd(keyRange)

                val shuffleLocal = mb.newLocal[ShuffleClient]("shuffleClient")
                val shuffle = new ValueShuffleClient(shuffleLocal)

                val stream = unfold[EmitCode](
                  { (_, k) =>
                    k(
                      COption(
                        shuffle.getValueFinished(),
                        EmitCode.present(
                          shuffleType.rowDecodedPType, shuffle.getValue(outerRegion.code))))
                  },
                  setup = Some(Code(
                    shuffleLocal := CodeShuffleClient.create(
                      mb.ecb.getType(shuffleType),
                      uuid.loadBytes(),
                      Code._null,
                      mb.ecb.getPType(keyPType)),
                    shuffle.startGet(startt, startInclusivet, endt, endInclusivet))),
                  close = Some(Code(
                    shuffle.getDone(),
                    shuffle.close())))
                SizedStream.unsized(eltRegion => stream)
              }
            }
          }

      case x@ShufflePartitionBounds(idIR, nPartitionsIR) =>
          val shuffleType = coerce[TShuffle](idIR.typ)
          assert(shuffleType.keyDecodedPType == coerce[PStream](x.pType).elementType)
          COption.fromEmitCode(emitIR(idIR)).doIfNone(
            Code._fatal[Unit]("ShufflePartitionBounds cannot have null ID")
          ).flatMap { case (idt: PCanonicalShuffleCode) =>
            COption.fromEmitCode(emitIR(nPartitionsIR)).doIfNone(
              Code._fatal[Unit]("ShufflePartitionBounds cannot have null number of partitions")
            ).map { case (nPartitionst: PPrimitiveCode) =>
              val uuidLocal = mb.newLocal[Long]("shuffleUUID")
              val uuid = new PCanonicalShuffleSettable(idt.pt.asInstanceOf[PCanonicalShuffle], uuidLocal)
              val shuffleLocal = mb.newLocal[ShuffleClient]("shuffleClient")
              val shuffle = new ValueShuffleClient(shuffleLocal)
              val stream = unfold[EmitCode](
                { (_, k) =>
                  k(
                    COption(
                      shuffle.partitionBoundsValueFinished(),
                      EmitCode.present(
                        shuffleType.keyDecodedPType, shuffle.partitionBoundsValue(outerRegion.code))))
                },
                setup = Some(Code(
                  uuidLocal := idt.tcode[Long],
                  shuffleLocal := CodeShuffleClient.create(mb.ecb.getType(shuffleType), uuid.loadBytes()),
                  shuffle.startPartitionBounds(nPartitionst.codeTuple()(0).asInstanceOf[Code[Int]]))),
                close = Some(Code(
                  shuffle.endPartitionBounds(),
                  shuffle.close())))
              SizedStream.unsized(eltRegion => stream)
          }
        }

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }
    }

    COption.toEmitCode(
      _emitStream(streamIR0, outerRegion, env0).map { stream =>
        PCanonicalStreamCode(streamIR0.pType.asInstanceOf[PCanonicalStream], stream)
      }, mb)
  }

  private[ir] def multiplicity(root: IR, refName: String): Int = {
    var uses = 0

    // assumes no name collisions, a bit hacky...
    def traverse(ir: IR, mult: Int): Unit = ir match {
      case Ref(name, _) => if (refName == name) uses += mult
      case StreamMap(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamFilter(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamFlatMap(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamJoinRightDistinct(l, r, _, _, _, c, j, _) =>
        traverse(l, mult); traverse(r, mult); traverse(j, 2)
      case StreamScan(a, z, _, _, b) =>
        traverse(a, mult); traverse(z, 2); traverse(b, 2)
      case RunAggScan(a, _, i, s, r, _) =>
        traverse(a, mult); traverse(i, 2); traverse(s, 2); traverse(r, 2)
      case StreamZipJoin(as, _, _, _, f) =>
        as.foreach(traverse(_, mult)); traverse(f, 2)
      case StreamZip(as, _, body, _) =>
        as.foreach(traverse(_, mult)); traverse(body, 2)
      case StreamFold(a, zero, _, _, body) =>
        traverse(a, mult); traverse(zero, mult); traverse(body, 2)
      case StreamFold2(a, accs, _, seqs, res) =>
        traverse(a, mult)
        accs.foreach { case (_, acc) => traverse(acc, mult) }
        seqs.foreach(traverse(_, 2))
        traverse(res, 2)
      case StreamFor(a, _, body) =>
        traverse(a, mult); traverse(body, 2)
      case NDArrayMap(a, _, body) =>
        traverse(a, mult); traverse(body, 2)
      case NDArrayMap2(l, r, _, _, body) =>
        traverse(l, mult); traverse(r, mult); traverse(body, 2)

      case _ => ir.children.foreach {
        case child: IR => traverse(child, mult)
        case _ =>
      }
    }
    traverse(root, 1)
    uses min 2
  }

  def isIterationLinear(ir: IR, refName: String): Boolean =
    multiplicity(ir, refName) <= 1

}
