package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.services.shuffler._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerSettable, SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable, SIntervalPointer, SIntervalPointerSettable, SSubsetStruct, SSubsetStructCode}
import is.hail.types.physical.stypes.{interfaces, _}
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamCode, SStruct}
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.virtual._
import is.hail.utils._

import scala.language.{existentials, higherKinds}

case class EmitStreamContext(mb: EmitMethodBuilder[_], ectx: ExecuteContext)

abstract class COption[+A] { self =>
  def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl]

  def cases(
    mb: EmitMethodBuilder[_], ctx: ExecuteContext
  )(none: Code[Unit], some: A => Code[Unit]
  )(implicit line: LineNumber
  ): Code[Unit] = {
    implicit val sctx: EmitStreamContext = EmitStreamContext(mb, ctx)
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

  def addSetup(f: Code[Unit])(implicit line: LineNumber): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      Code(f, self.apply(none, some))
  }

  def doIfNone(f: Code[Unit])(implicit line: LineNumber): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] =
      self.apply(Code(f, none), some)
  }

  def flatMap[B](f: A => COption[B])(implicit line: LineNumber): COption[B] = new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      self(Code(L, none), f(_).apply(L.goto, some))
    }
  }

  def filter(cond: Code[Boolean])(implicit line: LineNumber): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      self(Code(L, none), (a) => cond.mux(L.goto, some(a)))
    }
  }

  def flatMapCPS[B](f: (A, EmitStreamContext, COption[B] => Code[Ctrl]) => Code[Ctrl])(implicit line: LineNumber): COption[B] = new COption[B] {
    def apply(none: Code[Ctrl], some: B => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      val L = CodeLabel()
      self(Code(L, none), f(_, ctx, (b) => b(L.goto, some)))
    }
  }
}

object COption {
  def apply[A](missing: Code[Boolean], value: A)(implicit line: LineNumber): COption[A] = new COption[A] {
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

  def lift[A](opts: IndexedSeq[COption[A]])(implicit line: LineNumber): COption[IndexedSeq[A]] =
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
  def choose[A](useLeft: Code[Boolean], left: COption[A], right: COption[A], fuse: (A, A) => A)(implicit line: LineNumber): COption[A] = new COption[A] {
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

  def fromEmitCode(et: EmitCode)(implicit line: LineNumber): COption[PCode] = new COption[PCode] {
    def apply(none: Code[Ctrl], some: PCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      Code(et.setup, et.m.mux(none, some(et.pv)))
    }
  }

  def toEmitCode(ctx: ExecuteContext, opt: COption[PCode], mb: EmitMethodBuilder[_])(implicit line: LineNumber): EmitCode = {
    implicit val sctx = EmitStreamContext(mb, ctx)
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

  def fold(ctx: ExecuteContext, mb: EmitMethodBuilder[_], init: => Code[Unit], f: (A) => Code[Unit], ret: => Code[Ctrl])(implicit line: LineNumber): Code[Ctrl] = {
    implicit val sctx = EmitStreamContext(mb, ctx)
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

  def forEachCPS(ctx: ExecuteContext, mb: EmitMethodBuilder[_], f: (A, Code[Ctrl]) => Code[Ctrl])(implicit line: LineNumber): Code[Unit] =
    mapCPS[Unit]((_, a, k) => f(a, k(()))).run(ctx, mb)

  def forEach(ctx: ExecuteContext, mb: EmitMethodBuilder[_], f: A => Code[Unit])(implicit line: LineNumber): Code[Unit] =
    mapCPS[Unit]((_, a, k) => Code(f(a), k(()))).run(ctx, mb)

  def forEachI(ctx: ExecuteContext, cb: EmitCodeBuilder, f: A => Unit): Unit = {
    implicit val line = cb.lineNumber
    val savedCode = cb.code
    cb.code = Code._empty
    val streamCode = forEach(ctx, cb.emb, a => { f(a); cb.code })
    cb.code = Code(savedCode, streamCode)
  }

  def run(ctx: ExecuteContext, mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Unit] = {
    implicit val sctx = EmitStreamContext(mb, ctx)
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
  )(implicit line: LineNumber
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
  )(implicit line: LineNumber
  ): Stream[B] =
    mapCPS((_, a, k) => k(f(a)), setup0, setup, close0, close)

  def addSetup(setup: Code[Unit])(implicit line: LineNumber) = map(x => x, setup = Some(setup))

  def flatMap[B](f: A => Stream[B])(implicit line: LineNumber): Stream[B] =
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

  def iota(mb: EmitMethodBuilder[_], start: Code[Int], step: Code[Int])(implicit line: LineNumber): Stream[Code[Int]] = {
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

  def iotaL(mb: EmitMethodBuilder[_], start: Code[Long], step: Code[Int])(implicit line: LineNumber): Stream[Code[Long]] = {
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

  def range(mb: EmitMethodBuilder[_], start: Code[Int], step: Code[Int], len: Code[Int])(implicit line: LineNumber): Stream[Code[Int]] =
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

  def grouped[A](
    mb: EmitMethodBuilder[_],
    childStream: ChildStagedRegion => Stream[A],
    innerStreamType: PStream,
    size: Code[Int],
    eltRegion: ChildStagedRegion
  )(implicit line: LineNumber
  ): Stream[ChildStagedRegion => Stream[A]] = new Stream[ChildStagedRegion => Stream[A]] {
    def apply(outerEos: Code[Ctrl], outerPush: (ChildStagedRegion => Stream[A]) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[ChildStagedRegion => Stream[A]] = {
      val xCounter = ctx.mb.genFieldThisRef[Int]("st_grp_ctr")
      val xInOuter = ctx.mb.genFieldThisRef[Boolean]("st_grp_io")
      val xSize = ctx.mb.genFieldThisRef[Int]("st_grp_sz")
      val LchildPull = CodeLabel()
      val LouterPush = CodeLabel()

      // Need to be able to free the memory used by a child stream element
      // when the outer stream advances before all inner stream elements
      // are consumed.
      var childEltRegion: OwnedStagedRegion = null

      var childSource: Source[A] = null
      val inner = (innerEltRegion: ChildStagedRegion) => new Stream[A] {
        def apply(innerEos: Code[Ctrl], innerPush: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
          childEltRegion = innerEltRegion.createSiblingRegion(mb)
          val LinnerEos = CodeLabel()
          val LinnerPush = CodeLabel()

          childSource = childStream(childEltRegion)(
            xInOuter.mux(outerEos, LinnerEos.goto),
            { a =>
              Code(LinnerPush, childEltRegion.giveToSibling(innerEltRegion), innerPush(a))

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

        val innerEltRegion = eltRegion
          .asParent(innerStreamType.separateRegions, "StreamGrouped inner")
          .createChildRegion(mb)

        // LinnerPush is never executed; childEltRegion is cleared every element.
        val unusedInnerSource = inner(innerEltRegion)(Lunreachable.goto, _ => Lunreachable.goto)
      }

      Code(LchildPull, childSource.pull)

      Source[StagedRegion => Stream[A]](
        setup0 = Code(childSource.setup0, childEltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
        close0 = Code(childEltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()), childSource.close0),
        setup = Code(
          childSource.setup,
          xSize := size,
          xCounter := xSize),
        close = childSource.close,
        pull = Code(xInOuter := true, LchildPull.goto))
    }
  }

  implicit class StreamStream[A](val outer: Stream[Stream[A]]) extends AnyVal {
    def flatten(implicit line: LineNumber): Stream[A] = new Stream[A] {
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
    def flatten(implicit line: LineNumber): Stream[A] = new Stream[A] {
      def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
        val Lpull = CodeLabel()
        val source = stream(
          eos = eos,
          push = _.apply(none = Lpull.goto, some = push))
        source.copy(pull = Code(Lpull, source.pull))
      }
    }

    def take(implicit line: LineNumber): Stream[A] = new Stream[A] {
      def apply(eos: Code[Ctrl], push: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
        val Leos = CodeLabel()
        stream(
          eos = Code(Leos, eos),
          push = _.apply(none = Leos.goto, some = push)).asInstanceOf[Source[A]]
      }
    }
  }

  def zip[A, B](left: Stream[A], right: Stream[B])(implicit line: LineNumber): Stream[(A, B)] = new Stream[(A, B)] {
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

  def multiZip[A](streams: IndexedSeq[Stream[A]])(implicit line: LineNumber): Stream[IndexedSeq[A]] = new Stream[IndexedSeq[A]] {
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
    lElemType: PType, mkLeft: ChildStagedRegion => Stream[EmitCode],
    rElemType: PType, mkRight: ChildStagedRegion => Stream[EmitCode],
    destRegion: ChildStagedRegion,
    comp: (EmitValue, EmitValue) => Code[Int]
  )(implicit line: LineNumber
  ): Stream[(EmitCode, EmitCode)] = new Stream[(EmitCode, EmitCode)] {
    def apply(eos: Code[Ctrl], push: ((EmitCode, EmitCode)) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[(EmitCode, EmitCode)] = {
      val pulledRight = mb.genFieldThisRef[Boolean]()
      val rightEOS = mb.genFieldThisRef[Boolean]()
      val lx = mb.newEmitField(lElemType) // last value received from left
      val rx = mb.newEmitField(rElemType) // last value received from right
      val rxOut = mb.newEmitField(rElemType.setRequired(false)) // right value to push (may be missing while rx is not)
      val rightRegion = destRegion.createSiblingRegion(mb)

      var rightSource: Source[EmitCode] = null
      val leftSource = mkLeft(destRegion).apply(
        eos = eos,
        push = a => {
          val Lpush = CodeLabel()
          val LpullRight = CodeLabel()
          val Lcompare = CodeLabel()

          rightSource = mkRight(rightRegion).apply(
            eos = EmitCodeBuilder.scopedVoid(mb){cb =>
              cb.assign(rightEOS, true)
              cb.assign(rxOut, EmitCode.missing(rElemType))
              cb.append(Lpush.goto)
            },
            push = b => EmitCodeBuilder.scopedVoid(mb)({ cb =>
              rx.store(cb, b)
              cb.append(Lcompare.goto) }))

          EmitCodeBuilder.scopedVoid(mb) ({ cb =>
            cb.append(Lcompare)
            val c = cb.newField[Int]("left_join_right_distinct_c")
            cb.assign(c, comp(lx, rx))
            cb.ifx(c > 0,
              {
                cb.append(rightRegion.clear())
                cb.append(LpullRight.goto)
              },
              {
                cb.ifx(c < 0,
                {
                  cb.assign(rxOut, EmitCode.missing(rElemType))
                  cb.append(Lpush.goto)
                },
                {
                  cb.append(rightRegion.shareWithSibling(destRegion))
                  cb.assign(rxOut, rx)
                  cb.append(Lpush.goto)
                })
              }
            )
          })
          Code(Lpush, push((lx, rxOut)))
          Code(LpullRight, rightSource.pull)

          EmitCodeBuilder.scopedVoid(mb) ({ cb =>
            cb.assign(lx, a)
            cb.ifx(pulledRight,
              cb.ifx(rightEOS,
                cb.append(Lpush.goto),
                cb.append(Lcompare.goto)),
              {
                cb.assign(pulledRight, true)
                cb.append(LpullRight.goto)
              })
          })
        })

      Source[(EmitCode, EmitCode)](
        setup0 = Code(leftSource.setup0, rightSource.setup0, rightRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
        close0 = Code(rightRegion.free(), leftSource.close0, rightSource.close0),
        setup = Code(pulledRight := false, rightEOS := false, leftSource.setup, rightSource.setup),
        close = Code(leftSource.close, rightSource.close),
        pull = leftSource.pull)
    }
  }

  def merge(
    mb: EmitMethodBuilder[_],
    lElemType: PType, mkLeft: ChildStagedRegion => Stream[EmitCode],
    rElemType: PType, mkRight: ChildStagedRegion => Stream[EmitCode],
    outElemType: PType, destRegion: ChildStagedRegion,
    comp: (EmitValue, EmitValue) => Code[Int]
  )(implicit line: LineNumber
  ): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val pulledRight = mb.genFieldThisRef[Boolean]()
      val rightEOS = mb.genFieldThisRef[Boolean]()
      val leftEOS = mb.genFieldThisRef[Boolean]()
      val lx = mb.newEmitField(lElemType) // last value received from left
      val rx = mb.newEmitField(rElemType) // last value received from right
      val leftRegion = destRegion.createSiblingRegion(mb)
      val rightRegion = destRegion.createSiblingRegion(mb)
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
            push = b => EmitCodeBuilder.scopedVoid(mb) { cb =>
              cb.assign(rx, b)
              cb.ifx(leftEOS, cb.append(Lpush.goto), cb.append(Lcompare.goto))
            })

          Code(Lpush,
            // Push smaller of 'lx' and 'rx', with 'lx' breaking ties.
            EmitCodeBuilder.scopedVoid(mb) { cb =>
              cb.ifx(c <= 0,
                {
                  cb.assign(outx, lx.castTo(mb, destRegion.code, outElemType))
                  cb.append(leftRegion.giveToSibling(destRegion))
                },
                {
                  cb.assign(outx, rx.castTo(mb, destRegion.code, outElemType))
                  cb.append(rightRegion.giveToSibling(destRegion))
                }
              )
            },
            push(outx))
          Code(LpullRight, rightSource.pull)

          EmitCodeBuilder.scopedVoid(mb) { cb =>
            cb.assign(lx, a)
            // If this was the first pull, still need to pull from 'right.
            // Otherwise, if 'right' has ended, we know 'c' == -1, so jumping
            // to 'Lpush' will push 'lx'. If 'right' has not ended, compare 'lx'
            // and 'rx' and push smaller.

            cb.ifx(pulledRight,
              cb.ifx(rightEOS, cb.append(Lpush.goto), cb.append(Lcompare.goto)),
              {
                cb.assign(pulledRight, true)
                cb.append(LpullRight.goto)
              })
          }
        })

      Source[EmitCode](
        setup0 = Code(leftSource.setup0,
          rightSource.setup0,
          leftRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()),
          rightRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
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
    lElemType: PType, mkLeft: ChildStagedRegion => Stream[EmitCode],
    rElemType: PType, mkRight: ChildStagedRegion => Stream[EmitCode],
    destRegion: ChildStagedRegion,
    comp: (EmitValue, EmitValue) => Code[Int]
  )(implicit line: LineNumber
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
      val leftRegion = destRegion.createSiblingRegion(mb)
      val rightRegion = destRegion.createSiblingRegion(mb)

      val Leos = CodeLabel()
      Code(Leos, eos)
      val LpullRight = CodeLabel()
      val LpullLeft = CodeLabel()
      val Lpush = CodeLabel()

      var rightSource: Source[EmitCode] = null
      val leftSource = mkLeft(leftRegion)(
        eos = rightEOS.mux(
          Leos.goto,
          Code(
            leftEOS := true,
            lOutMissing := true,
            rOutMissing := false,
            (pulledRight && c.cne(0)).mux(
              Code(rightRegion.shareWithSibling(destRegion), Lpush.goto),
              Code(pulledRight := true, rightRegion.clear(), LpullRight.goto)))),
        push = a => {
          val Lcompare = CodeLabel()

          Code(Lcompare,
            c := comp(lx, rx),
            lOutMissing := false,
            rOutMissing := false,
            (c > 0).mux(
              pulledRight.mux(
                Code(lOutMissing := true,
                     rightRegion.shareWithSibling(destRegion),
                     Lpush.goto),
                Code(pulledRight := true,
                     LpullRight.goto)),
              (c < 0).mux(
                Code(rOutMissing := true,
                     leftRegion.giveToSibling(destRegion),
                     Lpush.goto),
                Code(pulledRight := true,
                     leftRegion.giveToSibling(destRegion),
                     rightRegion.shareWithSibling(destRegion),
                     Lpush.goto))))

          rightSource = mkRight(rightRegion).apply(
            eos = leftEOS.mux(
              Leos.goto,
              Code(rightEOS := true,
                   lOutMissing := false,
                   rOutMissing := true,
                   leftRegion.giveToSibling(destRegion),
                   Lpush.goto)),
            push = b => EmitCodeBuilder.scopedVoid(mb) { cb =>
              cb.assign(rx, b)
              cb.ifx(leftEOS, cb.append(Lpush.goto), cb.append(Lcompare.goto))
            })

          Code(Lpush, push((lx.missingIf(mb, lOutMissing), rx.missingIf(mb, rOutMissing))))
          Code(LpullRight, rightSource.pull)

          EmitCodeBuilder.scopedVoid(mb) { cb =>
            cb.assign(lx, a)
            cb.ifx(pulledRight,
              cb.ifx(rightEOS,
                {
                  cb.append(leftRegion.giveToSibling(destRegion))
                  cb.append(Lpush.goto)
                },
                {
                  cb.ifx(c.ceq(0),
                    cb.assign(pulledRight, false))
                  cb.append(Lcompare.goto)
                }
              ),
              {
                cb.assign(pulledRight, true)
                cb.append(LpullRight.goto)
              })
          }
        })

      Code(LpullLeft, leftSource.pull)

      Source[(EmitCode, EmitCode)](
        setup0 = Code(leftSource.setup0, rightSource.setup0,
                      leftRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()),
                      rightRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
        close0 = Code(leftRegion.free(), rightRegion.free(),
                      leftSource.close0, rightSource.close0),
        setup = Code(pulledRight := false, c := 0,
                     leftEOS := false, rightEOS := false,
                     leftSource.setup, rightSource.setup),
        close = Code(leftSource.close, rightSource.close),
        pull = leftEOS.mux(
          Code(rightRegion.clear(), LpullRight.goto),
          rightEOS.mux(LpullLeft.goto,
                       (c <= 0).mux(LpullLeft.goto,
                                    Code(rightRegion.clear(), LpullRight.goto)))))
    }
  }

  def kWayMerge[A: TypeInfo](
    mb: EmitMethodBuilder[_],
    streams: IndexedSeq[ChildStagedRegion => Stream[Code[A]]],
    destRegion: ChildStagedRegion,
    // compare two (idx, value) pairs, where 'value' is a value from the 'idx'th
    // stream
    lt: (Code[Int], Code[A], Code[Int], Code[A]) => Code[Boolean]
  )(implicit line: LineNumber
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
      val eltRegions = destRegion.createSiblingRegionArray(mb, k)

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
            Code(eltRegions(winner).giveToSibling(destRegion), push((winner, heads(winner))))),
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

trait StreamArgType {
  def apply(outerRegion: Region, eltRegion: Region): Iterator[java.lang.Long]
}

object EmitStream {

  import Stream._

  def write(
    ctx: ExecuteContext,
    mb: EmitMethodBuilder[_],
    pcStream: SStreamCode,
    ab: StagedArrayBuilder,
    destRegion: ParentStagedRegion
  )(implicit line: LineNumber
  ): Code[Unit] = {
    _write(ctx, mb, pcStream.stream, ab, destRegion)
  }

  private def _write(
    ctx: ExecuteContext,
    mb: EmitMethodBuilder[_],
    sstream: SizedStream,
    ab: StagedArrayBuilder,
    destRegion: ParentStagedRegion
  )(implicit line: LineNumber
  ): Code[Unit] = {
    val SizedStream(ssSetup, stream, optLen) = sstream
    val eltRegion = destRegion.createChildRegion(mb)
    Code(FastSeq(
      eltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()),
      ssSetup,
      ab.clear,
      ab.ensureCapacity(optLen.getOrElse(16)),
      stream(eltRegion).forEach(ctx, mb, { elt => Code(
        elt.setup,
        elt.m.mux(
          ab.addMissing(),
          EmitCodeBuilder.scopedVoid(mb)(cb => cb += ab.add(eltRegion.copyToParent(cb, elt.pv).code))),
        eltRegion.clear())
      }),
      eltRegion.free()))
  }

  def toArray(
    ctx: ExecuteContext,
    mb: EmitMethodBuilder[_],
    aTyp: PArray,
    pcStream: SStreamCode,
    destRegion: ParentStagedRegion
  )(implicit line: LineNumber
  ): PCode = {
    val srvb = new StagedRegionValueBuilder(mb, aTyp, destRegion.code)
    val ss = pcStream.stream
    ss.length match {
      case None =>
        val xLen = mb.newLocal[Int]("sta_len")
        val i = mb.newLocal[Int]("sta_i")
        val vab = new StagedArrayBuilder(aTyp.elementType, mb, 0)
        val ptr = Code(
          _write(ctx, mb, ss, vab, destRegion),
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
            eltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()),
            ss.setup,
            srvb.start(len),
            ss.stream(eltRegion).forEach(ctx, mb, { et =>
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

  def sequence(mb: EmitMethodBuilder[_], elemPType: PType, elements: IndexedSeq[EmitCode]
  )(implicit line: LineNumber
  ): Stream[EmitCode] = new Stream[EmitCode] {
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
            Code.switch(i, Leos.goto, elements.map(elem => EmitCodeBuilder.scopedVoid(mb){ cb => cb.assign(t, elem); cb.append(Lpush.goto)})),
            Lpush,
            i := i + 1,
            push(t)),
          Code(Leos, eos)))
    }
  }

  case class SizedStream(setup: Code[Unit], stream: ChildStagedRegion => Stream[EmitCode], length: Option[Code[Int]]) {
    def getStream(eltRegion: ChildStagedRegion)(implicit line: LineNumber): Stream[EmitCode] =
      stream(eltRegion).addSetup(setup)
  }

  object SizedStream {
    def unsized(stream: ChildStagedRegion => Stream[EmitCode]): SizedStream =
      SizedStream(Code._empty, stream, None)
  }

  def mux(
    mb: EmitMethodBuilder[_], eltType: PType, cond: Code[Boolean], left: Stream[EmitCode], right: Stream[EmitCode]
  )(implicit line: LineNumber
  ): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val b = mb.genFieldThisRef[Boolean]()
      val Leos = CodeLabel()
      val elt = mb.newEmitField("stream_mux_elt", eltType)
      val Lpush = CodeLabel()

      val l = left(Code(Leos, eos), (a) => Code(EmitCodeBuilder.scopedVoid(mb)(_.assign(elt, a)), Lpush, push(elt)))
      val r = right(Leos.goto, (a) => Code(EmitCodeBuilder.scopedVoid(mb)(_.assign(elt, a)), Lpush.goto))

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
    streams: IndexedSeq[ChildStagedRegion => Stream[PCode]],
    destRegion: ChildStagedRegion,
    resultType: PArray,
    key: IndexedSeq[String]
  )(implicit line: LineNumber
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
      val eltRegions = destRegion.createSiblingRegionArray(mb, k)

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

      val winnerPc = PCode(keyViewType, heads(winner))

      Code(LstartNewKey,
        Code.forLoop(i := 0, i < k, i := i + 1, result(i) = 0L),
        EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(curKey, eltRegions(winner).copyTo(cb, winnerPc, destRegion, keyType))),
        LaddToResult.goto)

      Code(LaddToResult,
        result(winner) = heads(winner),
        eltRegions(winner).giveToSibling(destRegion),
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
            heads(idx) = EmitCodeBuilder.scopedCode(mb)(cb => elt.castTo(cb, eltRegions(idx).code, eltType).tcode[Long]),
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
    stream: ChildStagedRegion => Stream[PCode],
    innerStreamType: PStream,
    key: Array[String],
    eltRegion: ChildStagedRegion
  )(implicit line: LineNumber
  ): Stream[ChildStagedRegion => Stream[PCode]] = new Stream[ChildStagedRegion => Stream[PCode]] {
    def apply(outerEos: Code[Ctrl], outerPush: (ChildStagedRegion => Stream[PCode]) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[ChildStagedRegion => Stream[PCode]] = {
      val eltType = coerce[PStruct](innerStreamType.elementType)
      val keyType = eltType.selectFields(key)
      val keyViewType = SSubsetStruct(eltType.sType.asInstanceOf[SStruct], key)
      val ordering = keyType.codeOrdering(mb, keyViewType.pType, missingFieldsEqual = false).asInstanceOf[CodeOrdering { type T = Long }]

      val xCurKey = ctx.mb.newPField("st_grpby_curkey", keyType)
      val xCurElt = ctx.mb.newPField("st_grpby_curelt", eltType)
      val xInOuter = ctx.mb.genFieldThisRef[Boolean]("st_grpby_io")
      val xEOS = ctx.mb.genFieldThisRef[Boolean]("st_grpby_eos")
      val xNextGrpReady = ctx.mb.genFieldThisRef[Boolean]("st_grpby_ngr")

      var holdingRegion: OwnedStagedRegion = null
      var keyRegion: OwnedStagedRegion = null

      val LchildPull = CodeLabel()
      val LouterPush = CodeLabel()
      val LouterEos = CodeLabel()

      var childSource: Source[PCode] = null
      val inner = (innerEltRegion: ChildStagedRegion) => new Stream[PCode] {
        def apply(innerEos: Code[Ctrl], innerPush: PCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[PCode] = {
          holdingRegion = innerEltRegion.createSiblingRegion(mb)
          keyRegion = innerEltRegion.createSiblingRegion(mb)

          val LinnerEos = CodeLabel()
          val LinnerPush = CodeLabel()

          childSource = stream(holdingRegion).apply(
            xInOuter.mux(LouterEos.goto, Code(xEOS := true, LinnerEos.goto)),
            { a: PCode =>
              Code(
                EmitCodeBuilder.scopedVoid(mb)(_.assign(xCurElt, a)),
                // !xInOuter iff this element was requested by an inner stream.
                // Else we are stepping to the beginning of the next group.
                (xCurKey.tcode[Long].cne(0L) && ordering.equivNonnull(xCurKey.tcode[Long], xCurElt.tcode[Long])).mux(
                  xInOuter.mux(
                    Code(holdingRegion.clear(), LchildPull.goto),
                    LinnerPush.goto),
                  Code(
                    keyRegion.clear(),
                    EmitCodeBuilder.scopedVoid(mb) { cb =>
                      val pc = new SSubsetStructCode(keyViewType, xCurElt.load().asBaseStruct)
                      cb.assign(xCurKey, pc.castTo(cb, keyRegion.code, keyType))},
                    xInOuter.mux(
                      LouterPush.goto,
                      Code(xNextGrpReady := true, LinnerEos.goto)))))
            })

          Code(LinnerPush, holdingRegion.giveToSibling(innerEltRegion), innerPush(xCurElt))
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

        val innerEltRegion = eltRegion
          .asParent(innerStreamType.separateRegions, "StreamGroupByKey inner")
          .createChildRegion(mb)

        // LinnerPush is never executed; holdingRegion will be cleared every
        // element, and keyRegion cleared every new key.
        val unusedInnerSource = inner(innerEltRegion)(Lunreachable.goto, _ => Lunreachable.goto)
      }

      // Precondition: holdingRegion is empty
      Code(LchildPull, childSource.pull)

      Code(LouterEos, outerEos)

      Source[ChildStagedRegion => Stream[PCode]](
        setup0 = Code(childSource.setup0, holdingRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()), keyRegion.allocateRegion(Region.TINIER, mb.ecb.pool())),
        close0 = Code(holdingRegion.free(), keyRegion.free(), childSource.close0),
        setup = Code(
          childSource.setup,
          EmitCodeBuilder.scopedVoid(mb)(_.assign(xCurKey, keyType.defaultValue)),
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

  def extendNA(
    mb: EmitMethodBuilder[_], eltType: PType, stream: Stream[EmitCode]
  )(implicit line: LineNumber
  ): Stream[COption[EmitCode]] = new Stream[COption[EmitCode]] {
    def apply(eos: Code[Ctrl], push: COption[EmitCode] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[COption[EmitCode]] = {
      val atEnd = mb.genFieldThisRef[Boolean]()
      val x = mb.newEmitField(eltType)
      val Lpush = CodeLabel()
      val source = stream(Code(atEnd := true, Lpush.goto), a => Code(EmitCodeBuilder.scopedVoid(mb)(_.assign(x, a)), Lpush, push(COption(atEnd.get, x.load))))
      Source[COption[EmitCode]](
        setup0 = source.setup0,
        close0 = source.close0,
        setup = Code(atEnd := false, source.setup),
        close = source.close,
        pull = atEnd.get.mux(Lpush.goto, source.pull))
    }
  }

  private[ir] def emit[C](
    ctx: ExecuteContext,
    emitter: Emit[C],
    streamIR0: IR,
    mb: EmitMethodBuilder[C],
    outerRegion: ParentStagedRegion,
    env0: Emit.E,
    container: Option[AggContainer]
  ): EmitCode = {

    def _emitStream(streamIR: IR, outerRegion: ParentStagedRegion, env: Emit.E): COption[SizedStream] = {

      implicit val line = LineNumber(streamIR.lineNumber)

      def emitStream(streamIR: IR, outerRegion: ParentStagedRegion = outerRegion, env: Emit.E = env): COption[SizedStream] =
        _emitStream(streamIR, outerRegion, env)

      def emitStreamToEmitCode(streamIR: IR, outerRegion: ParentStagedRegion = outerRegion, env: Emit.E = env): EmitCode =
        COption.toEmitCode(ctx,
          _emitStream(streamIR, outerRegion, env).map { stream =>
            SStreamCode(streamIR.pType.sType.asInstanceOf[SStream], stream)
          }, mb)

      def emitIR(ir: IR, env: Emit.E = env, region: StagedRegion = outerRegion, container: Option[AggContainer] = container): EmitCode =
        emitter.emitWithRegion(ir, mb, region, env, container)

      def emitVoidIR(ir: IR, env: Emit.E = env, region: StagedRegion = outerRegion, container: Option[AggContainer] = container): Code[Unit] = {
        EmitCodeBuilder.scopedVoid(mb) { cb =>
          emitter.emitVoid(cb, ir, mb, region, env, container, None)
        }
      }

      def sized(setup: Code[Unit], stream: ChildStagedRegion => Stream[EmitCode], length: Option[Code[Int]], outerRegion: ParentStagedRegion = outerRegion): SizedStream =
        SizedStream(setup, r => { r assertSubRegion outerRegion; stream(r) }, length)

      def unsized(stream: ChildStagedRegion => Stream[EmitCode], outerRegion: ParentStagedRegion = outerRegion): SizedStream =
        SizedStream.unsized(r => { r assertSubRegion outerRegion; stream(r) })

      streamIR match {
        case x@NA(_) =>
          COption.none(coerce[PCanonicalStream](x.pType).defaultValue.stream)

        case x@Ref(name, _) =>
          val typ = coerce[PStream](x.pType)
          val ev = env.lookup(name)
          if (ev.pt != typ)
            throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $typ")
          COption.fromEmitCode(ev.load).map(_.asStream.stream)

        case x@StreamRange(startIR, stopIR, stepIR, _) =>
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
                      some(sized(
                        len := llen.toI,
                        eltRegion => range(mb, start, step, len)
                          .map(i => EmitCode(Code._empty, const(false), PCode(eltType, i))),
                        Some(len)))))))
            }
          }

        case ToStream(containerIR, _) =>
          COption.fromEmitCode(emitIR(containerIR)).mapCPS { (containerAddr, k) =>
            val (asetup, a) = EmitCodeBuilder.scoped(mb) { cb =>
              containerAddr.asIndexable.memoizeField(cb, "ts_a")
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
                          a.loadElement(cb, i - 1).typecast[PCode]
                        })),
                    eos))
            }

            Code(
              asetup,
              len := a.loadLength(),
              k(sized(Code._empty, eltRegion => newStream, Some(len))))
          }

        case x@MakeStream(elements, _, _) =>
          val eltType = coerce[PStream](x.pType).elementType
          val stream = (eltRegion: StagedRegion) =>
            sequence(mb, eltType, elements.toFastIndexedSeq.map { ir =>
              emitIR(ir, region = eltRegion)
                .castTo(mb, eltRegion.code, eltType)
            })

          COption.present(sized(Code._empty, stream, Some(elements.length)))

        case x@ReadPartition(context, rowType, reader) =>
          reader.emitStream(ctx, context, rowType, emitter, mb, outerRegion, env, container)

        case In(n, PCanonicalStream(eltType, _, _)) =>
          val xIter = mb.genFieldThisRef[Iterator[java.lang.Long]]("streamInIterator")
          val hasNext = mb.genFieldThisRef[Boolean]("streamInHasNext")
          val next = mb.genFieldThisRef[Long]("streamInNext")

          // this, Region, ...
          mb.getStreamEmitParam(2 + n).map { mkIter =>
            unsized { eltRegion =>
              unfold[Code[Long]](
                (_, k) => Code(
                  hasNext := xIter.load().hasNext,
                  hasNext.orEmpty(next := xIter.load().next().invoke[Long]("longValue")),
                  k(COption(!hasNext, next)))
                ).map(
                rv => EmitCodeBuilder.scopedEmitCode(mb)(cb => EmitCode.present(eltType.loadCheapPCode(cb, (rv)))),
                setup0 = None,
                setup = Some(
                  xIter := mkIter.invoke[Region, Region, Iterator[java.lang.Long]](
                    "apply", outerRegion.code, eltRegion.code))
                )
            }
          }

        case StreamTake(a, num) =>
          val optStream = emitStream(a)
          val optN = COption.fromEmitCode(emitIR(num))
          val xN = mb.genFieldThisRef[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optN.map { n => sized(
              Code(setup,
                   EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xN, n.asInt.intCode(cb))),
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
            optN.map { n => sized(
              Code(setup,
                EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xN, n.asInt.intCode(cb))),
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
              val newStream = (eltRegion: ChildStagedRegion) =>
                Stream.grouped(mb, stream, innerType, xS, eltRegion)
                  .map { inner =>
                    EmitCode(
                      Code._empty,
                      false,
                      interfaces.SStreamCode(
                        innerType.sType,
                        unsized(inner)))
                  }
              sized(
                Code(setup,
                  EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xS, s.asInt.intCode(cb))),
                  (xS <= 0).orEmpty(Code._fatal[Unit](const("StreamGrouped: nonpositive size")))),
                newStream,
                len.map(l => ((l.toL + xS.toL - 1L) / xS.toL).toI)) // rounding up integer division
            }
          }

        case x@StreamGroupByKey(a, key) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          val optStream = emitStream(a)
          optStream.map { ss =>
            val nonMissingStream = (eltRegion: ChildStagedRegion) => ss.getStream(eltRegion).mapCPS[PCode] { (_, ec, k) =>
              Code(ec.setup, ec.m.orEmpty(Code._fatal[Unit](const("expected non-missing"))), k(ec.pv))
            }
            val newStream = (eltRegion: ChildStagedRegion) =>
              groupBy(mb, nonMissingStream, innerType, key.toArray, eltRegion)
                .map { inner =>
                  EmitCode.present(
                    interfaces.SStreamCode(
                      innerType.sType,
                      unsized { innerEltRegion =>
                        inner(innerEltRegion).map(EmitCode.present)
                      }))
                }
            unsized(newStream)
          }

        case StreamMap(childIR, name, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR)
          optStream.map { case SizedStream(setup, stream, len) =>
            def newStream(eltRegion: ChildStagedRegion) = stream(eltRegion).map { eltt => (eltType, bodyIR.pType) match {
              case (eltType: PCanonicalStream, bodyType: PCanonicalStream) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))
                val outerRegion = eltRegion.asParent(bodyType.separateRegions, "StreamMap body")

                emitStreamToEmitCode(bodyIR, outerRegion, bodyenv)
              case (eltType: PCanonicalStream, _) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))

                emitIR(bodyIR, region = eltRegion, env = bodyenv)
              case (_, bodyType: PCanonicalStream) =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)
                val outerRegion = eltRegion.asParent(bodyType.separateRegions, "StreamMap body")

                EmitCode(
                  EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xElt, eltt)),
                  emitStreamToEmitCode(bodyIR, outerRegion, env = bodyenv))
              case _ =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)
                val bodyt = emitIR(bodyIR, region = eltRegion, env = bodyenv)

                EmitCodeBuilder.scopedEmitCode(mb){ cb =>
                  cb.assign(xElt, eltt)
                  bodyt
                }
            }}

            sized(setup, newStream, len)
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR)

          optStream.map { ss =>
            val newStream = (eltRegion: ChildStagedRegion) => {
              val tmpRegion = eltRegion.createSiblingRegion(mb)
              ss.getStream(tmpRegion)
                .map (
                  { elt =>
                    val xElt = mb.newEmitField(name, childEltType)
                    val cond = emitIR(condIR, env = env.bind(name -> xElt), region = tmpRegion)

                    new COption[EmitCode] {
                      def apply(none: Code[Ctrl], some: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
                        Code(
                          EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xElt, elt)),
                          cond.setup,
                          (cond.m || !cond.value[Boolean]).mux(
                            Code(tmpRegion.clear(), none),
                            some(EmitCode.fromI(mb) { cb =>
                              xElt.toI(cb)
                                .mapMissing(cb) { cb += tmpRegion.clear() }
                                .map(cb) { pc =>
                                  cb += tmpRegion.giveToSibling(eltRegion)
                                  pc
                                }
                            })))
                      }
                    }
                  },
                  setup0 = Some(tmpRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
                  close0 = Some(tmpRegion.free()))
                .flatten
            }

            unsized(newStream)
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
              sized(
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
            val lengths = emitStreams.map(_.length)

            behavior match {

              case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
                val newStream = (eltRegion: ChildStagedRegion) =>
                  multiZip(emitStreams.map(_.stream(eltRegion)))
                    .map { elts =>
                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      val body = emitIR(bodyIR, env = bodyEnv, region = eltRegion)
                      EmitCode(Code(Code((eltVars, elts).zipped.map { (v, x) => EmitCodeBuilder.scopedVoid(mb)(_.assign(v, x)) }), body.setup), body.m, body.pv)
                    }
                val newLength = behavior match {
                  case ArrayZipBehavior.TakeMinLength =>
                    lengths.reduceLeft(_.liftedZip(_).map {
                      case (l1, l2) => l1.min(l2)
                    })
                  case ArrayZipBehavior.AssumeSameLength =>
                    lengths.flatten.headOption
                }

                sized(lenSetup, newStream, newLength)

              case ArrayZipBehavior.AssertSameLength =>
                val newStream = (eltRegion: ChildStagedRegion) => {
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
                          optEC.cases(mb, ctx)(
                            anyEOS := true,
                            ec => EmitCodeBuilder.scopedVoid(mb) { cb =>
                              cb.assign(allEOS, false)
                              cb.assign(eltVar, ec)
                            })
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

                sized(lenSetup, newStream, newLength)

              case ArrayZipBehavior.ExtendNA =>
                val newStream = (eltRegion: ChildStagedRegion) => {
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

                          COption.toEmitCode(ctx, optElt, mb)
                        }
                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      val body = emitIR(bodyIR, env = bodyEnv, region = eltRegion)

                      Code(
                        allEOS := true,
                        Code((eltVars, checkedElts).zipped.map { (v, x) => EmitCodeBuilder.scopedVoid(mb)(_.assign(v, x)) }),
                        k(COption(allEOS, body)))
                    }

                  // termininate the stream when all streams are EOS
                  flagged.take
                }

                val newLength = lengths.reduceLeft(_.liftedZip(_).map {
                  case (l1, l2) => l1.max(l2)
                })

                sized(lenSetup, newStream, newLength)
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
            val streams = sss.map { ss => (eltRegion: ChildStagedRegion) =>
              ss.stream(eltRegion).map { ec =>
                EmitCodeBuilder.scopedCode(mb)(cb => ec.get().castTo(cb, outerRegion.code, eltType).tcode[Long])
              }
            }
            sized(
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
                emit(ctx, emitter, joinIR, mb, outerRegion, newEnv, container)
              case _ =>
                emitIR(joinIR, env = newEnv, region = eltRegion)
            }

            EmitCodeBuilder.scopedEmitCode(mb) { cb =>
              cb.assign(xKey, k)
              cb.assign(xElts, vs)
              joint
            }
          }

          COption.lift(as.map(emitStream(_))).map { sss =>
            val streams = sss.map { ss => (eltRegion: ChildStagedRegion) =>
              ss.getStream(eltRegion).map(_.get())
            }
            unsized { eltRegion =>
              kWayZipJoin(mb, streams, eltRegion, curValsType, key)
                .map(joinF(eltRegion))
            }
          }

        case StreamFlatMap(outerIR, name, innerIR) =>
          val outerEltType = coerce[PStream](outerIR.pType).elementType
          val innerStreamType = coerce[PStream](innerIR.pType)

          val optOuter = emitStream(outerIR)
          val outerEltRegion = outerRegion.createChildRegion(mb)

          optOuter.map { outer =>
            val newStream = (eltRegion: ChildStagedRegion) => {
              outer.getStream(outerEltRegion)
                .map[COption[Stream[EmitCode]]] { elt =>
                  val innerStreamOuterRegion =
                    outerEltRegion.asParent(innerStreamType.separateRegions, "StreamFlatMap inner")
                  val optInner = if (outerEltType.isRealizable) {
                    val xElt = mb.newEmitField(name, outerEltType)
                    val innerEnv = env.bind(name -> xElt)
                    val optInner = emitStream(
                      innerIR,
                      outerRegion = innerStreamOuterRegion,
                      env = innerEnv)

                    optInner.addSetup(EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xElt, elt)))
                  } else {
                    val innerEnv = env.bind(name -> new EmitUnrealizableValue(outerEltType, elt))

                    emitStream(
                      innerIR,
                      outerRegion = innerStreamOuterRegion,
                      env = innerEnv)
                  }

                  optInner.map { inner =>
                    // We know that eltRegion is a subregion of innerStreamOuterRegion,
                    // even though the former was constructed before the later.
                    inner.getStream(eltRegion.asSubregionOf(innerStreamOuterRegion))
                      .map(x => x,
                           close = Some(outerEltRegion.clear()))
                  }
                }
                .flatten.flatten
                .map(x => x,
                     setup0 = Some(outerEltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
                     close0 = Some(outerEltRegion.free()))
            }

            unsized(newStream)
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
                  val newStream = (eltRegion: ChildStagedRegion) => mux(mb, eltType,
                    xCond,
                    leftStream(eltRegion),
                    rightStream(eltRegion))
                  val newLen = lLen.liftedZip(rLen).map { case (l1, l2) =>
                    xCond.mux(l1, l2)
                  }
                  val newSetup = xCond.mux(leftSetup, rightSetup)

                  sized(newSetup, newStream, newLen)
              })

            newOptStream.addSetup(xCond := cond.tcode[Boolean])
          }

        case Let(name, valueIR, bodyIR) =>
          val valueType = valueIR.pType

          valueType match {
            case _: PCanonicalStream =>
              val valuet = emit(ctx, emitter, valueIR, mb, outerRegion, env, container)
              val bodyEnv = env.bind(name -> new EmitUnrealizableValue(valueType, valuet))

              emitStream(bodyIR, env = bodyEnv)

            case _ =>
              val xValue = mb.newEmitField(name, valueType)
              val bodyEnv = env.bind(name -> xValue)
              val valuet = emitIR(valueIR)

              emitStream(bodyIR, env = bodyEnv).addSetup(EmitCodeBuilder.scopedVoid(mb)(_.assign(xValue, valuet)))
          }

        case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = x.accPType

          val streamOpt = emitStream(childIR)
          streamOpt.map { case SizedStream(setup, stream, len) =>
            val Lpush = CodeLabel()
            val hasPulled = mb.genFieldThisRef[Boolean]()

            val xElt = mb.newEmitField(eltName, eltType)
            val xAccInEltR = mb.newEmitField(accName, accType)
            val xAccInAccR = mb.newEmitField(accName, accType)


            val newStream = (eltRegion: ChildStagedRegion) => new Stream[EmitCode] {
              def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
                val accRegion = eltRegion.createSiblingRegion(mb)

                Code(Lpush,
                  EmitCodeBuilder.scopedVoid(mb) { cb => cb.assign(xAccInAccR, xAccInEltR.map(eltRegion.copyTo(cb, _, accRegion))) },
                  push(xAccInEltR))

                val bodyEnv = env.bind(accName -> xAccInAccR, eltName -> xElt)
                val body = emitIR(bodyIR, env = bodyEnv, region = eltRegion)
                  .castTo(mb, eltRegion.code, accType)

                val source = stream(eltRegion).apply(
                  eos = eos,
                  push = a => EmitCodeBuilder.scopedVoid(mb) { cb =>
                    cb.assign(xElt, a)
                    cb.assign(xAccInEltR, body)
                    cb.append(accRegion.clear())
                    cb.append(Lpush.goto)
                  })

                Source[EmitCode](
                  setup0 = Code(source.setup0, accRegion.allocateRegion(Region.TINIER, mb.ecb.pool())),
                  setup = Code(hasPulled := false, source.setup),
                  close = source.close,
                  close0 = Code(accRegion.free(), source.close0),
                  pull = hasPulled.mux(
                    source.pull,
                    EmitCodeBuilder.scopedVoid(mb) { cb =>
                      cb.assign(hasPulled, true)
                      cb.assign(xAccInEltR, emitIR(zeroIR, region = eltRegion)
                        .map(_.castTo(cb, eltRegion.code, accType)))
                      cb.append(Lpush.goto)
                    }))
              }
            }

            val newLen = len.map(l => l + 1)
            sized(setup, newStream, newLen)
          }

        case x@RunAggScan(array, name, init, seqs, result, states) =>
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb,"array_agg_scan")

          val eltType = coerce[PStream](array.pType).elementType

          val xElt = mb.newEmitField("aggscan_elt", eltType)
          val xResult = mb.newEmitField("aggscan_result", result.pType)

          val bodyEnv = env.bind(name -> xElt)

          val optStream = emitStream(array)

          optStream.map { case SizedStream(setup, stream, len) =>
            val newStream = (eltRegion: ChildStagedRegion) => {
              val tmpRegion = eltRegion.createSiblingRegion(mb)
              val cInit = emitVoidIR(init, region = tmpRegion, container = Some(newContainer))
              val postt = emitIR(result, region = eltRegion, env = bodyEnv, container = Some(newContainer))
              val seqPerElt = emitVoidIR(seqs, region = eltRegion, env = bodyEnv, container = Some(newContainer))
              stream(eltRegion).map[EmitCode](
                { eltt =>
                  EmitCodeBuilder.scopedEmitCode(mb) { cb =>
                    cb.assign(xElt, eltt)
                    cb.assign(xResult, postt)
                    cb.append(seqPerElt)
                    xResult.load
                  }
                },
                setup0 = Some(aggSetup),
                close0 = Some(aggCleanup),
                setup = Some(Code(tmpRegion.allocateRegion(Region.SMALL, mb.ecb.pool()), cInit, tmpRegion.free())))
            }


            sized(setup, newStream, len)
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
                emit(ctx, emitter, joinIR, mb, outerRegion, newEnv, container)
              case _ =>
                emitIR (joinIR, newEnv)
            }

            EmitCodeBuilder.scopedEmitCode(mb) { cb =>
              cb.assign(xLElt, lelt)
              cb.assign(xRElt, relt)
              joint
            }
          }

          emitStream(leftIR).flatMap { case SizedStream(leftSetup, leftStream, leftLen) =>
            emitStream(rightIR).map { rightSS =>
              val newStream = (eltRegion: ChildStagedRegion) => {
                if (joinType == "left")
                  leftJoinRightDistinct(
                    mb,
                    lEltType, leftStream,
                    rEltType, rightSS.getStream,
                    eltRegion,
                    compare)
                    .map(joinF)
                else
                  outerJoinRightDistinct(
                    mb,
                    lEltType, leftStream,
                    rEltType, rightSS.getStream,
                    eltRegion,
                    compare)
                    .map(joinF)
              }

              sized(leftSetup,
                    newStream,
                    if (joinType == "left") leftLen else None)
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
          ).flatMap { case (idt: SCanonicalShufflePointerCode) =>
            COption.fromEmitCode(emitIR(keyRangeIR)).doIfNone(
              Code._fatal[Unit]("ShuffleRead cannot have null key range")
            ).flatMap { case (keyRangeCode: PIntervalCode) =>

              val uuidLocal = mb.newPLocal("shuffleUUID", idt.st.pType.representation)
                .asInstanceOf[SBinaryPointerSettable]
              val uuid = new SCanonicalShufflePointerSettable(idt.st, uuidLocal)
              val keyRange = SIntervalPointerSettable(mb.fieldBuilder, keyRangeCode.st.asInstanceOf[SIntervalPointer], "shuffleClientKeyRange")
              COption(
                EmitCodeBuilder.scopedCode(mb) { cb =>
                  cb.assign(uuid, idt)
                  cb.assign(keyRange, keyRangeCode)
                  !keyRange.startDefined(cb) || !keyRange.endDefined(cb)
                },
                keyRange
              ).doIfNone(
                Code._fatal[Unit]("ShuffleRead cannot have null start or end points of key range")
              ).map { keyRange =>

                val shuffleLocal = mb.newLocal[ShuffleClient]("shuffleClient")
                val shuffle = new ValueShuffleClient(shuffleLocal)

                val stream = (eltRegion: ChildStagedRegion) => unfold[EmitCode](
                  { (_, k) =>
                    k(
                      COption(
                        shuffle.getValueFinished(),
                        EmitCode.present(
                          shuffleType.rowDecodedPType, shuffle.getValue(eltRegion.code))))
                  },
                  setup = Some(EmitCodeBuilder.scopedVoid(mb) { cb =>
                    cb.assign(shuffleLocal, CodeShuffleClient.create(
                      mb.ecb.getType(shuffleType),
                      uuid.loadBytes(),
                      Code._null,
                      mb.ecb.getPType(keyPType)))

                    val startt = keyRange.loadStart(cb)
                        .handle(cb, cb._fatal("shuffle expects defined endpoints"))
                        .tcode[Long]
                    val endt =
                      keyRange.loadEnd(cb)
                        .handle(cb, cb._fatal("shuffle expects defined endpoints"))
                        .tcode[Long]

                    cb.append(shuffle.startGet(startt, keyRange.includesStart, endt, keyRange.includesEnd))
                  }),
                  close = Some(Code(
                    shuffle.getDone(),
                    shuffle.close())))
                unsized(stream)
              }
            }
          }

      case x@ShufflePartitionBounds(idIR, nPartitionsIR) =>
          val shuffleType = coerce[TShuffle](idIR.typ)
          assert(shuffleType.keyDecodedPType == coerce[PStream](x.pType).elementType)
          COption.fromEmitCode(emitIR(idIR)).doIfNone(
            Code._fatal[Unit]("ShufflePartitionBounds cannot have null ID")
          ).flatMap { case (idt: SCanonicalShufflePointerCode) =>
            COption.fromEmitCode(emitIR(nPartitionsIR)).doIfNone(
              Code._fatal[Unit]("ShufflePartitionBounds cannot have null number of partitions")
            ).map { case (nPartitionst: SInt32Code) =>
              val uuidLocal = mb.newLocal[Long]("shuffleUUID")
              val uuid = new SCanonicalShufflePointerSettable(idt.st, new SBinaryPointerSettable(SBinaryPointer(idt.st.pType.representation), uuidLocal))
              val shuffleLocal = mb.newLocal[ShuffleClient]("shuffleClient")
              val shuffle = new ValueShuffleClient(shuffleLocal)
              val stream = (eltRegion: ChildStagedRegion) => unfold[EmitCode](
                { (_, k) =>
                  k(
                    COption(
                      shuffle.partitionBoundsValueFinished(),
                      EmitCode.present(
                        shuffleType.keyDecodedPType, shuffle.partitionBoundsValue(eltRegion.code))))
                },
                setup = Some(EmitCodeBuilder.scopedVoid(mb) { cb =>
                  uuid.store(cb, idt)
                  cb.assign(shuffleLocal, CodeShuffleClient.create(mb.ecb.getType(shuffleType), uuid.loadBytes()))
                  cb += shuffle.startPartitionBounds(nPartitionst.intCode(cb)) }),
                close = Some(Code(
                  shuffle.endPartitionBounds(),
                  shuffle.close())))
              unsized(stream)
          }
        }

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }
    }

    COption.toEmitCode(ctx,
      _emitStream(streamIR0, outerRegion, env0).map { stream =>
        interfaces.SStreamCode(coerce[PCanonicalStream](streamIR0.pType).sType, stream)
      }, mb)(LineNumber(streamIR0.lineNumber))
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
