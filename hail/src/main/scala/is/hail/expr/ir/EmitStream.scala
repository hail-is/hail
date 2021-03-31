package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.services.shuffler._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerSettable, SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable, SIntervalPointer, SIntervalPointerSettable, SSubsetStruct, SSubsetStructCode}
import is.hail.types.physical.stypes.{interfaces, _}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SStream, SStreamCode}
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.virtual._
import is.hail.utils._

import scala.language.{existentials, higherKinds}

case class EmitStreamContext(mb: EmitMethodBuilder[_])

abstract class COption[+A] { self =>
  def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl]

  def cases(mb: EmitMethodBuilder[_])(none: Code[Unit], some: A => Code[Unit]): Code[Unit] = {
    implicit val sctx: EmitStreamContext = EmitStreamContext(mb)
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

  def fromIEmitCode[A](mb: EmitMethodBuilder[_])(f: EmitCodeBuilder => IEmitCodeGen[A]): COption[A] = new COption[A] {
    def apply(none: Code[Ctrl], some: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
      EmitCodeBuilder.scopedVoid(mb) { cb =>
        f(cb).consume(cb, cb += none, cb += some(_))
      }
    }
  }

  def toEmitCode(opt: COption[PCode], mb: EmitMethodBuilder[_]): EmitCode = {
    implicit val sctx = EmitStreamContext(mb)
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
    implicit val sctx = EmitStreamContext(mb)
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
    implicit val sctx = EmitStreamContext(mb)
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

  def grouped[A](
    mb: EmitMethodBuilder[_],
    childStream: ChildStagedRegion => Stream[A],
    innerSeparateRegions: Boolean,
    size: Code[Int],
    eltRegion: ChildStagedRegion
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
          .asParent(innerSeparateRegions, "StreamGrouped inner")
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
    lElemType: PType, mkLeft: ChildStagedRegion => Stream[EmitCode],
    rElemType: PType, mkRight: ChildStagedRegion => Stream[EmitCode],
    destRegion: ChildStagedRegion,
    comp: (EmitValue, EmitValue) => Code[Int]
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
              cb.assign(rxOut, EmitCode.missing(mb, rElemType))
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
                  cb.assign(rxOut, EmitCode.missing(mb, rElemType))
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
    mb: EmitMethodBuilder[_],
    pcStream: SStreamCode,
    ab: StagedArrayBuilder,
    destRegion: ParentStagedRegion
  ): Code[Unit] = {
    _write(mb, pcStream.stream, ab, destRegion)
  }

  private def _write(
    mb: EmitMethodBuilder[_],
    sstream: SizedStream,
    ab: StagedArrayBuilder,
    destRegion: ParentStagedRegion
  ): Code[Unit] = {
    val SizedStream(ssSetup, stream, optLen) = sstream
    val eltRegion = destRegion.createChildRegion(mb)
    Code(FastSeq(
      eltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool()),
      ssSetup,
      ab.clear,
      ab.ensureCapacity(optLen.getOrElse(16)),
      stream(eltRegion).forEach(mb, { elt => Code(
        elt.setup,
        elt.m.mux(
          ab.addMissing(),
          EmitCodeBuilder.scopedVoid(mb)(cb => cb += ab.add(eltRegion.copyToParent(cb, elt.pv).code))),
        eltRegion.clear())
      }),
      eltRegion.free()))
  }

  def toArray(
    cb: EmitCodeBuilder,
    aTyp: PCanonicalArray,
    pcStream: SStreamCode,
    destRegion: ParentStagedRegion
  ): PCode = {
    val mb = cb.emb
    val ss = pcStream.stream
    val xLen = mb.newLocal[Int]("sta_len")
    ss.length match {
      case None =>
        val vab = new StagedArrayBuilder(aTyp.elementType, mb, 0)
        cb += _write(mb, ss, vab, destRegion)
        cb.assign(xLen, vab.size)

        aTyp.constructFromElements(cb, destRegion.code, xLen, deepCopy = false) { (cb, i) =>
          IEmitCode(cb, vab.isMissing(i), PCode(aTyp.elementType, vab(i)))
        }

      case Some(len) =>
        val eltRegion = destRegion.createChildRegion(mb)
        cb += eltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())
        cb += ss.setup
        cb.assign(xLen, len)

        aTyp.constructFromStream(cb, ss.stream(eltRegion), destRegion.code, xLen, deepCopy = eltRegion.isStrictChild)
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
            Code.switch(i, Leos.goto, elements.map(elem => EmitCodeBuilder.scopedVoid(mb){ cb => cb.assign(t, elem); cb.append(Lpush.goto)})),
            Lpush,
            i := i + 1,
            push(t)),
          Code(Leos, eos)))
    }
  }

  case class SizedStream(setup: Code[Unit], stream: ChildStagedRegion => Stream[EmitCode], length: Option[Code[Int]]) {
    def getStream(eltRegion: ChildStagedRegion): Stream[EmitCode] = stream(eltRegion).addSetup(setup)
  }

  object SizedStream {
    def unsized(stream: ChildStagedRegion => Stream[EmitCode]): SizedStream =
      SizedStream(Code._empty, stream, None)
  }

  def mux(mb: EmitMethodBuilder[_], eltType: PType, cond: Code[Boolean], region: ChildStagedRegion, left: Stream[EmitCode], right: Stream[EmitCode]): Stream[EmitCode] = new Stream[EmitCode] {
    def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
      val b = mb.genFieldThisRef[Boolean]()
      val Leos = CodeLabel()
      val elt = mb.newEmitField("stream_mux_elt", eltType)
      val Lpush = CodeLabel()

      val l = left(Code(Leos, eos), (a) => Code(EmitCodeBuilder.scopedVoid(mb)(_.assign(elt, a.castTo(mb, region.code, eltType))), Lpush, push(elt)))
      val r = right(Leos.goto, (a) => Code(EmitCodeBuilder.scopedVoid(mb)(_.assign(elt, a.castTo(mb, region.code, eltType))), Lpush.goto))

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
    resultType: PCanonicalArray,
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
      val eltRegions = destRegion.createSiblingRegionArray(mb, k)

      val runMatch = CodeLabel()
      val LpullChild = CodeLabel()
      val LloopEnd = CodeLabel()
      val LaddToResult = CodeLabel()
      val LstartNewKey = CodeLabel()
      val Leos = CodeLabel()
      val Lpush = CodeLabel()

      Code(Leos, eos)

      val (pushSetup, curResult) = EmitCodeBuilder.scoped(mb) { cb =>
        resultType.constructFromElements(cb, destRegion.code, k, deepCopy = false) { (cb, i) =>
          IEmitCode(cb, result(i).ceq(0L), PCode(eltType, result(i)))
        }
      }
      Code(Lpush,
        pushSetup,
        push((curKey, curResult)))

      Code(LstartNewKey,
        Code.forLoop(i := 0, i < k, i := i + 1, result(i) = 0L),
        EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb.assign(curKey, eltRegions(winner).copyTo(cb, eltType.loadCheapPCode(cb, heads(winner)).subset(key: _*), destRegion, keyType))
        },
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
        (challenger.cne(k) && (winner.ceq(k)
          || EmitCodeBuilder.scopedCode(mb) { cb =>
          val left = eltType.loadCheapPCode(cb, heads(challenger)).subset(key: _*)
          val right = eltType.loadCheapPCode(cb, heads(winner)).subset(key: _*)
          val ord = StructOrdering.make(left.st, right.st, cb.emb.ecb, missingFieldsEqual = false)
          ord.lteqNonnull(cb, left, right)
        })
          ).orEmpty(Code(
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
            (winner.cne(k)
              && EmitCodeBuilder.scopedCode(mb) { cb =>
              val left = eltType.loadCheapPCode(cb, heads(winner)).subset(key: _*)
              val right = curKey
              val ord = StructOrdering.make(left.st, right.st.asInstanceOf[SBaseStruct],
                cb.emb.ecb, missingFieldsEqual = false)
              ord.equivNonnull(cb, left, right)
            }).mux(
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
    innerStreamSeparateRegions: Boolean,
    innerStreamType: PStream,
    key: Array[String],
    eltRegion: ChildStagedRegion
  ): Stream[ChildStagedRegion => Stream[PCode]] = new Stream[ChildStagedRegion => Stream[PCode]] {
    def apply(outerEos: Code[Ctrl], outerPush: (ChildStagedRegion => Stream[PCode]) => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[ChildStagedRegion => Stream[PCode]] = {
      val eltType = coerce[PStruct](innerStreamType.elementType)
      val keyType = eltType.selectFields(key)

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
                (xCurKey.tcode[Long].cne(0L) &&
                  EmitCodeBuilder.scopedCode(mb) { cb =>
                    val right = xCurElt.asBaseStruct.subset(key: _*).asPCode
                    StructOrdering.make(xCurKey.st.asInstanceOf[SBaseStruct], right.st.asInstanceOf[SBaseStruct],
                      cb.emb.ecb, missingFieldsEqual = false)
                      .equivNonnull(cb, xCurKey, right)
                  }).mux(
                  xInOuter.mux(
                    Code(holdingRegion.clear(), LchildPull.goto),
                    LinnerPush.goto),
                  Code(
                    keyRegion.clear(),
                    EmitCodeBuilder.scopedVoid(mb) { cb =>
                      val pc = xCurElt.asBaseStruct.subset(key: _*)
                      cb.assign(xCurKey, pc.castTo(cb, keyRegion.code, keyType, deepCopy = true))
                    },
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

        assert(innerStreamType.separateRegions == innerStreamSeparateRegions)
        val innerEltRegion = eltRegion
          .asParent(innerStreamSeparateRegions, "StreamGroupByKey inner")
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
          EmitCodeBuilder.scopedVoid(mb)(_.assign(xCurKey, keyType.defaultValue(mb))),
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
    emitter: Emit[C],
    streamIR0: IR,
    mb: EmitMethodBuilder[C],
    outerRegion: ParentStagedRegion,
    env0: Emit.E,
    container: Option[AggContainer]
  ): EmitCode = {

    def _emitStream(cb: EmitCodeBuilder, streamIR: IR, outerRegion: ParentStagedRegion, env: Emit.E): IEmitCode = {
      assert(cb.isOpenEnded)

      def emitStream(streamIR: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, outerRegion: ParentStagedRegion = outerRegion): IEmitCode =
        _emitStream(cb, streamIR, outerRegion, env)

      def emitIR(ir: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, region: StagedRegion = outerRegion, container: Option[AggContainer] = container): IEmitCode =
        emitter.emitI(ir, cb, region, env, container, None)

      def emitVoidIR(ir: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, region: StagedRegion = outerRegion, container: Option[AggContainer] = container): Unit =
        emitter.emitVoid(cb, ir, mb, region, env, container, None)

      def sized(setup: Code[Unit], stream: ChildStagedRegion => Stream[EmitCode], length: Option[Code[Int]], outerRegion: ParentStagedRegion = outerRegion): SizedStream =
        SizedStream(setup, r => { r assertSubRegion outerRegion; stream(r) }, length)

      def unsized(stream: ChildStagedRegion => Stream[EmitCode], outerRegion: ParentStagedRegion = outerRegion): SizedStream =
        SizedStream.unsized(r => { r assertSubRegion outerRegion; stream(r) })

      val result: IEmitCode = streamIR match {
        case x@NA(_) =>
          IEmitCode.missing(cb, x.pType.defaultValue(mb))

        case x@Ref(name, _) =>
          val typ = coerce[PStream](x.pType)
          val ev = env.lookup(name)
          if (ev.pt != typ)
            throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $typ")
          ev.toI(cb)

        case x@StreamRange(startIR, stopIR, stepIR, _) =>
          val eltType = coerce[PStream](x.pType).elementType
          val llen = mb.genFieldThisRef[Long]("sr_llen")
          val len = mb.genFieldThisRef[Int]("sr_len")

          emitIR(startIR).flatMap(cb) { startc =>
            emitIR(stopIR).flatMap(cb) { stopc =>
              emitIR(stepIR).map(cb) { stepc =>
                val start = cb.memoizeField(startc, "sr_step")
                val stop = cb.memoizeField(stopc, "sr_stop")
                val step = cb.memoizeField(stepc, "sr_step")
                cb.ifx(step.asInt.intCode(cb) ceq const(0), cb._fatal("Array range cannot have step size 0."))
                cb.ifx(step.asInt.intCode(cb) < const(0), {
                  cb.ifx(start.asInt.intCode(cb).toL <= stop.asInt.intCode(cb).toL, {
                    cb.assign(llen, 0L)
                  }, {
                    cb.assign(llen, (start.asInt.intCode(cb).toL - stop.asInt.intCode(cb).toL - 1L) / (-step.asInt.intCode(cb).toL) + 1L)
                  })
                }, {
                  cb.ifx(start.asInt.intCode(cb).toL >= stop.asInt.intCode(cb).toL, {
                    cb.assign(llen, 0L)
                  }, {
                    cb.assign(llen, (stop.asInt.intCode(cb).toL - start.asInt.intCode(cb).toL - 1L) / step.asInt.intCode(cb).toL + 1L)
                  })
                })
                cb.ifx(llen > const(Int.MaxValue.toLong), {
                  cb._fatal("Array range cannot have more than MAXINT elements.")
                })

                val stream = sized(
                  len := llen.toI,
                  eltRegion => range(mb, start.asInt.intCode(cb), step.asInt.intCode(cb), len)
                    .map(i => EmitCode.present(mb, PCode(eltType, i))),
                  Some(len))

                interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, stream)
              }
            }
          }

        case x@ToStream(containerIR, _) =>
          emitIR(containerIR).map(cb) { containerAddr =>
            val a = containerAddr.asIndexable.memoizeField(cb, "ts_a")

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

            cb.assign(len, a.loadLength())
            val stream = sized(Code._empty, eltRegion => newStream, Some(len))
            interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, stream)
          }

        case x@MakeStream(elements, _, _) =>
          val eltType = coerce[PStream](x.pType).elementType
          val stream = (eltRegion: StagedRegion) =>
            sequence(mb, eltType, elements.toFastIndexedSeq.map { ir =>
              EmitCode.fromI(mb) { cb =>
                emitIR(ir, cb = cb, region = eltRegion).map(cb)(_.castTo(cb, eltRegion.code, eltType))
              }
            })

          val sstream = sized(Code._empty, stream, Some(elements.length))
          IEmitCode.present(cb, interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, sstream))

        case x@ReadPartition(context, rowType, reader) =>
          reader.emitStream(emitter.ctx.executeContext, context, rowType, emitter, cb, outerRegion, env, container)

        case In(n, _) =>
          // this, Code[Region], ...
          val param = mb.getEmitParam(2 + n, outerRegion.code)
          param.st match {
            case _: SStream =>
            case t => throw new RuntimeException(s"parameter ${ 2 + n } is not a stream! t=$t, params=${ mb.emitParamTypes }")
          }
          param.load.toI(cb)

        case StreamTake(a, num) =>
          val xN = mb.genFieldThisRef[Int]("st_n")
          emitStream(a).flatMap(cb) { pc =>
            val SizedStream(setup, stream, len) = pc.asStream.stream

            emitIR(num).map(cb) { n =>
              val sstream = sized(
                Code(setup,
                     EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xN, n.asInt.intCode(cb))),
                     (xN < 0).orEmpty(Code._fatal[Unit](const("StreamTake: negative length")))),
                eltRegion => zip(stream(eltRegion), range(mb, 0, 1, xN))
                  .map({ case (elt, count) => elt }),
                len.map(_.min(xN)))

              interfaces.SStreamCode(pc.st.asInstanceOf[SStream], sstream)
            }
          }

        case StreamDrop(a, num) =>
          val xN = mb.genFieldThisRef[Int]("st_n")
          emitStream(a).flatMap(cb) { pc =>
            val SizedStream(setup, stream, len) = pc.asStream.stream

            emitIR(num).map(cb) { n =>
              val sstream = sized(
                Code(setup,
                  EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xN, n.asInt.intCode(cb))),
                  (xN < 0).orEmpty(Code._fatal[Unit](const("StreamDrop: negative num")))),
                eltRegion => zip(stream(eltRegion), iota(mb, 0, 1))
                  .map({ case (elt, count) => COption(count < xN, elt) })
                  .flatten,
                len.map(l => (l - xN).max(0)))

              interfaces.SStreamCode(pc.st.asInstanceOf[SStream], sstream)
            }
          }

        case x@StreamGrouped(a, size) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          val xS = mb.genFieldThisRef[Int]("st_n")
          emitStream(a).flatMap(cb) { pc =>
            val SizedStream(setup, stream, len) = pc.asStream.stream

            emitIR(size).map(cb) { s =>

              val innerSeparateRegions = emitter.ctx.smm.lookup(a).separateRegions
              assert(innerSeparateRegions == innerType.separateRegions)

              val newStream = (eltRegion: ChildStagedRegion) =>
                Stream.grouped(mb, stream, innerSeparateRegions, xS, eltRegion)
                  .map { inner =>
                    EmitCode.present(mb,
                      interfaces.SStreamCode(
                        innerType.sType,
                        unsized(inner)))
                  }

              val sstream = sized(
                Code(setup,
                  EmitCodeBuilder.scopedVoid(mb)(cb => cb.assign(xS, s.asInt.intCode(cb))),
                  (xS <= 0).orEmpty(Code._fatal[Unit](const("StreamGrouped: nonpositive size")))),
                newStream,
                len.map(l => ((l.toL + xS.toL - 1L) / xS.toL).toI)) // rounding up integer division

              interfaces.SStreamCode(pc.st.asInstanceOf[SStream], sstream)
            }
          }

        case x@StreamGroupByKey(a, key) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          emitStream(a).map(cb) { pc =>
            val nonMissingStream = (eltRegion: ChildStagedRegion) => pc.asStream.stream.getStream(eltRegion).mapCPS[PCode] { (_, ec, k) =>
              Code(ec.setup, ec.m.orEmpty(Code._fatal[Unit](const("expected non-missing"))), k(ec.pv))
            }
            val newStream = (eltRegion: ChildStagedRegion) =>
              groupBy(mb, nonMissingStream, emitter.ctx.smm.lookup(a).separateRegions, innerType, key.toArray, eltRegion)
                .map { inner =>
                  EmitCode.present(mb,
                    interfaces.SStreamCode(
                      innerType.sType,
                      unsized { innerEltRegion =>
                        inner(innerEltRegion).map(pc => EmitCode.present(mb, pc))
                      }))
                }

            interfaces.SStreamCode(pc.st.asInstanceOf[SStream], unsized(newStream))
          }

        case StreamMap(childIR, name, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType

          emitStream(childIR).map(cb) { pc =>
            val SizedStream(setup, stream, len) = pc.asStream.stream

            def newStream(eltRegion: ChildStagedRegion): Stream[EmitCode] = stream(eltRegion).map { eltt => (eltType, bodyIR.pType) match {
              case (eltType: PCanonicalStream, bodyType: PCanonicalStream) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))
                val bodySeparateRegions = emitter.ctx.smm.lookup(bodyIR).separateRegions
                assert(bodySeparateRegions == bodyType.separateRegions)
                val outerRegion = eltRegion.asParent(bodySeparateRegions, "StreamMap body")

                EmitCode.fromI(mb)(cb => emitStream(bodyIR, cb = cb, env = bodyenv, outerRegion = outerRegion))
              case (eltType: PCanonicalStream, _) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))

                EmitCode.fromI(mb)(cb => emitIR(bodyIR, cb = cb, region = eltRegion, env = bodyenv))
              case (_, bodyType: PCanonicalStream) =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)
                val bodySeparateRegions = emitter.ctx.smm.lookup(bodyIR).separateRegions
                assert(bodySeparateRegions == bodyType.separateRegions)
                val outerRegion = eltRegion.asParent(bodySeparateRegions, "StreamMap body")

                EmitCode.fromI(mb) { cb =>
                  cb.assign(xElt, eltt)
                  emitStream(bodyIR, cb = cb, env = bodyenv, outerRegion = outerRegion)
                }
              case _ =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)

                EmitCode.fromI(mb) { cb =>
                  cb.assign(xElt, eltt)
                  emitIR(bodyIR, cb = cb, env = bodyenv, region = eltRegion)
                }
            }}

            interfaces.SStreamCode(
              pc.st.asInstanceOf[SStream],
              sized(setup, newStream, len))
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType

          emitStream(childIR).map(cb) { pc =>
            val newStream = (eltRegion: ChildStagedRegion) => {
              val tmpRegion = eltRegion.createSiblingRegion(mb)
              pc.asStream.stream.getStream(tmpRegion)
                .map (
                  { elt =>
                    new COption[EmitCode] {
                      def apply(none: Code[Ctrl], some: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Code[Ctrl] = {
                        EmitCodeBuilder.scopedVoid(mb) { cb =>
                          val xElt = mb.newEmitField(name, childEltType)
                          cb.assign(xElt, elt)
                          val Lfalse = CodeLabel()
                          val Ltrue = CodeLabel()
                          emitIR(condIR, cb = cb, env = env.bind(name -> xElt), region = tmpRegion).consume(cb, {
                            cb.goto(Lfalse)
                          }, { cond =>
                            cb.ifx(cond.asBoolean.boolCode(cb), cb.goto(Ltrue), cb.goto(Lfalse))
                          })
                          cb.define(Lfalse)
                          cb += tmpRegion.clear()
                          cb += none
                          cb.define(Ltrue)
                          cb += some(EmitCode.fromI(mb) { cb =>
                            xElt.toI(cb)
                                .mapMissing(cb) { cb += tmpRegion.clear() }
                                .map(cb) { pc =>
                                  cb += tmpRegion.giveToSibling(eltRegion)
                                  pc
                                }
                          })
                        }
                      }
                    }
                  },
                  setup0 = Some(tmpRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
                  close0 = Some(tmpRegion.free()))
                .flatten
            }

            interfaces.SStreamCode(pc.st.asInstanceOf[SStream], unsized(newStream))
          }

        case x@StreamMerge(leftIR, rightIR, key) =>
          val lElemType = coerce[PStruct](coerce[PStream](leftIR.pType).elementType)
          val rElemType = coerce[PStruct](coerce[PStream](rightIR.pType).elementType)
          val outElemType = coerce[PStream](x.pType).elementType


          def compare(lelt: EmitValue, relt: EmitValue): Code[Int] = EmitCodeBuilder.scopedCode(mb) { cb =>
            assert(lelt.pt == lElemType)
            assert(relt.pt == rElemType)
            val lhs = lelt.map(_.asBaseStruct.subset(key: _*).asPCode)
            val rhs = relt.map(_.asBaseStruct.subset(key: _*).asPCode)

            StructOrdering.make(lhs.st.asInstanceOf[SBaseStruct], rhs.st.asInstanceOf[SBaseStruct],
              cb.emb.ecb, missingFieldsEqual = false)
              .compare(cb, lhs, rhs, missingEqual = true)
          }

          emitStream(leftIR).flatMap(cb) { leftPC =>
            emitStream(rightIR).map(cb) { rightPC =>
              val SizedStream(leftSetup, leftStream, leftLen) = leftPC.asStream.stream
              val SizedStream(rightSetup, rightStream, rightLen) = rightPC.asStream.stream
              val newStream = sized(
                Code(leftSetup, rightSetup),
                eltRegion => merge(mb,
                  lElemType, leftStream,
                  rElemType, rightStream,
                  outElemType, eltRegion, compare),
                for (l <- leftLen; r <- rightLen) yield l + r)

              interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, newStream)
            }
          }

        case x@StreamZip(as, names, bodyIR, behavior) =>
          // FIXME: should make StreamZip support unrealizable element types
          val eltTypes = {
            val types = as.map(ir => coerce[PStream](ir.pType).elementType)
            behavior match {
              case ArrayZipBehavior.ExtendNA => types.map(_.setRequired(false))
              case _ => types
            }
          }
          val eltVars = (names, eltTypes).zipped.map(mb.newEmitField)

          IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(cb.emb)(cb => emitStream(a, cb)))) { pcs =>
            val emitStreams = pcs.map(_.asStream.stream)
            val lenSetup = Code(emitStreams.map(_.setup))
            val lengths = emitStreams.map(_.length)

            behavior match {

              case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
                val newStream = (eltRegion: ChildStagedRegion) =>
                  multiZip(emitStreams.map(_.stream(eltRegion)))
                    .map { elts =>
                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      EmitCode.fromI(mb) { cb =>
                        for((v, x) <- (eltVars, elts).zipped) {
                          cb.assign(v, x)
                        }
                        emitIR(bodyIR, cb = cb, env = bodyEnv, region = eltRegion)
                      }
                    }
                val newLength = behavior match {
                  case ArrayZipBehavior.TakeMinLength =>
                    lengths.reduceLeft(_.liftedZip(_).map {
                      case (l1, l2) => l1.min(l2)
                    })
                  case ArrayZipBehavior.AssumeSameLength =>
                    lengths.flatten.headOption
                }

                interfaces.SStreamCode(
                  coerce[PCanonicalStream](x.pType).sType,
                  sized(lenSetup, newStream, newLength))

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
                          optEC.cases(mb)(
                            anyEOS := true,
                            ec => EmitCodeBuilder.scopedVoid(mb) { cb =>
                              cb.assign(allEOS, false)
                              cb.assign(eltVar, ec)
                            })
                        }

                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      val body = EmitCode.fromI(mb)(cb => emitIR(bodyIR, cb = cb, env = bodyEnv, region = eltRegion))

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

                interfaces.SStreamCode(
                  coerce[PCanonicalStream](x.pType).sType,
                  sized(lenSetup, newStream, newLength))

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

                          COption.toEmitCode(optElt, mb)
                        }
                      val bodyEnv = env.bind(names.zip(eltVars): _*)
                      val body = EmitCode.fromI(mb)(cb => emitIR(bodyIR, cb = cb, env = bodyEnv, region = eltRegion))

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

                interfaces.SStreamCode(
                  coerce[PCanonicalStream](x.pType).sType,
                  sized(lenSetup, newStream, newLength))
            }
          }

        case x@StreamMultiMerge(as, key) =>
          val eltType = x.pType.elementType.asInstanceOf[PStruct]

          val keyViewType = PSubsetStruct(eltType, key: _*)
          def comp(li: Code[Int], lv: Code[Long], ri: Code[Int], rv: Code[Long]): Code[Boolean] =
            EmitCodeBuilder.scopedCode(mb) { cb =>
              val l = PCode(keyViewType, lv)
              val r = PCode(keyViewType, rv)
              val ord = cb.emb.ecb.getOrdering(l.st, r.st)
              val c = cb.newLocal("stream_merge_comp", ord.compareNonnull(cb, l, r))
              c < 0 || (c.ceq(0) && li < ri)
            }

          IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(cb.emb)(cb => emitStream(a, cb)))) { pcs =>
            val sss = pcs.map(_.asStream.stream)
            val streams = sss.map { ss => (eltRegion: ChildStagedRegion) =>
              ss.stream(eltRegion).map { ec =>
                EmitCodeBuilder.scopedCode(mb)(cb => ec.get().castTo(cb, outerRegion.code, eltType).tcode[Long])
              }
            }

            val newStream = sized(
              Code(sss.map(_.setup)),
              eltRegion => kWayMerge[Long](mb, streams, eltRegion, comp).map { case (i, elt) =>
                EmitCode.present(mb, PCode(eltType, elt))
              },
              sss.map(_.length).reduce(_.liftedZip(_).map {
                case (l, r) => l + r
              }))

            interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, newStream)
          }

        case x@StreamZipJoin(as, key, curKey, curVals, joinIR) =>
          val curValsType = x.curValsType
          val eltType = curValsType.elementType.setRequired(true).asInstanceOf[PStruct]
          val keyType = eltType.selectFields(key)

          def joinF(eltRegion: StagedRegion): ((PCode, PCode)) => EmitCode = { case (k, vs) =>
            val xKey = mb.newPresentEmitField("zipjoin_key", keyType)
            val xElts = mb.newPresentEmitField("zipjoin_elts", curValsType)
            val newEnv = env.bind(curKey -> xKey, curVals -> xElts)

            EmitCode.fromI(mb) { cb =>
              cb.assign(xKey, k)
              cb.assign(xElts, vs)

              joinIR.pType match {
                case _: PCanonicalStream =>
                  emit(emitter, joinIR, mb, outerRegion, newEnv, container).toI(cb)
                case _ =>
                  emitIR(joinIR, cb = cb, env = newEnv, region = eltRegion)
              }
            }
          }

          IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(cb.emb)(cb => emitStream(a, cb)))) { pcs =>
            val sss = pcs.map(_.asStream.stream)
            val streams = sss.map { ss => (eltRegion: ChildStagedRegion) =>
              ss.getStream(eltRegion).map(_.get())
            }
            val newStream = unsized { eltRegion =>
              kWayZipJoin(mb, streams, eltRegion, curValsType, key)
                .map(joinF(eltRegion))
            }

            interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, newStream)
          }

        case x@StreamFlatMap(outerIR, name, innerIR) =>
          val outerEltType = coerce[PStream](outerIR.pType).elementType
          val innerStreamType = coerce[PStream](innerIR.pType)

          val outerEltRegion = outerRegion.createChildRegion(mb)

          emitStream(outerIR).map(cb) { outer =>
            val newStream = (eltRegion: ChildStagedRegion) => {
              outer.asStream.stream.getStream(outerEltRegion)
                .map[COption[Stream[EmitCode]]] { elt =>
                  val innerSeparateRegions = emitter.ctx.smm.lookup(innerIR).separateRegions
                  assert(innerSeparateRegions == innerStreamType.separateRegions)

                  val innerStreamOuterRegion =
                    outerEltRegion.asParent(innerSeparateRegions, "StreamFlatMap inner")
                  val optInner = if (outerEltType.isRealizable) {
                    COption.fromIEmitCode(mb) { cb =>
                      val xElt = cb.memoizeField(elt, "sfm_elt")
                      emitStream(
                        innerIR,
                        cb = cb,
                        outerRegion = innerStreamOuterRegion,
                        env = env.bind(name -> xElt))
                    }
                  } else {
                    COption.fromIEmitCode(mb) { cb =>
                      emitStream(
                        innerIR,
                        cb = cb,
                        outerRegion = innerStreamOuterRegion,
                        env = env.bind(name -> new EmitUnrealizableValue(outerEltType, elt)))
                    }
                  }

                  optInner.map { inner =>
                    // We know that eltRegion is a subregion of innerStreamOuterRegion,
                    // even though the former was constructed before the later.
                    inner.asStream.stream.getStream(eltRegion.asSubregionOf(innerStreamOuterRegion))
                      .map(x => x,
                           close = Some(outerEltRegion.clear()))
                  }
                }
                .flatten.flatten
                .map(x => x,
                     setup0 = Some(outerEltRegion.allocateRegion(Region.REGULAR, mb.ecb.pool())),
                     close0 = Some(outerEltRegion.free()))
            }

            interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, unsized(newStream))
          }

        case x@If(condIR, thn, els) =>
          val eltType = coerce[PStream](x.pType).elementType

          emitIR(condIR).flatMap(cb) { cond =>
            val xCond = mb.genFieldThisRef[Boolean]("stream_if_cond")
            cb.assign(xCond, cond.asBoolean.boolCode(cb))
            var leftSS: SizedStream = null
            var rightSS: SizedStream = null
            val Lmissing = CodeLabel()
            val Lpresent = CodeLabel()

            cb.ifx(xCond, {
              emitStream(thn).consume(cb, cb.goto(Lmissing), { s => leftSS = s.asStream.stream; cb.goto(Lpresent) })
            }, {
              emitStream(els).consume(cb, cb.goto(Lmissing), { s => rightSS = s.asStream.stream; cb.goto(Lpresent) })
            })

            val SizedStream(leftSetup, leftStream, lLen) = leftSS
            val SizedStream(rightSetup, rightStream, rLen) = rightSS
            val newStream = (eltRegion: ChildStagedRegion) => {
              mux(mb, eltType,
                xCond,
                eltRegion,
                leftStream(eltRegion),
                rightStream(eltRegion))
            }
            val newLen = lLen.liftedZip(rLen).map { case (l1, l2) =>
              xCond.mux(l1, l2)
            }
            val newSetup = xCond.mux(leftSetup, rightSetup)

            val newSS = interfaces.SStreamCode(
              coerce[PCanonicalStream](x.pType).sType,
              sized(newSetup, newStream, newLen))

            IEmitCode(Lmissing, Lpresent, newSS)
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
              cb.assign(xValue, emitIR(valueIR))
              val bodyEnv = env.bind(name -> xValue)

              emitStream(bodyIR, env = bodyEnv)
          }

        case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = x.accPType

          emitStream(childIR).map(cb) { pc =>
            val SizedStream(setup, stream, len) = pc.asStream.stream
            val Lpush = CodeLabel()
            val hasPulled = mb.genFieldThisRef[Boolean]()

            val xAccInEltR = mb.newEmitField(accName, accType)
            val xAccInAccR = mb.newEmitField(accName, accType)

            val newStream = (eltRegion: ChildStagedRegion) => new Stream[EmitCode] {
              def apply(eos: Code[Ctrl], push: EmitCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[EmitCode] = {
                val accRegion = eltRegion.createSiblingRegion(mb)

                Code(Lpush,
                  EmitCodeBuilder.scopedVoid(mb) { cb => cb.assign(xAccInAccR, xAccInEltR.map(eltRegion.copyTo(cb, _, accRegion))) },
                  push(xAccInEltR))

                val source = stream(eltRegion).apply(
                  eos = eos,
                  push = a => EmitCodeBuilder.scopedVoid(mb) { cb =>
                    val xElt = cb.memoizeField(a, "st_scan_elt")
                    val bodyEnv = env.bind(accName -> xAccInAccR, eltName -> xElt)
                    val body = emitIR(bodyIR, cb = cb, env = bodyEnv, region = eltRegion)
                      .map(cb)(_.castTo(cb, eltRegion.code, accType))
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
                      cb.assign(xAccInEltR, emitIR(zeroIR, cb = cb, region = eltRegion)
                        .map(cb)(_.castTo(cb, eltRegion.code, accType)))
                      cb.append(Lpush.goto)
                    }))
              }
            }

            val newLen = len.map(l => l + 1)
            interfaces.SStreamCode(
              coerce[PCanonicalStream](x.pType).sType,
              sized(setup, newStream, newLen))
          }

        case x@RunAggScan(array, name, init, seqs, result, states) =>
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb,"array_agg_scan")

          val eltType = coerce[PStream](array.pType).elementType

          emitStream(array).map(cb) { pc =>
            val SizedStream(setup, stream, len) = pc.asStream.stream

            val newStream = (eltRegion: ChildStagedRegion) => {
              val tmpRegion = eltRegion.createSiblingRegion(mb)

              val setup = EmitCodeBuilder.scopedVoid(mb) { cb =>
                cb += tmpRegion.allocateRegion(Region.SMALL, mb.ecb.pool())
                emitVoidIR(init, cb = cb, region = tmpRegion, container = Some(newContainer))
                cb += tmpRegion.free()
              }

              stream(eltRegion).map[EmitCode](
                { eltt =>
                  EmitCode.fromI(mb) { cb =>
                    val xElt = cb.memoizeField(eltt, "aggscan_elt")
                    val bodyEnv = env.bind(name -> xElt)
                    val postt = emitIR(result, cb = cb, region = eltRegion, env = bodyEnv, container = Some(newContainer))
                    val xResult = cb.memoizeField(postt, "aggscan_result")
                    emitVoidIR(seqs, cb = cb, region = eltRegion, env = bodyEnv, container = Some(newContainer))
                    xResult.toI(cb)
                  }
                },
                setup0 = Some(aggSetup),
                close0 = Some(aggCleanup),
                setup = Some(setup))
            }

            interfaces.SStreamCode(
              coerce[PCanonicalStream](x.pType).sType,
              sized(setup, newStream, len))
          }

        case x@StreamJoinRightDistinct(leftIR, rightIR, lKey, rKey, leftName, rightName, joinIR, joinType) =>
          assert(joinType == "left" || joinType == "outer")
          val lEltType = coerce[PStruct](coerce[PStream](leftIR.pType).elementType)
          val rEltType = coerce[PStruct](coerce[PStream](rightIR.pType).elementType)
          val xLElt = mb.newEmitField("join_lelt", lEltType.orMissing(joinType == "left"))
          val xRElt = mb.newEmitField("join_relt", rEltType.setRequired(false))
          val newEnv = env.bind(leftName -> xLElt, rightName -> xRElt)

          def compare(lelt: EmitValue, relt: EmitValue): Code[Int] = {
            assert(lelt.pt == lEltType)
            assert(relt.pt == rEltType)

            EmitCodeBuilder.scopedCode(mb) { cb =>
              val lhs = lelt.map(_.asBaseStruct.subset(lKey: _*).asPCode)
              val rhs = relt.map(_.asBaseStruct.subset(rKey: _*).asPCode)
              StructOrdering.make(lhs.st.asInstanceOf[SBaseStruct], rhs.st.asInstanceOf[SBaseStruct],
                cb.emb.ecb, missingFieldsEqual = false)
                .compare(cb, lhs, rhs, missingEqual = false)
            }
          }

          def joinF: ((EmitCode, EmitCode)) => EmitCode = { case (lelt, relt) =>
            EmitCode.fromI(mb) { cb =>
              cb.assign(xLElt, lelt)
              cb.assign(xRElt, relt)

              joinIR.pType match {
                case _: PCanonicalStream =>
                  emit(emitter, joinIR, mb, outerRegion, newEnv, container).toI(cb)
                case _ =>
                  emitIR (joinIR, cb = cb, newEnv)
              }
            }
          }

          emitStream(leftIR).flatMap(cb) { leftPC =>
            val SizedStream(leftSetup, leftStream, leftLen) = leftPC.asStream.stream
            emitStream(rightIR).map(cb) { rightSS =>
              val newStream = (eltRegion: ChildStagedRegion) => {
                if (joinType == "left")
                  leftJoinRightDistinct(
                    mb,
                    lEltType, leftStream,
                    rEltType, rightSS.asStream.stream.getStream,
                    eltRegion,
                    compare)
                    .map(joinF)
                else
                  outerJoinRightDistinct(
                    mb,
                    lEltType, leftStream,
                    rEltType, rightSS.asStream.stream.getStream,
                    eltRegion,
                    compare)
                    .map(joinF)
              }

              val newSStream = sized(leftSetup,
                    newStream,
                    if (joinType == "left") leftLen else None)

              interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, newSStream)
            }
          }

        case x@ShuffleRead(idIR, keyRangeIR) =>
          val shuffleType = coerce[TShuffle](idIR.typ)
          assert(shuffleType.rowDecodedPType == coerce[PStream](x.pType).elementType)
          val keyType = coerce[TInterval](keyRangeIR.typ).pointType
          val keyPType = coerce[PInterval](keyRangeIR.pType).pointType
          assert(keyType == shuffleType.keyType)
          assert(keyPType == shuffleType.keyDecodedPType)

          val idt = emitIR(idIR).get(cb, "ShuffleRead cannot have null ID").asShuffle
          val keyRangeCode =
            emitIR(keyRangeIR).get(cb, "ShuffleRead cannot have null key range").asInterval

          val uuid = idt.memoize(cb, "shuffleUUID")
          val keyRange = keyRangeCode.memoizeField(cb, "shuffleClientKeyRange")

          cb.ifx(!keyRange.startDefined(cb) || !keyRange.endDefined(cb), {
            Code._fatal[Unit]("ShuffleRead cannot have null start or end points of key range")
          })

          val shuffleLocal = mb.newLocal[ShuffleClient]("shuffleClient")
          val shuffle = new ValueShuffleClient(shuffleLocal)

          val stream = (eltRegion: ChildStagedRegion) => unfold[EmitCode](
            { (_, k) =>
              k(
                COption(
                  shuffle.getValueFinished(),
                  EmitCode.present(mb,
                    PCode(shuffleType.rowDecodedPType, shuffle.getValue(eltRegion.code)))))
            },
            setup = Some(EmitCodeBuilder.scopedVoid(mb) { cb =>
              cb.assign(shuffleLocal, CodeShuffleClient.create(
                mb.ecb.getType(shuffleType),
                uuid.loadBytes(),
                Code._null,
                mb.ecb.getPType(keyPType)))

              val startt = keyRange.loadStart(cb)
                  .get(cb, "shuffle expects defined endpoints")
                  .asPCode
                  .tcode[Long]
              val endt = keyRange.loadEnd(cb)
                  .get(cb, "shuffle expects defined endpoints")
                  .asPCode
                  .tcode[Long]

              cb.append(shuffle.startGet(startt, keyRange.includesStart, endt, keyRange.includesEnd))
            }),
            close = Some(Code(
              shuffle.getDone(),
              shuffle.close())))

          IEmitCode.present(cb,
            interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, unsized(stream)))

      case x@ShufflePartitionBounds(idIR, nPartitionsIR) =>
          val shuffleType = coerce[TShuffle](idIR.typ)
          assert(shuffleType.keyDecodedPType == coerce[PStream](x.pType).elementType)

        val idt = emitIR(idIR).get(cb, "ShufflePartitionBounds cannot have null ID").asInstanceOf[SCanonicalShufflePointerCode]
        val nPartitionst = emitIR(nPartitionsIR).get(cb, "ShufflePartitionBounds cannot have null number of partitions").asInt

        val uuidLocal = mb.newLocal[Long]("shuffleUUID")
        val uuid = new SCanonicalShufflePointerSettable(idt.st, new SBinaryPointerSettable(SBinaryPointer(idt.st.pType.representation), uuidLocal))
        val shuffleLocal = mb.newLocal[ShuffleClient]("shuffleClient")
        val shuffle = new ValueShuffleClient(shuffleLocal)

        val stream = (eltRegion: ChildStagedRegion) => unfold[EmitCode](
          { (_, k) =>
            k(
              COption(
                shuffle.partitionBoundsValueFinished(),
                EmitCode.present(mb,
                  PCode(shuffleType.keyDecodedPType, shuffle.partitionBoundsValue(eltRegion.code)))))
          },
          setup = Some(EmitCodeBuilder.scopedVoid(mb) { cb =>
            uuid.store(cb, idt)
            cb.assign(shuffleLocal, CodeShuffleClient.create(mb.ecb.getType(shuffleType), uuid.loadBytes()))
            cb += shuffle.startPartitionBounds(nPartitionst.intCode(cb)) }),
          close = Some(Code(
            shuffle.endPartitionBounds(),
            shuffle.close())))

        IEmitCode.present(cb,
          interfaces.SStreamCode(coerce[PCanonicalStream](x.pType).sType, unsized(stream)))

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }

      if (!result.pt.isInstanceOf[PStream])
        throw new RuntimeException(s"expected stream, got ${ result.pt }")
      result
    }

    EmitCode.fromI(mb) { cb =>
      _emitStream(cb, streamIR0, outerRegion, env0)
    }
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
