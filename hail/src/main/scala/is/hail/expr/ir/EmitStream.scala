 package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region, RegionValue, StagedRegionValueBuilder}
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

  def grouped(size: Code[Int]): Stream[Stream[A]] = new Stream[Stream[A]] {
    def apply(outerEos: Code[Ctrl], outerPush: Stream[A] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[Stream[A]] = {
      val xCounter = ctx.mb.newLocal[Int]("st_grp_ctr")
      val xInOuter = ctx.mb.newLocal[Boolean]("st_grp_io")
      val xSize = ctx.mb.newLocal[Int]("st_grp_sz")
      val LchildPull = CodeLabel()
      val LouterPush = CodeLabel()
      val LinnerPush = CodeLabel()
      val LouterEos = CodeLabel()

      var childSource: Source[A] = null
      val inner = new Stream[A] {
        def apply(innerEos: Code[Ctrl], innerPush: A => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[A] = {
          val LinnerEos = CodeLabel()

          childSource = self(
            xInOuter.mux(LouterEos.goto, LinnerEos.goto),
            { a =>
              Code(LinnerPush, innerPush(a))

              Code(
                // xCounter takes values in [1, xSize + 1]
                xCounter := xCounter + 1,
                // !xInOuter iff this element was requested by an inner stream.
                // Else we are stepping to the beginning of the next group.
                xInOuter.mux(
                  (xCounter > xSize).mux(
                    // first of a group
                    Code(xCounter := 1, LouterPush.goto),
                    LchildPull.goto),
                  LinnerPush.goto))
            })

          Code(LinnerEos, innerEos)
          Code(LchildPull, childSource.pull)

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
      Code(LouterEos, outerEos)

      Source[Stream[A]](
        setup0 = childSource.setup0,
        close0 = childSource.close0,
        setup = Code(
          childSource.setup,
          xSize := size,
          xCounter := xSize),
        close = childSource.close,
        pull = Code(xInOuter := true, LchildPull.goto))
    }
  }

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
    val lstep = mb.newLocal[Int]("sr_lstep")
    val cur = mb.newLocal[Int]("sr_cur")

    unfold[Code[Int]](
      f = {
        case (_ctx, k) =>
          implicit val ctx = _ctx
          Code(cur := cur + lstep, k(COption.present(cur)))
      },
      setup = Some(Code(lstep := step, cur := start - lstep)))
  }

  def iotaL(mb: EmitMethodBuilder[_], start: Code[Long], step: Code[Int]): Stream[Code[Long]] = {
    val lstep = mb.newLocal[Int]("sr_lstep")
    val cur = mb.newLocal[Long]("sr_cur")

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

  implicit class StreamStream[A](val outer: Stream[Stream[A]]) extends AnyVal {
    def flatten: Stream[A] = new Stream[A] {
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

  import Stream._

  def write(mb: EmitMethodBuilder[_], sstream: SizedStream, ab: StagedArrayBuilder): Code[Unit] = {
    val SizedStream(ssSetup, stream, optLen) = sstream
    Code(
      ssSetup,
      ab.clear,
      ab.ensureCapacity(optLen.getOrElse(16)),
      stream.forEach(mb, { et =>
        Code(et.setup, et.m.mux(ab.addMissing(), ab.add(et.v)))
      }))
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

        case Some(len) =>
          PCode(aTyp, Code(
            ss.setup,
            srvb.start(len),
            ss.stream.forEach(mb, { et =>
              Code(
                et.setup,
                et.m.mux(srvb.setMissing(), srvb.addIRIntermediate(aTyp.elementType)(et.v)),
                srvb.advance())
            }),
            srvb.offset))
      }
    }

    COption.toEmitCode(result, mb)
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
  case class SizedStream(setup: Code[Unit], stream: Stream[EmitCode], length: Option[Code[Int]]) {
    def getStream: Stream[EmitCode] = stream.addSetup(setup)
  }

  object SizedStream {
    def unsized(stream: Stream[EmitCode]): SizedStream =
      SizedStream(Code._empty, stream, None)
  }

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

  def groupBy(mb: EmitMethodBuilder[_], region: Value[Region], stream: Stream[PCode], eltType: PStruct, key: Array[String]): Stream[Stream[PCode]] = new Stream[Stream[PCode]] {
    def apply(outerEos: Code[Ctrl], outerPush: Stream[PCode] => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[Stream[PCode]] = {
      val keyType = eltType.selectFields(key)
      val keyViewType = PSubsetStruct(eltType, key)
      val ordering = keyType.codeOrdering(mb, keyViewType).asInstanceOf[CodeOrdering { type T = Long }]

      val xCurKey = ctx.mb.newPLocal("st_grpby_curkey", keyType)
      val xCurElt = ctx.mb.newPLocal("st_grpby_curelt", eltType)
      val xInOuter = ctx.mb.newLocal[Boolean]("st_grpby_io")
      val xEOS = ctx.mb.newLocal[Boolean]("st_grpby_eos")
      val xNextGrpReady = ctx.mb.newLocal[Boolean]("st_grpby_ngr")

      val LchildPull = CodeLabel()
      val LouterPush = CodeLabel()
      val LinnerPush = CodeLabel()
      val LouterEos = CodeLabel()
      val LinnerEos = CodeLabel()

      var childSource: Source[PCode] = null
      val inner = new Stream[PCode] {
        def apply(innerEos: Code[Ctrl], innerPush: PCode => Code[Ctrl])(implicit ctx: EmitStreamContext): Source[PCode] = {
          childSource = stream(
            xInOuter.mux(LouterEos.goto, Code(xEOS := true, LinnerEos.goto)),
            { a: PCode =>
              Code(
                xCurElt := a,
                // !xInOuter iff this element was requested by an inner stream.
                // Else we are stepping to the beginning of the next group.
                (xCurKey.tcode[Long].cne(0L) && ordering.equivNonnull(xCurKey.tcode[Long], xCurElt.tcode[Long])).mux(
                  xInOuter.mux(
                    LchildPull.goto,
                    LinnerPush.goto),
                  Code(
                    xCurKey := keyType.copyFromPValue(mb, region, new PSubsetStructCode(keyViewType, xCurElt.tcode[Long])),
                    xInOuter.mux(
                      LouterPush.goto,
                      Code(xNextGrpReady := true, LinnerEos.goto)))))
            })

          Code(LinnerPush, innerPush(xCurElt))
          Code(LinnerEos, innerEos)
          Code(LchildPull, childSource.pull)

          Source[PCode](
            setup0 = Code._empty,
            close0 = Code._empty,
            setup = Code._empty,
            close = Code._empty,
            pull = xInOuter.mux(
              // xInOuter iff this is the first pull from inner stream,
              // in which case the element has already been produced
              Code(xInOuter := false, LinnerPush.goto),
              LchildPull.goto))
        }
      }

      Code(LouterPush, outerPush(inner))
      Code(LouterEos, outerEos)

      Source[Stream[PCode]](
        setup0 = childSource.setup0,
        close0 = childSource.close0,
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

  private[ir] def emit[C](
    emitter: Emit[C],
    streamIR0: IR,
    mb: EmitMethodBuilder[C],
    env0: Emit.E,
    container: Option[AggContainer]
  ): COption[SizedStream] =
    emit(emitter, streamIR0, mb, mb.getCodeParam[Region](1), env0, container)

  private[ir] def emit[C](
    emitter: Emit[C],
    streamIR0: IR,
    mb: EmitMethodBuilder[C],
    region: Value[Region],
    env0: Emit.E,
    container: Option[AggContainer]
  ): COption[SizedStream] = {

    def emitStream(streamIR: IR, env: Emit.E): COption[SizedStream] = {

      def emitIR(ir: IR, env: Emit.E = env, region: Value[Region] = region, container: Option[AggContainer] = container): EmitCode =
        emitter.emitWithRegion(ir, mb, region, env, container)

      def emitVoidIR(ir: IR, env: Emit.E = env, container: Option[AggContainer] = container): Code[Unit] = {
        EmitCodeBuilder.scopedVoid(mb) { cb =>
          emitter.emitVoid(cb, ir, mb, region, env, container, None)
        }
      }

      streamIR match {
        case x@NA(_) =>
          COption.none(SizedStream(
            Code._empty,
            coerce[PCanonicalStream](x.pType).defaultValue.stream,
            Some(0)))

        case x@Ref(name, _) =>
          val typ = coerce[PStream](x.pType)
          val ev = env.lookup(name)
          if (ev.pt != typ)
            throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $typ")
          COption.fromEmitCode(ev.get).map { pc =>
            SizedStream.unsized(pc.asStream.stream)
          }

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
                        len := llen.toI,
                        range(mb, start, step, len)
                          .map(i => EmitCode(Code._empty, const(false), PCode(eltType, i))),
                        Some(len)))))))
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

            Code(
              asetup,
              len := a.loadLength(),
              k(SizedStream(Code._empty, newStream, Some(len))))
          }

        case x@MakeStream(elements, t) =>
          val eltType = coerce[PStream](x.pType).elementType
          val stream = sequence(mb, eltType, elements.toFastIndexedSeq.map { ir =>
              val et = emitIR(ir)
              EmitCode(et.setup, et.m, PCode(eltType, eltType.copyFromTypeAndStackValue(mb, region, ir.pType, et.value)))
          })

          COption.present(SizedStream(Code._empty, stream, Some(elements.length)))

        case x@ReadPartition(context, rowType, reader) =>
          reader.emitStream(context, rowType, emitter, mb, region, env, container)

        case In(n, PCanonicalStream(eltType, _)) =>
          val xIter = mb.newLocal[Iterator[RegionValue]]()

          // this, Region, ...
          mb.getStreamEmitParam(2 + n).map { iter =>
            val stream = unfold[Code[RegionValue]](
              (_, k) => k(COption(
                !xIter.load().hasNext,
                xIter.load().next()))
            ).map(
              rv => EmitCode.present(eltType, Region.loadIRIntermediate(eltType)(rv.invoke[Long]("getOffset"))),
              setup0 = None,
              setup = Some(xIter := iter)
            )

            SizedStream.unsized(stream)
          }

        case StreamTake(a, num) =>
          val optStream = emitStream(a, env)
          val optN = COption.fromEmitCode(emitIR(num))
          val xN = mb.newLocal[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optN.map { n =>
              val newStream = zip(stream, range(mb, 0, 1, xN))
                .map({ case (elt, count) => elt })
              SizedStream(
                Code(setup, xN := n.tcode[Int], (xN < 0).orEmpty(Code._fatal[Unit](const("StreamTake: negative length")))),
                newStream,
                len.map(_.min(xN)))
            }
          }

        case StreamDrop(a, num) =>
          val optStream = emitStream(a, env)
          val optN = COption.fromEmitCode(emitIR(num))
          val xN = mb.newLocal[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optN.map { n =>
              val newStream =
               zip(stream, iota(mb, 0, 1))
                .map({ case (elt, count) => COption(count < xN, elt) })
                .flatten
              SizedStream(
                Code(setup, xN := n.tcode[Int], (xN < 0).orEmpty(Code._fatal[Unit](const("StreamDrop: negative num")))),
                newStream,
                len.map(l => (l - xN).max(0)))
            }
          }

        case x@StreamGrouped(a, size) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          val optStream = emitStream(a, env)
          val optSize = COption.fromEmitCode(emitIR(size))
          val xS = mb.newLocal[Int]("st_n")
          optStream.flatMap { case SizedStream(setup, stream, len) =>
            optSize.map { s =>
              val newStream = stream.grouped(xS).map(inner => EmitCode(Code._empty, false, PCanonicalStreamCode(innerType, inner)))
              SizedStream(
                Code(setup, xS := s.tcode[Int], (xS <= 0).orEmpty(Code._fatal[Unit](const("StreamGrouped: nonpositive size")))),
                newStream,
                len.map(l => ((l.toL + xS.toL - 1L) / xS.toL).toI)) // rounding up integer division
            }
          }

        case x@StreamGroupByKey(a, key) =>
          val innerType = coerce[PCanonicalStream](coerce[PStream](x.pType).elementType)
          val eltType = coerce[PStruct](innerType.elementType)
          val optStream = emitStream(a, env)
          optStream.map { ss =>
            val nonMissingStream = ss.getStream.mapCPS[PCode] { (_, ec, k) =>
              Code(ec.setup, ec.m.orEmpty(Code._fatal[Unit](const("expected non-missing"))), k(ec.pv))
            }
            val newStream = groupBy(mb, region, nonMissingStream, eltType, key.toArray)
              .map(inner => EmitCode.present(PCanonicalStreamCode(innerType, inner.map(EmitCode.present))))

            SizedStream.unsized(newStream)
          }

        case StreamMap(childIR, name, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR, env)
          optStream.map { case SizedStream(setup, stream, len) =>
            val newStream = stream.map { eltt => (eltType, bodyIR.pType) match {
              case (eltType: PCanonicalStream, bodyType: PCanonicalStream) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))

                COption.toEmitCode(
                  emitStream(bodyIR, env = bodyenv)
                    .map(ss => PCanonicalStreamCode(bodyType, ss.getStream)),
                  mb)
              case (eltType: PCanonicalStream, _) =>
                val bodyenv = env.bind(name -> new EmitUnrealizableValue(eltType, eltt))

                emitIR(bodyIR, env = bodyenv)
              case (_, bodyType: PCanonicalStream) =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)

                EmitCode(
                  xElt := eltt,
                  COption.toEmitCode(
                    emitStream(bodyIR, env = bodyenv)
                      .map(ss => PCanonicalStreamCode(bodyType, ss.getStream)),
                    mb))
              case _ =>
                val xElt = mb.newEmitField(name, eltType)
                val bodyenv = env.bind(name -> xElt)
                val bodyt = emitIR(bodyIR, env = bodyenv)

                EmitCode(xElt := eltt, bodyt)
            }}

            SizedStream(setup, newStream, len)
          }

        case StreamFilter(childIR, name, condIR) =>
          val childEltType = coerce[PStream](childIR.pType).elementType

          val optStream = emitStream(childIR, env)

          optStream.map { ss =>
            val newStream = ss.getStream
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
              }.flatten

            SizedStream.unsized(newStream)
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

          val optStreams = COption.lift(as.map(emitStream(_, env)))

          optStreams.map { emitStreams =>
            val lenSetup = Code(emitStreams.map(_.setup))
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
                      case (l1, l2) => l1.min(l2)
                    })
                  case ArrayZipBehavior.AssumeSameLength =>
                    lengths.flatten.headOption
                }

                SizedStream(lenSetup, newStream, newLength)

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
                val newStream = flagged.take

                val newLength = lengths.flatten match {
                  case Seq() => None
                  case ls =>
                    val len = mb.newLocal[Int]("zip_asl_len")
                    val lenTemp = mb.newLocal[Int]("zip_asl_len_temp")
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

                        COption.toEmitCode(optElt, mb)
                      }
                    val bodyEnv = env.bind(names.zip(eltVars): _*)
                    val body = emitIR(bodyIR, env = bodyEnv)

                    Code(
                      allEOS := true,
                      Code((eltVars, checkedElts).zipped.map { (v, x) => v := x }),
                      k(COption(allEOS, body)))
                  }

                // termininate the stream when all streams are EOS
                val newStream = flagged.take

                val newLength = lengths.reduceLeft(_.liftedZip(_).map {
                  case (l1, l2) => l1.max(l2)
                })

                SizedStream(lenSetup, newStream, newLength)
            }
          }

        case StreamFlatMap(outerIR, name, innerIR) =>
          val outerEltType = coerce[PStream](outerIR.pType).elementType

          val optOuter = emitStream(outerIR, env)

          optOuter.map { outer =>
            val nested = outer.getStream.map[COption[Stream[EmitCode]]] { elt =>
              if (outerEltType.isRealizable) {
                val xElt = mb.newEmitField(name, outerEltType)
                val innerEnv = env.bind(name -> xElt)
                val optInner = emitStream(innerIR, innerEnv).map(_.getStream)

                optInner.addSetup(xElt := elt)
              } else {
                val innerEnv = env.bind(name -> new EmitUnrealizableValue(outerEltType, elt))

                emitStream(innerIR, innerEnv).map(_.getStream)
              }

            }

            SizedStream.unsized(nested.flatten.flatten)
          }

        case If(condIR, thn, els) =>
          val eltType = coerce[PStream](thn.pType).elementType
          val xCond = mb.genFieldThisRef[Boolean]("stream_if_cond")

          val condT = COption.fromEmitCode(emitIR(condIR))
          val optLeftStream = emitStream(thn, env)
          val optRightStream = emitStream(els, env)

          condT.flatMap[SizedStream] { cond =>
            val newOptStream = COption.choose[SizedStream](
              xCond,
              optLeftStream,
              optRightStream,
              { case (SizedStream(leftSetup, leftStream, lLen), SizedStream(rightSetup, rightStream, rLen)) =>
                  val newStream = mux(mb, eltType,
                    xCond,
                    leftStream,
                    rightStream)
                  val newLen = lLen.liftedZip(rLen).map { case (l1, l2) =>
                    xCond.mux(l1, l2)
                  }
                  val newSetup = xCond.mux(leftSetup, rightSetup)

                  SizedStream(newSetup, newStream, newLen)
              })

            newOptStream.addSetup(xCond := cond.tcode[Boolean])
          }

        case Let(name, valueIR, bodyIR) =>
          val valueType = valueIR.pType

          valueType match {
            case streamType: PCanonicalStream =>
              val valuet = COption.toEmitCode(
                emitStream(valueIR, env)
                  .map(ss => PCanonicalStreamCode(streamType, ss.getStream)),
                mb)
              val bodyEnv = env.bind(name -> new EmitUnrealizableValue(valueType, valuet))

              emitStream(bodyIR, bodyEnv)

            case _ =>
              val xValue = mb.newEmitField(name, valueType)
              val bodyEnv = env.bind(name -> xValue)
              val valuet = emitIR(valueIR)

              emitStream(bodyIR, bodyEnv).addSetup(xValue := valuet)
          }

        case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
          val eltType = coerce[PStream](childIR.pType).elementType
          val accType = coerce[PStream](x.pType).elementType

          val streamOpt = emitStream(childIR, env)
          streamOpt.map { case SizedStream(setup, stream, len) =>
            val Lpush = CodeLabel()
            val hasPulled = mb.newLocal[Boolean]()

            val xElt = mb.newEmitField(eltName, eltType)
            val xAcc = mb.newEmitField(accName, accType)
            val tmpAcc = mb.newEmitField(accName, accType)

            val zero = emitIR(zeroIR).map(accType.copyFromPValue(mb, region, _))
            val bodyEnv = env.bind(accName -> tmpAcc, eltName -> xElt)

            val body = emitIR(bodyIR, env = bodyEnv).map(accType.copyFromPValue(mb, region, _))

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

            val newLen = len.map(l => l + 1)
            SizedStream(setup, newStream, newLen)
          }

        case x@RunAggScan(array, name, init, seqs, result, _) =>
          val aggs = x.physicalSignatures
          val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(aggs, mb, "array_agg_scan")

          val eltType = coerce[PStream](array.pType).elementType

          val xElt = mb.newEmitField("aggscan_elt", eltType)
          val xResult = mb.newEmitField("aggscan_result", result.pType)

          val bodyEnv = env.bind(name -> xElt)
          val cInit = emitVoidIR(init, container = Some(newContainer))
          val seqPerElt = emitVoidIR(seqs, env = bodyEnv, container = Some(newContainer))
          val postt = emitIR(result, env = bodyEnv, container = Some(newContainer))

          val optStream = emitStream(array, env)

          optStream.map { case SizedStream(setup, stream, len) =>
            val newStream = stream.map[EmitCode](
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

            SizedStream(setup, newStream, len)
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

          emitStream(leftIR, env).flatMap { case SizedStream(leftSetup, leftStream, leftLen) =>
            emitStream(rightIR, env).map { ss =>
              val rightStream = ss.getStream
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

              SizedStream(leftSetup, newStream, leftLen)
            }
          }

        case _ =>
          fatal(s"not a streamable IR: ${Pretty(streamIR)}")
      }
    }

    emitStream(streamIR0, env0)
  }
}
