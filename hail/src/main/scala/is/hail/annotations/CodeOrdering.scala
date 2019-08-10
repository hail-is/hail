package is.hail.annotations

import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types._
import is.hail.asm4s.coerce
import is.hail.expr.types.physical._
import is.hail.utils._

object CodeOrdering {

  type Op = Int
  val compare: Op = 0
  val equiv: Op = 1
  val lt: Op = 2
  val lteq: Op = 3
  val gt: Op = 4
  val gteq: Op = 5
  val neq: Op = 6

  type F[R] = ((Code[Boolean], Code[_]), (Code[Boolean], Code[_])) => Code[R]

  def rowOrdering(t1: PBaseStruct, t2: PBaseStruct, mb: EmitMethodBuilder): CodeOrdering = new CodeOrdering {
    type T = Long

    val m1: LocalRef[Boolean] = mb.newLocal[Boolean]
    val m2: LocalRef[Boolean] = mb.newLocal[Boolean]

    val v1s: Array[LocalRef[_]] = t1.types.map(tf => mb.newLocal(ir.typeToTypeInfo(tf)))
    val v2s: Array[LocalRef[_]] = t2.types.map(tf => mb.newLocal(ir.typeToTypeInfo(tf)))

    def setup(i: Int)(x: Code[Long], y: Code[Long]): Code[Unit] = {
      val tf1 = t1.types(i)
      val tf2 = t2.types(i)
      Code(
        m1 := t1.isFieldMissing(x, i),
        m2 := t2.isFieldMissing(y, i),
        v1s(i).storeAny(m1.mux(ir.defaultValue(tf1), Region.loadIRIntermediate(tf1)(t1.fieldOffset(x, i)))),
        v2s(i).storeAny(m2.mux(ir.defaultValue(tf2), Region.loadIRIntermediate(tf2)(t2.fieldOffset(y, i)))))
    }

    override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
      val cmp = mb.newLocal[Int]

      val c = Array.tabulate(t1.size) { i =>
        val mbcmp = mb.getCodeOrdering[Int](t1.types(i), t2.types(i), CodeOrdering.compare)
        Code(setup(i)(x, y),
          mbcmp((m1, v1s(i)), (m2, v2s(i))))
      }.foldRight[Code[Int]](cmp.load()) { (ci, cont) => cmp.ceq(0).mux(Code(cmp := ci, cont), cmp) }

      Code(cmp := 0, c)
    }

    override def ltNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      Array.tabulate(t1.size) { i =>
        val mblt = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.lt)
        val mbequiv = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.equiv)
        (Code(setup(i)(x, y), mblt((m1, v1s(i)), (m2, v2s(i)))),
          mbequiv((m1, v1s(i)), (m2, v2s(i))))
      }.foldRight[Code[Boolean]](false) { case ((clt, ceq), cont) => clt || (ceq && cont) }
    }

    override def lteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      Array.tabulate(t1.size) { i =>
        val mblteq = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.lteq)
        val mbequiv = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.equiv)
        (Code(setup(i)(x, y), mblteq((m1, v1s(i)), (m2, v2s(i)))),
          mbequiv((m1, v1s(i)), (m2, v2s(i))))
      }.foldRight[Code[Boolean]](true) { case ((clteq, ceq), cont) => clteq && (!ceq || cont) }
    }

    override def gtNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      Array.tabulate(t1.size) { i =>
        val mbgt = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.gt)
        val mbequiv = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.equiv)
        (Code(setup(i)(x, y), mbgt((m1, v1s(i)), (m2, v2s(i)))),
          mbequiv((m1, v1s(i)), (m2, v2s(i))))
      }.foldRight[Code[Boolean]](false) { case ((cgt, ceq), cont) => cgt || (ceq && cont) }
    }

    override def gteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      Array.tabulate(t1.size) { i =>
        val mbgteq = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.gteq)
        val mbequiv = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.equiv)
        (Code(setup(i)(x, y), mbgteq((m1, v1s(i)), (m2, v2s(i)))),
          mbequiv((m1, v1s(i)), (m2, v2s(i))))
      }.foldRight[Code[Boolean]](true) { case ((cgteq, ceq), cont) => cgteq && (!ceq || cont) }
    }

    override def equivNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      Array.tabulate(t1.size) { i =>
        val mbequiv = mb.getCodeOrdering[Boolean](t1.types(i), t2.types(i), CodeOrdering.equiv)
        Code(setup(i)(x, y),
          mbequiv((m1, v1s(i)), (m2, v2s(i))))
      }.foldRight[Code[Boolean]](const(true))(_ && _)
    }
  }

  def iterableOrdering(t1: PArray, t2: PArray, mb: EmitMethodBuilder): CodeOrdering = new CodeOrdering {
    type T = Long
    val lord: CodeOrdering = PInt32().codeOrdering(mb)
    val ord: CodeOrdering = t1.elementType.codeOrdering(mb, t2.elementType)
    val len1: LocalRef[Int] = mb.newLocal[Int]
    val len2: LocalRef[Int] = mb.newLocal[Int]
    val lim: LocalRef[Int] = mb.newLocal[Int]
    val i: LocalRef[Int] = mb.newLocal[Int]
    val m1: LocalRef[Boolean] = mb.newLocal[Boolean]
    val v1: LocalRef[ord.T] = mb.newLocal(ir.typeToTypeInfo(t1.elementType)).asInstanceOf[LocalRef[ord.T]]
    val m2: LocalRef[Boolean] = mb.newLocal[Boolean]
    val v2: LocalRef[ord.T] = mb.newLocal(ir.typeToTypeInfo(t2.elementType)).asInstanceOf[LocalRef[ord.T]]
    val eq: LocalRef[Boolean] = mb.newLocal[Boolean]

    def loop(cmp: Code[Unit], loopCond: Code[Boolean])
      (x: Code[Long], y: Code[Long]): Code[Unit] = {
      Code(
        i := 0,
        len1 := t1.loadLength(x),
        len2 := t2.loadLength(y),
        lim := (len1 < len2).mux(len1, len2),
        Code.whileLoop(loopCond && i < lim,
          m1 := t1.isElementMissing(x, i),
          v1.storeAny(Region.loadIRIntermediate(t1.elementType)(t1.elementOffset(x, len1, i))),
          m2 := t2.isElementMissing(y, i),
          v2.storeAny(Region.loadIRIntermediate(t2.elementType)(t2.elementOffset(y, len2, i))),
          cmp, i += 1))
    }

    override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
      val mbcmp = mb.getCodeOrdering[Int](t1.elementType, t2.elementType, CodeOrdering.compare)
      val cmp = mb.newLocal[Int]

      Code(cmp := 0,
        loop(cmp := mbcmp((m1, v1), (m2, v2)), cmp.ceq(0))(x, y),
        cmp.ceq(0).mux(
          lord.compareNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load())),
          cmp))
    }

    override def ltNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mblt = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.lt)
      val mbequiv = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.equiv)
      val lt = mb.newLocal[Boolean]
      val lcmp = Code(
        lt := mblt((m1, v1), (m2, v2)),
        eq := mbequiv((m1, v1), (m2, v2)))

      Code(lt := false, eq := true,
        loop(lcmp, !lt && eq)(x, y),
        lt || eq && lord.ltNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load())))
    }

    override def lteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mblteq = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.lteq)
      val mbequiv = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.equiv)

      val lteq = mb.newLocal[Boolean]
      val lcmp = Code(
        lteq := mblteq((m1, v1), (m2, v2)),
        eq := mbequiv((m1, v1), (m2, v2)))

      Code(lteq := true, eq := true,
        loop(lcmp, eq)(x, y),
        lteq && (!eq || lord.lteqNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load()))))
    }

    override def gtNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mbgt = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.gt)
      val mbequiv = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.equiv)
      val gt = mb.newLocal[Boolean]
      val lcmp = Code(
        gt := mbgt((m1, v1), (m2, v2)),
        eq := !gt && mbequiv((m1, v1), (m2, v2)))

      Code(gt := false,
        eq := true,
        loop(lcmp, eq)(x, y),
        gt || (eq &&
            lord.gtNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load()))))
    }

    override def gteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mbgteq = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.gteq)
      val mbequiv = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.equiv)

      val gteq = mb.newLocal[Boolean]
      val lcmp = Code(
        gteq := mbgteq((m1, v1), (m2, v2)),
        eq := mbequiv((m1, v1), (m2, v2)))

      Code(gteq := true,
        eq := true,
        loop(lcmp, eq)(x, y),
        gteq && (!eq || lord.gteqNonnull(coerce[lord.T](len1.load()),  coerce[lord.T](len2.load()))))
    }

    override def equivNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mbequiv = mb.getCodeOrdering[Boolean](t1.elementType, t2.elementType, CodeOrdering.equiv)
      val lcmp = eq := mbequiv((m1, v1), (m2, v2))
      Code(eq := true,
        loop(lcmp, eq)(x, y),
        eq && lord.equivNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load())))
    }
  }

  def intervalOrdering(t1: PInterval, t2: PInterval, mb: EmitMethodBuilder): CodeOrdering = new CodeOrdering {
    type T = Long
    val mp1: LocalRef[Boolean] = mb.newLocal[Boolean]
    val mp2: LocalRef[Boolean] = mb.newLocal[Boolean]
    val p1: LocalRef[_] = mb.newLocal(ir.typeToTypeInfo(t1.pointType))
    val p2: LocalRef[_] = mb.newLocal(ir.typeToTypeInfo(t2.pointType))

    def loadStart(x: Code[T], y: Code[T]): Code[Unit] = {
      Code(
        mp1 := !t1.startDefined(x),
        mp2 := !t2.startDefined(y),
        p1.storeAny(mp1.mux(ir.defaultValue(t1.pointType), Region.loadIRIntermediate(t1.pointType)(t1.startOffset(x)))),
        p2.storeAny(mp2.mux(ir.defaultValue(t2.pointType), Region.loadIRIntermediate(t2.pointType)(t2.startOffset(y)))))
    }

    def loadEnd(x: Code[T], y: Code[T]): Code[Unit] = {
      Code(
        mp1 := !t1.endDefined(x),
        mp2 := !t2.endDefined(y),
        p1.storeAny(mp1.mux(ir.defaultValue(t1.pointType), Region.loadIRIntermediate(t1.pointType)(t1.endOffset(x)))),
        p2.storeAny(mp2.mux(ir.defaultValue(t2.pointType), Region.loadIRIntermediate(t2.pointType)(t2.endOffset(y)))))
    }

    override def compareNonnull(x: Code[T], y: Code[T]): Code[Int] = {
      val mbcmp = mb.getCodeOrdering[Int](t1.pointType, t2.pointType, CodeOrdering.compare)

      val cmp = mb.newLocal[Int]
      Code(loadStart(x, y),
        cmp := mbcmp((mp1, p1), (mp2, p2)),
        cmp.ceq(0).mux(
          Code(mp1 := t1.includeStart(x),
            mp1.cne(t2.includeStart(y)).mux(
              mp1.mux(-1, 1),
              Code(
                loadEnd(x, y),
                cmp := mbcmp((mp1, p1), (mp2, p2)),
                cmp.ceq(0).mux(
                  Code(mp1 := t1.includeEnd(x),
                    mp1.cne(t2.includeEnd(y)).mux(mp1.mux(1, -1), 0)),
                  cmp)))),
          cmp))
    }

    override def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mbeq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code(loadStart(x, y), mbeq((mp1, p1), (mp2, p2))) &&
        t1.includeStart(x).ceq(t2.includeStart(y)) &&
        Code(loadEnd(x, y), mbeq((mp1, p1), (mp2, p2))) &&
        t1.includeEnd(x).ceq(t2.includeEnd(y))
    }

    override def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mblt = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.lt)
      val mbeq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code(loadStart(x, y), mblt((mp1, p1), (mp2, p2))) || (
        mbeq((mp1, p1), (mp2, p2)) && (
          Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), mp1 && !mp2) || (mp1.ceq(mp2) && (
            Code(loadEnd(x, y), mblt((mp1, p1), (mp2, p2))) || (
              mbeq((mp1, p1), (mp2, p2)) &&
                !t1.includeEnd(x) && t2.includeEnd(y))))))
    }

    override def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mblteq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.lteq)
      val mbeq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code(loadStart(x, y), mblteq((mp1, p1), (mp2, p2))) && (
        !mbeq((mp1, p1), (mp2, p2)) || ( // if not equal, then lt
          Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), mp1 && !mp2) || (mp1.ceq(mp2) && (
            Code(loadEnd(x, y), mblteq((mp1, p1), (mp2, p2))) && (
              !mbeq((mp1, p1), (mp2, p2)) ||
                !t1.includeEnd(x) || t2.includeEnd(y))))))
    }

    override def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mbgt = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.gt)
      val mbeq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code(loadStart(x, y), mbgt((mp1, p1), (mp2, p2))) || (
        mbeq((mp1, p1), (mp2, p2)) && (
          Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), !mp1 && mp2) || (mp1.ceq(mp2) && (
            Code(loadEnd(x, y), mbgt((mp1, p1), (mp2, p2))) || (
              mbeq((mp1, p1), (mp2, p2)) &&
                t1.includeEnd(x) && !t2.includeEnd(y))))))
    }

    override def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mbgteq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.gteq)
      val mbeq = mb.getCodeOrdering[Boolean](t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code(loadStart(x, y), mbgteq((mp1, p1), (mp2, p2))) && (
        !mbeq((mp1, p1), (mp2, p2)) || ( // if not equal, then lt
          Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), !mp1 && mp2) || (mp1.ceq(mp2) && (
            Code(loadEnd(x, y), mbgteq((mp1, p1), (mp2, p2))) && (
              !mbeq((mp1, p1), (mp2, p2)) ||
                t1.includeEnd(x) || !t2.includeEnd(y))))))
    }
  }

  def mapOrdering(t1: PDict, t2: PDict, mb: EmitMethodBuilder): CodeOrdering =
    iterableOrdering(PArray(t1.elementType, t1.required), PArray(t2.elementType, t2.required), mb)

  def setOrdering(t1: PSet, t2: PSet, mb: EmitMethodBuilder): CodeOrdering =
    iterableOrdering(PArray(t1.elementType, t1.required), PArray(t2.elementType, t2.required), mb)

}

abstract class CodeOrdering {
  outer =>

  type T
  type P = (Code[Boolean], Code[T])

  def compareNonnull(x: Code[T], y: Code[T]): Code[Int]

  def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) < 0

  def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) <= 0

  def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) > 0

  def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) >= 0

  def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y).ceq(0)

  def compare(x: P, y: P): Code[Int] = {
    val (xm, xv) = x
    val (ym, yv) = y
    val compMissing = (xm && ym).mux(0, xm.mux(1, -1))

    (xm || ym).mux(compMissing, compareNonnull(xv, yv))
  }

  def lt(x: P, y: P): Code[Boolean] = {
    val (xm, xv) = x
    val (ym, yv) = y
    val compMissing = (xm && ym).mux(false, !xm)

    (xm || ym).mux(compMissing, ltNonnull(xv, yv))
  }

  def lteq(x: P, y: P): Code[Boolean] = {
    val (xm, xv) = x
    val (ym, yv) = y
    val compMissing = (xm && ym).mux(true, !xm)

    (xm || ym).mux(compMissing, lteqNonnull(xv, yv))
  }

  def gt(x: P, y: P): Code[Boolean] = {
    val (xm, xv) = x
    val (ym, yv) = y
    val compMissing = (xm && ym).mux(false, xm)

    (xm || ym).mux(compMissing, gtNonnull(xv, yv))
  }

  def gteq(x: P, y: P): Code[Boolean] = {
    val (xm, xv) = x
    val (ym, yv) = y
    val compMissing = (xm && ym).mux(true, xm)

    (xm || ym).mux(compMissing, gteqNonnull(xv, yv))
  }

  def equiv(x: P, y: P): Code[Boolean] = {
    val (xm, xv) = x
    val (ym, yv) = y

    (xm || ym).mux(xm && ym, equivNonnull(xv, yv))
  }
}
