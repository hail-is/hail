package is.hail.annotations

import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.{Ascending, EmitMethodBuilder, SortOrder}
import is.hail.expr.types._
import is.hail.asm4s.coerce
import is.hail.expr.types.physical._
import is.hail.expr.ir.{Ascending, Descending}
import is.hail.utils._

object CodeOrdering {

  sealed trait Op {
    type ReturnType
    val rtti: TypeInfo[ReturnType]
  }
  final case object compare extends Op {
    type ReturnType = Int
    val rtti = typeInfo[Int]
  }
  sealed trait BooleanOp extends Op {
    type ReturnType = Boolean
    val rtti = typeInfo[Boolean]
  }
  final case object equiv extends BooleanOp
  final case object lt extends BooleanOp
  final case object lteq extends BooleanOp
  final case object gt extends BooleanOp
  final case object gteq extends BooleanOp
  final case object neq extends BooleanOp

  type F[R] = ((Code[Boolean], Code[_]), (Code[Boolean], Code[_])) => Code[R]

  def rowOrdering(
    t1: PBaseStruct,
    t2: PBaseStruct,
    mb: EmitMethodBuilder[_],
    sortOrders: Array[SortOrder] = null
  ): CodeOrdering = new CodeOrdering {
    require(sortOrders == null || sortOrders.size == t1.size)
    type T = Long
    
    val m1: LocalRef[Boolean] = mb.newLocal[Boolean]()
    val m2: LocalRef[Boolean] = mb.newLocal[Boolean]()

    val v1s: Array[LocalRef[_]] = t1.types.map(tf => mb.newLocal()(ir.typeToTypeInfo(tf)))
    val v2s: Array[LocalRef[_]] = t2.types.map(tf => mb.newLocal()(ir.typeToTypeInfo(tf)))

    def setup(i: Int)(x: Value[Long], y: Value[Long]): Code[Unit] = {
      val tf1 = t1.types(i)
      val tf2 = t2.types(i)
      Code(
        m1 := t1.isFieldMissing(x, i),
        m2 := t2.isFieldMissing(y, i),
        v1s(i).storeAny(m1.mux(ir.defaultValue(tf1), Region.loadIRIntermediate(tf1)(t1.fieldOffset(x, i)))),
        v2s(i).storeAny(m2.mux(ir.defaultValue(tf2), Region.loadIRIntermediate(tf2)(t2.fieldOffset(y, i)))))
    }

    private[this] def fieldOrdering(i: Int, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
      mb.getCodeOrdering(
        t1.types(i),
        t2.types(i),
        if (sortOrders == null) Ascending else sortOrders(i),
        op)

    override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
      val cmp = mb.newLocal[Int]()

      Code.memoize(x, "cord_row_comp_x", y, "cord_row_comp_y") { (x, y) =>
        val c = Array.tabulate(t1.size) { i =>
          val mbcmp = fieldOrdering(i, CodeOrdering.compare)
          Code(setup(i)(x, y),
            mbcmp((m1, v1s(i)), (m2, v2s(i))))
        }.foldRight(cmp.get) { (ci, cont) => cmp.ceq(0).mux(Code(cmp := ci, cont), cmp) }

        Code(cmp := 0, c)
      }
    }

    private[this] def dictionaryOrderingFromFields(
      op: CodeOrdering.BooleanOp,
      zero: Code[Boolean],
      combine: (Code[Boolean], Code[Boolean], Code[Boolean]) => Code[Boolean]
    )(x: Code[Long],
      y: Code[Long]
    ): Code[Boolean] =
      Code.memoize(x, "cord_row_comp_x", y, "cord_row_comp_y") { (x, y) =>
        Array.tabulate(t1.size) { i =>
          val mbop = fieldOrdering(i, op)
          val mbequiv = fieldOrdering(i, CodeOrdering.equiv)
          (Code(setup(i)(x, y), mbop((m1, v1s(i)), (m2, v2s(i)))),
            mbequiv((m1, v1s(i)), (m2, v2s(i))))
        }.foldRight(zero) { case ((cop, ceq), cont) => combine(cop, ceq, cont) }
      }

    val _ltNonnull = dictionaryOrderingFromFields(
      CodeOrdering.lt,
      false,
      { (isLessThan, isEqual, subsequentLt) =>
        isLessThan || (isEqual && subsequentLt) }) _
    override def ltNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = _ltNonnull(x, y)

    val _lteqNonnull = dictionaryOrderingFromFields(
      CodeOrdering.lteq,
      true,
      { (isLessThanEq, isEqual, subsequentLtEq) =>
        isLessThanEq && (!isEqual || subsequentLtEq) }) _
    override def lteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = _lteqNonnull(x, y)

    val _gtNonnull = dictionaryOrderingFromFields(
      CodeOrdering.gt,
      false,
      { (isGreaterThan, isEqual, subsequentGt) =>
        isGreaterThan || (isEqual && subsequentGt) }) _
    override def gtNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = _gtNonnull(x, y)

    val _gteqNonnull = dictionaryOrderingFromFields(
      CodeOrdering.gteq,
      true,
      { (isGreaterThanEq, isEqual, subsequentGteq) =>
        isGreaterThanEq && (!isEqual || subsequentGteq) }) _
    override def gteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = _gteqNonnull(x, y)

    override def equivNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] =
      Code.memoize(x, "cord_row_equiv_x", y, "cord_row_equiv_y") { (x, y) =>
        Array.tabulate(t1.size) { i =>
          val mbequiv = fieldOrdering(i, CodeOrdering.equiv)
          Code(setup(i)(x, y),
            mbequiv((m1, v1s(i)), (m2, v2s(i))))
        }.foldRight[Code[Boolean]](const(true))(_ && _)
      }
  }

  def iterableOrdering(t1: PArray, t2: PArray, mb: EmitMethodBuilder[_]): CodeOrdering = new CodeOrdering {
    type T = Long
    val lord: CodeOrdering = PInt32().codeOrdering(mb)
    val ord: CodeOrdering = t1.elementType.codeOrdering(mb, t2.elementType)
    val len1: LocalRef[Int] = mb.newLocal[Int]()
    val len2: LocalRef[Int] = mb.newLocal[Int]()
    val lim: LocalRef[Int] = mb.newLocal[Int]()
    val i: LocalRef[Int] = mb.newLocal[Int]()
    val m1: LocalRef[Boolean] = mb.newLocal[Boolean]()
    val v1: LocalRef[ord.T] = mb.newLocal()(ir.typeToTypeInfo(t1.elementType)).asInstanceOf[LocalRef[ord.T]]
    val m2: LocalRef[Boolean] = mb.newLocal[Boolean]()
    val v2: LocalRef[ord.T] = mb.newLocal()(ir.typeToTypeInfo(t2.elementType)).asInstanceOf[LocalRef[ord.T]]
    val eq: LocalRef[Boolean] = mb.newLocal[Boolean]()

    def loop(cmp: Code[Unit], loopCond: Code[Boolean])
      (x: Code[Long], y: Code[Long]): Code[Unit] = {
      Code.memoize(x, "cord_iter_ord_x", y, "cord_iter_ord_y") { (x, y) =>
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
    }

    override def compareNonnull(x: Code[Long], y: Code[Long]): Code[Int] = {
      val mbcmp = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.compare)
      val cmp = mb.newLocal[Int]()

      Code(cmp := 0,
        loop(cmp := mbcmp((m1, v1), (m2, v2)), cmp.ceq(0))(x, y),
        cmp.ceq(0).mux(
          lord.compareNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load())),
          cmp))
    }

    override def ltNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mblt = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.lt)
      val mbequiv = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.equiv)
      val lt = mb.newLocal[Boolean]()
      val lcmp = Code(
        lt := mblt((m1, v1), (m2, v2)),
        eq := mbequiv((m1, v1), (m2, v2)))

      Code(lt := false, eq := true,
        loop(lcmp, !lt && eq)(x, y),
        lt || eq && lord.ltNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load())))
    }

    override def lteqNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mblteq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.lteq)
      val mbequiv = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.equiv)

      val lteq = mb.newLocal[Boolean]()
      val lcmp = Code(
        lteq := mblteq((m1, v1), (m2, v2)),
        eq := mbequiv((m1, v1), (m2, v2)))

      Code(lteq := true, eq := true,
        loop(lcmp, eq)(x, y),
        lteq && (!eq || lord.lteqNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load()))))
    }

    override def gtNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mbgt = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.gt)
      val mbequiv = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.equiv)
      val gt = mb.newLocal[Boolean]()
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
      val mbgteq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.gteq)
      val mbequiv = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.equiv)

      val gteq = mb.newLocal[Boolean]()
      val lcmp = Code(
        gteq := mbgteq((m1, v1), (m2, v2)),
        eq := mbequiv((m1, v1), (m2, v2)))

      Code(gteq := true,
        eq := true,
        loop(lcmp, eq)(x, y),
        gteq && (!eq || lord.gteqNonnull(coerce[lord.T](len1.load()),  coerce[lord.T](len2.load()))))
    }

    override def equivNonnull(x: Code[Long], y: Code[Long]): Code[Boolean] = {
      val mbequiv = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.equiv)
      val lcmp = eq := mbequiv((m1, v1), (m2, v2))
      Code(eq := true,
        loop(lcmp, eq)(x, y),
        eq && lord.equivNonnull(coerce[lord.T](len1.load()), coerce[lord.T](len2.load())))
    }
  }

  def intervalOrdering(t1: PInterval, t2: PInterval, mb: EmitMethodBuilder[_]): CodeOrdering = new CodeOrdering {
    type T = Long
    val mp1: LocalRef[Boolean] = mb.newLocal[Boolean]()
    val mp2: LocalRef[Boolean] = mb.newLocal[Boolean]()
    val p1: LocalRef[_] = mb.newLocal()(ir.typeToTypeInfo(t1.pointType))
    val p2: LocalRef[_] = mb.newLocal()(ir.typeToTypeInfo(t2.pointType))

    def loadStart(x: Value[T], y: Value[T]): Code[Unit] = {
      Code(
        mp1 := !t1.startDefined(x),
        mp2 := !t2.startDefined(y),
        p1.storeAny(mp1.mux(ir.defaultValue(t1.pointType), Region.loadIRIntermediate(t1.pointType)(t1.startOffset(x)))),
        p2.storeAny(mp2.mux(ir.defaultValue(t2.pointType), Region.loadIRIntermediate(t2.pointType)(t2.startOffset(y)))))
    }

    def loadEnd(x: Value[T], y: Value[T]): Code[Unit] = {
      Code(
        mp1 := !t1.endDefined(x),
        mp2 := !t2.endDefined(y),
        p1.storeAny(mp1.mux(ir.defaultValue(t1.pointType), Region.loadIRIntermediate(t1.pointType)(t1.endOffset(x)))),
        p2.storeAny(mp2.mux(ir.defaultValue(t2.pointType), Region.loadIRIntermediate(t2.pointType)(t2.endOffset(y)))))
    }

    override def compareNonnull(x: Code[T], y: Code[T]): Code[Int] = {
      val mbcmp = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.compare)

      val cmp = mb.newLocal[Int]()
      Code.memoize(x, "cord_int_comp_x", y, "cord_int_comp_y") { (x, y) =>
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
    }

    override def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mbeq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code.memoize(x, "cord_int_equiv_x", y, "cord_int_equiv_y") { (x, y) =>
        Code(loadStart(x, y), mbeq((mp1, p1), (mp2, p2))) &&
          t1.includeStart(x).ceq(t2.includeStart(y)) &&
          Code(loadEnd(x, y), mbeq((mp1, p1), (mp2, p2))) &&
          t1.includeEnd(x).ceq(t2.includeEnd(y))
      }
    }

    override def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mblt = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.lt)
      val mbeq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code.memoize(x, "cord_int_lt_x", y, "cord_int_lt_y") { (x, y) =>
        Code(loadStart(x, y), mblt((mp1, p1), (mp2, p2))) || (
          mbeq((mp1, p1), (mp2, p2)) && (
            Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), mp1 && !mp2) || (mp1.ceq(mp2) && (
              Code(loadEnd(x, y), mblt((mp1, p1), (mp2, p2))) || (
                mbeq((mp1, p1), (mp2, p2)) &&
                  !t1.includeEnd(x) && t2.includeEnd(y))))))
      }
    }

    override def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mblteq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.lteq)
      val mbeq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code.memoize(x, "cord_int_lteq_x", y, "cord_int_lteq_y") { (x, y) =>
        Code(loadStart(x, y), mblteq((mp1, p1), (mp2, p2))) && (
          !mbeq((mp1, p1), (mp2, p2)) || (// if not equal, then lt
            Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), mp1 && !mp2) || (mp1.ceq(mp2) && (
              Code(loadEnd(x, y), mblteq((mp1, p1), (mp2, p2))) && (
                !mbeq((mp1, p1), (mp2, p2)) ||
                  !t1.includeEnd(x) || t2.includeEnd(y))))))
      }
    }

    override def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mbgt = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.gt)
      val mbeq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code.memoize(x, "cord_int_gt_x", y, "cord_int_gt_y") { (x, y) =>
        Code(loadStart(x, y), mbgt((mp1, p1), (mp2, p2))) || (
          mbeq((mp1, p1), (mp2, p2)) && (
            Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), !mp1 && mp2) || (mp1.ceq(mp2) && (
              Code(loadEnd(x, y), mbgt((mp1, p1), (mp2, p2))) || (
                mbeq((mp1, p1), (mp2, p2)) &&
                  t1.includeEnd(x) && !t2.includeEnd(y))))))
      }
    }

    override def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = {
      val mbgteq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.gteq)
      val mbeq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.equiv)

      Code.memoize(x, "cord_int_gteq_x", y, "cord_int_gteq_y") { (x, y) =>
        Code(loadStart(x, y), mbgteq((mp1, p1), (mp2, p2))) && (
          !mbeq((mp1, p1), (mp2, p2)) || (// if not equal, then lt
            Code(mp1 := t1.includeStart(x), mp2 := t2.includeStart(y), !mp1 && mp2) || (mp1.ceq(mp2) && (
              Code(loadEnd(x, y), mbgteq((mp1, p1), (mp2, p2))) && (
                !mbeq((mp1, p1), (mp2, p2)) ||
                  t1.includeEnd(x) || !t2.includeEnd(y))))))
      }
    }
  }

  def mapOrdering(t1: PDict, t2: PDict, mb: EmitMethodBuilder[_]): CodeOrdering =
    iterableOrdering(PArray(t1.elementType, t1.required), PArray(t2.elementType, t2.required), mb)

  def setOrdering(t1: PSet, t2: PSet, mb: EmitMethodBuilder[_]): CodeOrdering =
    iterableOrdering(PArray(t1.elementType, t1.required), PArray(t2.elementType, t2.required), mb)

}

abstract class CodeOrdering {
  outer =>

  type T
  type P = (Code[Boolean], Code[T])

  def compareNonnull(x: Code[T], y: Code[T]): Code[Int]

  def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean]

  def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean]

  def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean]

  def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean]

  def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean]

  private[this] def liftMissing[U](
    op: (Code[T], Code[T]) => Code[U],
    whenMissing: (Code[Boolean], Code[Boolean]) => Code[U]
  ): (P, P) => Code[U] = { case ((xm, xv), (ym, yv)) =>
      Code.memoize(xm, "cord_lift_missing_xm",
        ym, "cord_lift_missing_ym") { (xm, ym) =>
        (xm || ym).mux(whenMissing(xm, ym), op(xv, yv))
      }
  }

  val compare: (P, P) => Code[Int] =
    liftMissing(compareNonnull, (xm, ym) =>
      Code.memoize(xm, "code_ord_compare_xm", ym, "code_ord_compare_ym") { (xm, ym) =>
        (xm && ym).mux(0, xm.mux(1, -1))
      })
  val lt: (P, P) => Code[Boolean] =
    liftMissing(ltNonnull, (xm, _) => !xm)
  val lteq: (P, P) => Code[Boolean] =
    liftMissing(lteqNonnull, (xm, ym) => !xm || ym)
  val gt: (P, P) => Code[Boolean] =
    liftMissing(gtNonnull, (_, ym) => !ym)
  val gteq: (P, P) => Code[Boolean] =
    liftMissing(gteqNonnull, (xm, ym) => !ym || xm)
  val equiv: (P, P) => Code[Boolean] =
    liftMissing(equivNonnull, (xm, ym) => xm && ym)

  // reverses the sense of the non-null comparison only
  def reverse: CodeOrdering = new CodeOrdering () {
    override def reverse: CodeOrdering = CodeOrdering.this
    override type T = CodeOrdering.this.T
    override type P = CodeOrdering.this.P

    override def compareNonnull(x: Code[T], y: Code[T]) = CodeOrdering.this.compareNonnull(y, x)
    override def ltNonnull(x: Code[T], y: Code[T]) = CodeOrdering.this.ltNonnull(y, x)
    override def lteqNonnull(x: Code[T], y: Code[T]) = CodeOrdering.this.lteqNonnull(y, x)
    override def gtNonnull(x: Code[T], y: Code[T]) = CodeOrdering.this.gtNonnull(y, x)
    override def gteqNonnull(x: Code[T], y: Code[T]) = CodeOrdering.this.gteqNonnull(y, x)
    override def equivNonnull(x: Code[T], y: Code[T]) = CodeOrdering.this.equivNonnull(y, x)
  }
}

abstract class CodeOrderingCompareConsistentWithOthers extends CodeOrdering {
  def ltNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) < 0

  def lteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) <= 0

  def gtNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) > 0

  def gteqNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y) >= 0

  def equivNonnull(x: Code[T], y: Code[T]): Code[Boolean] = compareNonnull(x, y).ceq(0)
}
