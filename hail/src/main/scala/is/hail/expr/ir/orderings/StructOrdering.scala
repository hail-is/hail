package is.hail.expr.ir.orderings

import is.hail.asm4s.{CodeLabel, Value}
import is.hail.expr.ir.{Ascending, EmitClassBuilder, EmitCodeBuilder, SortOrder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.interfaces.SBaseStruct

object StructOrdering {
  def make(
    t1: SBaseStruct,
    t2: SBaseStruct,
    ecb: EmitClassBuilder[_],
    sortOrders: Array[SortOrder] = null,
    missingFieldsEqual: Boolean = true,
  ): CodeOrdering = new CodeOrdering {

    override val type1: SBaseStruct = t1
    override val type2: SBaseStruct = t2

    require(sortOrders == null || sortOrders.size == t1.size)

    private[this] def fieldOrdering(i: Int, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
      ecb.getOrderingFunction(
        t1.fieldTypes(i),
        t2.fieldTypes(i),
        if (sortOrders == null) Ascending else sortOrders(i),
        op,
      )

    override def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
      val lhs = x.asBaseStruct
      val rhs = y.asBaseStruct
      val Lout = CodeLabel()
      val cmp = cb.newLocal("cmp", 0)

      var i = 0
      while (i < t1.size) {
        val fldCmp = fieldOrdering(i, CodeOrdering.Compare(missingFieldsEqual))
        val l = cb.memoize(lhs.loadField(cb, i))
        val r = cb.memoize(rhs.loadField(cb, i))
        cb.assign(cmp, fldCmp(cb, l, r))
        cb.if_(cmp.cne(0), cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      cmp
    }

    override def _ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val lhs = x.asBaseStruct
      val rhs = y.asBaseStruct
      val Lout = CodeLabel()
      val lt = cb.newLocal("lt", true)

      var i = 0
      while (i < t1.size) {
        val fldLt = fieldOrdering(i, CodeOrdering.Lt(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(lhs.loadField(cb, i))
        val r = cb.memoize(rhs.loadField(cb, i))
        cb.assign(lt, fldLt(cb, l, r))
        val eq = !lt && fldEq(cb, l, r)
        cb.if_(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      lt
    }

    override def _lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val lhs = x.asBaseStruct
      val rhs = y.asBaseStruct
      val Lout = CodeLabel()
      val lteq = cb.newLocal("lteq", true)

      var i = 0
      while (i < t1.size) {
        val fldLtEq = fieldOrdering(i, CodeOrdering.Lteq(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(lhs.loadField(cb, i))
        val r = cb.memoize(rhs.loadField(cb, i))
        cb.assign(lteq, fldLtEq(cb, l, r))
        val eq = fldEq(cb, l, r)
        cb.if_(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      lteq
    }

    override def _gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val lhs = x.asBaseStruct
      val rhs = y.asBaseStruct
      val Lout = CodeLabel()
      val gt = cb.newLocal("gt", false)

      var i = 0
      while (i < t1.size) {
        val fldGt = fieldOrdering(i, CodeOrdering.Gt(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(lhs.loadField(cb, i))
        val r = cb.memoize(rhs.loadField(cb, i))
        cb.assign(gt, fldGt(cb, l, r))
        val eq = !gt && fldEq(cb, l, r)
        cb.if_(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      gt
    }

    override def _gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val lhs = x.asBaseStruct
      val rhs = y.asBaseStruct
      val Lout = CodeLabel()
      val gteq = cb.newLocal("gteq", true)

      var i = 0
      while (i < t1.size) {
        val fldGtEq = fieldOrdering(i, CodeOrdering.Gteq(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(lhs.loadField(cb, i))
        val r = cb.memoize(rhs.loadField(cb, i))
        cb.assign(gteq, fldGtEq(cb, l, r))
        val eq = fldEq(cb, l, r)
        cb.if_(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      gteq
    }

    override def _equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val lhs = x.asBaseStruct
      val rhs = y.asBaseStruct
      val Lout = CodeLabel()
      val eq = cb.newLocal("cmp", true)

      var i = 0
      while (i < t1.size) {
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))
        val l = cb.memoize(lhs.loadField(cb, i))
        val r = cb.memoize(rhs.loadField(cb, i))
        cb.assign(eq, fldEq(cb, l, r))
        cb.if_(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      eq
    }
  }
}
