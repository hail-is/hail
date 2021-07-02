package is.hail.expr.ir.orderings

import is.hail.asm4s.{Code, CodeLabel}
import is.hail.expr.ir.{Ascending, EmitClassBuilder, EmitCode, EmitCodeBuilder, SortOrder}
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.physical.{PBaseStructValue, PCode}

object StructOrdering {
  def make(
    t1: SBaseStruct,
    t2: SBaseStruct,
    ecb: EmitClassBuilder[_],
    sortOrders: Array[SortOrder] = null,
    missingFieldsEqual: Boolean = true
  ): CodeOrdering = new CodeOrdering {

    val type1: SBaseStruct = t1
    val type2: SBaseStruct = t2

    require(sortOrders == null || sortOrders.size == t1.size)

    def setup(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): (PBaseStructValue, PBaseStructValue) = {
      lhs.asBaseStruct.memoize(cb, "structord_lhs") -> rhs.asBaseStruct.memoize(cb, "structord_rhs")
    }

    private[this] def fieldOrdering(i: Int, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
      ecb.getOrderingFunction(t1.fieldTypes(i), t2.fieldTypes(i),
        if (sortOrders == null) Ascending else sortOrders(i),
        op)

    override def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
      val (lhs, rhs) = setup(cb, x, y)
      val Lout = CodeLabel()
      val cmp = cb.newLocal("cmp", 0)

      var i = 0
      while (i < t1.size) {
        val fldCmp = fieldOrdering(i, CodeOrdering.Compare(missingFieldsEqual))
        val l = EmitCode.fromI(cb.emb) { cb => lhs.loadField(cb, i).typecast }
        val r = EmitCode.fromI(cb.emb) { cb => rhs.loadField(cb, i).typecast }
        cb.assign(cmp, fldCmp(cb, l, r))
        cb.ifx(cmp.cne(0), cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      cmp
    }

    override def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val (lhs, rhs) = setup(cb, x, y)
      val Lout = CodeLabel()
      val lt = cb.newLocal("lt", true)
      val eq = cb.newLocal("eq", true)

      var i = 0
      while (i < t1.size) {
        val fldLt = fieldOrdering(i, CodeOrdering.Lt(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhs.loadField(cb, i).typecast }, s"struct_lt_lhs_fld$i")
        val r = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhs.loadField(cb, i).typecast }, s"struct_lt_rhs_fld$i")
        cb.assign(lt, fldLt(cb, l, r))
        cb.assign(eq, !lt && fldEq(cb, l, r))
        cb.ifx(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      lt
    }

    override def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val (lhs, rhs) = setup(cb, x, y)
      val Lout = CodeLabel()
      val lteq = cb.newLocal("lteq", true)
      val eq = cb.newLocal("eq", true)

      var i = 0
      while (i < t1.size) {
        val fldLtEq = fieldOrdering(i, CodeOrdering.Lteq(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhs.loadField(cb, i).typecast }, s"struct_lteq_lhs_fld$i")
        val r = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhs.loadField(cb, i).typecast }, s"struct_lteq_rhs_fld$i")
        cb.assign(lteq, fldLtEq(cb, l, r))
        cb.assign(eq, fldEq(cb, l, r))
        cb.ifx(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      lteq
    }

    override def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val (lhs, rhs) = setup(cb, x, y)
      val Lout = CodeLabel()
      val gt = cb.newLocal("gt", false)
      val eq = cb.newLocal("eq", true)

      var i = 0
      while (i < t1.size) {
        val fldGt = fieldOrdering(i, CodeOrdering.Gt(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhs.loadField(cb, i).typecast }, s"struct_gt_lhs_fld$i")
        val r = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhs.loadField(cb, i).typecast }, s"struct_gt_rhs_fld$i")
        cb.assign(gt, fldGt(cb, l, r))
        cb.assign(eq, !gt && fldEq(cb, l, r))
        cb.ifx(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      gt
    }

    override def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val (lhs, rhs) = setup(cb, x, y)
      val Lout = CodeLabel()
      val gteq = cb.newLocal("gteq", true)
      val eq = cb.newLocal("eq", true)

      var i = 0
      while (i < t1.size) {
        val fldGtEq = fieldOrdering(i, CodeOrdering.Gteq(missingFieldsEqual))
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))

        val l = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhs.loadField(cb, i).typecast }, s"struct_gteq_lhs_fld$i")
        val r = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhs.loadField(cb, i).typecast }, s"struct_gteq_rhs_fld$i")
        cb.assign(gteq, fldGtEq(cb, l, r))
        cb.assign(eq, fldEq(cb, l, r))
        cb.ifx(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      gteq
    }

    override def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val (lhs, rhs) = setup(cb, x, y)
      val Lout = CodeLabel()
      val eq = cb.newLocal("cmp", true)

      var i = 0
      while (i < t1.size) {
        val fldEq = fieldOrdering(i, CodeOrdering.Equiv(missingFieldsEqual))
        val l = EmitCode.fromI(cb.emb) { cb => lhs.loadField(cb, i).typecast }
        val r = EmitCode.fromI(cb.emb) { cb => rhs.loadField(cb, i).typecast }
        cb.assign(eq, fldEq(cb, l, r))
        cb.ifx(!eq, cb.goto(Lout))
        i += 1
      }

      cb.define(Lout)
      eq
    }
  }
}
