package is.hail.annotations

import is.hail.asm4s._
import is.hail.expr.ir.{Ascending, Descending, EmitMethodBuilder, EmitCode, EmitCodeBuilder, SortField, SortOrder}
import is.hail.types._
import is.hail.asm4s.coerce
import is.hail.types.physical._
import is.hail.types.physical.stypes.interfaces._
import is.hail.utils._

object CodeOrdering {

  sealed trait Op {
    type ReturnType
    val rtti: TypeInfo[ReturnType]
    val missingEqual: Boolean
  }
  final case class Compare(missingEqual: Boolean = true) extends Op {
    type ReturnType = Int
    val rtti = typeInfo[Int]
  }
  sealed trait BooleanOp extends Op {
    type ReturnType = Boolean
    val rtti = typeInfo[Boolean]
  }
  final case class Equiv(missingEqual: Boolean = true) extends BooleanOp
  final case class Lt(missingEqual: Boolean = true) extends BooleanOp
  final case class Lteq(missingEqual: Boolean = true) extends BooleanOp
  final case class Gt(missingEqual: Boolean = true) extends BooleanOp
  final case class Gteq(missingEqual: Boolean = true) extends BooleanOp
  final case class Neq(missingEqual: Boolean = true) extends BooleanOp

  type F[R] = (EmitCodeBuilder, EmitCode, EmitCode) => Code[R]

  def rowOrdering(
    t1: PBaseStruct,
    t2: PBaseStruct,
    mb: EmitMethodBuilder[_],
    sortOrders: Array[SortOrder] = null,
    missingFieldsEqual: Boolean = true
  ): CodeOrdering = new CodeOrdering {
    require(sortOrders == null || sortOrders.size == t1.size)

    def setup(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode):  (PBaseStructValue, PBaseStructValue) = {
      lhs.asBaseStruct.memoize(cb, "structord_lhs") -> rhs.asBaseStruct.memoize(cb, "structord_rhs")
    }

    private[this] def fieldOrdering(i: Int, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
      mb.getCodeOrdering(
        t1.types(i),
        t2.types(i),
        if (sortOrders == null) Ascending else sortOrders(i),
        op)

    override def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
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

    override def ltNonnull(cb:EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
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

    override def lteqNonnull(cb:EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
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

    override def gtNonnull(cb:EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
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

    override def gteqNonnull(cb:EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
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

    override def equivNonnull(cb:EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
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

  def iterableOrdering(t1: PArray, t2: PArray, mb: EmitMethodBuilder[_]): CodeOrdering = new CodeOrdering {
    private[this] def setup(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): (PIndexableValue, PIndexableValue) = {
      val lhsv = lhs.asIndexable.memoize(cb, "container_ord_lhs")
      val rhsv = rhs.asIndexable.memoize(cb, "container_ord_rhs")
      lhsv -> rhsv
    }

    private[this] def loop(cb: EmitCodeBuilder, lhs: PIndexableValue, rhs: PIndexableValue)(
      f: (EmitCode, EmitCode) => Unit
    ): Unit = {
      val i = cb.newLocal[Int]("i")
      val lim = cb.newLocal("lim", lhs.loadLength().min(rhs.loadLength()))
      cb.forLoop(cb.assign(i, 0), i < lim, cb.assign(i, i + 1), {
        val left = EmitCode.fromI(cb.emb)(lhs.loadElement(_, i).typecast)
        val right = EmitCode.fromI(cb.emb)(rhs.loadElement(_, i).typecast)
        f(left, right)
      })
    }

    override def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
      val elemCmp = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Compare())

      val Lout = CodeLabel()
      val cmp = cb.newLocal[Int]("iterable_cmp", 0)

      val (lhs, rhs) = setup(cb, x, y)
      loop(cb, lhs, rhs) { (lhs, rhs) =>
        cb.assign(cmp, elemCmp(cb, lhs, rhs))
        cb.ifx(cmp.cne(0), cb.goto(Lout))
      }

      // if we get here, cmp is 0
      cb.assign(cmp,
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int](
          "compare", lhs.loadLength(), rhs.loadLength()))
      cb.define(Lout)
      cmp
    }

    override def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemLt = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Lt())
      val elemEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_lt")
      val Lout = CodeLabel()

      val (lhs, rhs) = setup(cb, x, y)
      val lt = cb.newLocal("lt", false)
      val eq = cb.newLocal("eq", true)

      loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
        val lhs = cb.memoize(lhsEC, "lhs_item")
        val rhs = cb.memoize(rhsEC, "rhs_item")
        cb.assign(lt, elemLt(cb, lhs, rhs))
        cb.assign(eq, !lt && elemEq(cb, lhs, rhs))

        cb.ifx(!eq, {
          cb.assign(ret, lt)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() < rhs.loadLength())
      cb.define(Lout)
      ret
    }

    override def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemLtEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Lteq())
      val elemEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_lteq")
      val Lout = CodeLabel()

      val (lhs, rhs) = setup(cb, x, y)
      val lteq = cb.newLocal("lteq", false)
      val eq = cb.newLocal("eq", true)

      loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
        val lhs = cb.memoize(lhsEC, "lhs_item")
        val rhs = cb.memoize(rhsEC, "rhs_item")
        cb.assign(lteq, elemLtEq(cb, lhs, rhs))
        cb.assign(eq, elemEq(cb, lhs, rhs))

        cb.ifx(!eq, {
          cb.assign(ret, lteq)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() <= rhs.loadLength)
      cb.define(Lout)
      ret
    }

    override def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemGt = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Gt())
      val elemEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_gt")
      val Lout = CodeLabel()

      val (lhs, rhs) = setup(cb, x, y)
      val gt = cb.newLocal("gt", false)
      val eq = cb.newLocal("eq", true)

      loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
        val lhs = cb.memoize(lhsEC, "lhs_item")
        val rhs = cb.memoize(rhsEC, "rhs_item")
        cb.assign(gt, elemGt(cb, lhs, rhs))
        cb.assign(eq, !gt && elemEq(cb, lhs, rhs))

        cb.ifx(!eq, {
          cb.assign(ret, gt)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() > rhs.loadLength())
      cb.define(Lout)
      ret
    }

    override def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemGtEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Gteq())
      val elemEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_gteq")
      val Lout = CodeLabel()

      val (lhs, rhs) = setup(cb, x, y)
      val gteq = cb.newLocal("gteq", true)
      val eq = cb.newLocal("eq", true)

      loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
        val lhs = cb.memoize(lhsEC, "lhs_item")
        val rhs = cb.memoize(rhsEC, "rhs_item")
        cb.assign(gteq, elemGtEq(cb, lhs, rhs))
        cb.assign(eq, elemEq(cb, lhs, rhs))

        cb.ifx(!eq, {
          cb.assign(ret, gteq)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() >= rhs.loadLength)
      cb.define(Lout)
      ret
    }

    override def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemEq = mb.getCodeOrdering(t1.elementType, t2.elementType, CodeOrdering.Equiv())
      val ret = cb.newLocal[Boolean]("iterable_eq", true)
      val Lout = CodeLabel()
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      cb.ifx(lhs.loadLength().cne(rhs.loadLength()), exitWith(false))
      loop(cb, lhs, rhs) { (lhs, rhs) =>
        cb.assign(ret, elemEq(cb, lhs, rhs))
        cb.ifx(!ret, cb.goto(Lout))
      }

      cb.define(Lout)
      ret
    }
  }

  def intervalOrdering(t1: PInterval, t2: PInterval, mb: EmitMethodBuilder[_]): CodeOrdering = new CodeOrdering {
    private val setup: (EmitCodeBuilder, PCode, PCode) => (PIntervalValue, PIntervalValue) = {
      case (cb, lhs: PIntervalCode, rhs: PIntervalCode) =>
        lhs.memoize(cb, "intervalord_lhs") -> rhs.memoize(cb, "intervalord_rhs")
    }

    override def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
      val pointCompare = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Compare())
      val cmp = cb.newLocal[Int]("intervalord_cmp", 0)

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast)
      val rstart = EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast)
      cb.assign(cmp, pointCompare(cb, lstart, rstart))
      cb.ifx(cmp.ceq(0), {
        cb.ifx(lhs.includesStart().cne(rhs.includesStart()), {
          cb.assign(cmp, lhs.includesStart().mux(-1, 1))
        }, {
          val lend = EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast)
          val rend = EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast)
          cb.assign(cmp, pointCompare(cb, lend, rend))
          cb.ifx(cmp.ceq(0), {
            cb.ifx(lhs.includesEnd().cne(rhs.includesEnd()), {
              cb.assign(cmp, lhs.includesEnd().mux(1, -1))
            })
          })
        })
      })

      cmp
    }

    override def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_eq", true)
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)

      cb.ifx(lhs.includesStart().cne(rhs.includesStart()) ||
        lhs.includesEnd().cne(rhs.includesEnd()), {
          exitWith(false)
        })

      val lstart = EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast)
      val rstart = EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast)
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(false))

      val lend = EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast)
      val rend = EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast)
      cb.ifx(!pointEq(cb, lend, rend), exitWith(false))

      cb.define(Lout)
      ret
    }

    override def ltNonnull(cb:EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointLt = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Lt())
      val pointEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_lt")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(pointLt(cb, lstart, rstart), exitWith(true))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(lhs.includesStart() && !rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")

      cb.ifx(pointLt(cb, lend, rend), exitWith(true))
      cb.assign(ret, pointEq(cb, lend, rend) && !lhs.includesEnd() && rhs.includesEnd())

      cb.define(Lout)
      ret
    }

    override def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointLtEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Lteq())
      val pointEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_lteq")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(!pointLtEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(true))
      cb.ifx(lhs.includesStart() && !rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")
      cb.ifx(!pointLtEq(cb, lend, rend), exitWith(false))
      cb.assign(ret, !pointEq(cb, lend, rend) || !lhs.includesEnd() || rhs.includesEnd())

      cb.define(Lout)
      ret
    }

    override def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointGt = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Gt())
      val pointEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_gt")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(pointGt(cb, lstart, rstart), exitWith(true))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(!lhs.includesStart() && rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")

      cb.ifx(pointGt(cb, lend, rend), exitWith(true))
      cb.assign(ret, pointEq(cb, lend, rend) && lhs.includesEnd() && !rhs.includesEnd())

      cb.define(Lout)
      ret
    }

    override def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointGtEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Gteq())
      val pointEq = mb.getCodeOrdering(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_gteq")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(!pointGtEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(true))
      cb.ifx(!lhs.includesStart() && rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")
      cb.ifx(!pointGtEq(cb, lend, rend), exitWith(false))
      cb.assign(ret, !pointEq(cb, lend, rend) || lhs.includesEnd() || !rhs.includesEnd())

      cb.define(Lout)
      ret
    }
  }

  def locusOrdering(t1: PLocus, t2: PLocus, mb: EmitMethodBuilder[_]): CodeOrdering =
    new CodeOrderingCompareConsistentWithOthers {
      require(t1.rg == t2.rg)

      def compareNonnull(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Int] = {
        val codeRG = mb.getReferenceGenome(t1.rg)
        val lhs: PLocusValue = lhsc.asLocus.memoize(cb, "locus_cmp_lhs")
        val rhs: PLocusValue = rhsc.asLocus.memoize(cb, "locus_cmp_rhs")
        val lhsContig = lhs.contig(cb).memoize(cb, "locus_cmp_lcontig").asInstanceOf[SStringValue]
        val rhsContig = rhs.contig(cb).memoize(cb, "locus_cmp_rcontig").asInstanceOf[SStringValue]

        // ugh
        val lhsContigBinType = lhsContig.get.asBytes().st.pType.asInstanceOf[PBinary]
        val rhsContigBinType = rhsContig.get.asBytes().st.pType.asInstanceOf[PBinary]
        val bincmp = lhsContigBinType.codeOrdering(mb, rhsContigBinType)

        val ret = cb.newLocal[Int]("locus_cmp_ret", 0)
        cb.ifx(bincmp.compareNonnull(cb,
            lhsContig.get.asBytes().asPCode,
            rhsContig.get.asBytes().asPCode).ceq(0), {
          cb.assign(ret, Code.invokeStatic2[java.lang.Integer, Int, Int, Int](
            "compare", lhs.position(cb), rhs.position(cb)))
        }, {
          cb.assign(ret, codeRG.invoke[String, String, Int](
            "compare", lhsContig.get.loadString(), rhsContig.get.loadString()))
        })
        ret
      }
    }

  def mapOrdering(t1: PDict, t2: PDict, mb: EmitMethodBuilder[_]): CodeOrdering =
    iterableOrdering(PCanonicalArray(t1.elementType, t1.required), PCanonicalArray(t2.elementType, t2.required), mb)

  def setOrdering(t1: PSet, t2: PSet, mb: EmitMethodBuilder[_]): CodeOrdering =
    iterableOrdering(PCanonicalArray(t1.elementType, t1.required), PCanonicalArray(t2.elementType, t2.required), mb)

}

abstract class CodeOrdering {
  outer =>

  def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int]

  def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean]

  def compare(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean = true): Code[Int] = {
    cb += x.setup
    cb += y.setup
    val xm = cb.newLocal("cord_compare_xm", x.m)
    val ym = cb.newLocal("cord_compare_ym", y.m)
    val cmp = cb.newLocal[Int]("cmp")
    cb.ifx(xm,
      cb.ifx(ym, cb.assign(cmp, if (missingEqual) 0 else -1), cb.assign(cmp, 1)),
      cb.ifx(ym, cb.assign(cmp, -1), cb.assign(cmp, compareNonnull(cb, x.pv, y.pv))))
    cmp
  }

  def lt(cb: EmitCodeBuilder, x:EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("lt")
    cb += x.setup
    cb += y.setup
    if (missingEqual) {
      cb.ifx(x.m,
        cb.assign(ret, false),
        cb.ifx(y.m,
          cb.assign(ret, true),
          cb.assign(ret, ltNonnull(cb, x.pv, y.pv))))
    } else {
      cb.ifx(y.m,
        cb.assign(ret, true),
        cb.ifx(x.m,
          cb.assign(ret, false),
          cb.assign(ret, ltNonnull(cb, x.pv, y.pv))))
    }
    ret
  }

  def lteq(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("lteq")
    cb += x.setup
    cb += y.setup
    cb.ifx(y.m,
      cb.assign(ret, true),
      cb.ifx(x.m,
        cb.assign(ret, false),
        cb.assign(ret, lteqNonnull(cb, x.pv, y.pv))))
    ret
  }

  def gt(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("gt")
    cb += x.setup
    cb += y.setup
    cb.ifx(y.m,
      cb.assign(ret, false),
      cb.ifx(x.m,
        cb.assign(ret, true),
        cb.assign(ret, gtNonnull(cb, x.pv, y.pv))))
    ret
  }

  def gteq(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("gteq")
    cb += x.setup
    cb += y.setup
    if (missingEqual) {
      cb.ifx(x.m,
        cb.assign(ret, true),
        cb.ifx(y.m,
          cb.assign(ret, false),
          cb.assign(ret, gteqNonnull(cb, x.pv, y.pv))))
    } else {
      cb.ifx(y.m,
        cb.assign(ret, false),
        cb.ifx(x.m,
          cb.assign(ret, true),
          cb.assign(ret, gteqNonnull(cb, x.pv, y.pv))))
    }
    ret
  }

  def equiv(cb: EmitCodeBuilder, x: EmitCode, y: EmitCode, missingEqual: Boolean): Code[Boolean] = {
    val ret = cb.newLocal[Boolean]("eq")
    cb += x.setup
    cb += y.setup
    if (missingEqual) {
      val xm = cb.newLocal("cord_equiv_xm", x.m)
      val ym = cb.newLocal("cord_equiv_ym", y.m)
      cb.ifx(xm && ym,
        cb.assign(ret, true),
        cb.ifx(!xm && !ym,
          cb.assign(ret, equivNonnull(cb, x.pv, y.pv)),
          cb.assign(ret, false)))
    } else {
      cb.ifx(!x.m && !y.m, cb.assign(ret, equivNonnull(cb, x.pv, y.pv)), cb.assign(ret, false))
    }
    ret
  }

  // reverses the sense of the non-null comparison only
  def reverse: CodeOrdering = new CodeOrdering () {
    override def reverse: CodeOrdering = CodeOrdering.this

    override def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode) = CodeOrdering.this.compareNonnull(cb, y, x)
    override def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode) = CodeOrdering.this.ltNonnull(cb, y, x)
    override def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode) = CodeOrdering.this.lteqNonnull(cb, y, x)
    override def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode) = CodeOrdering.this.gtNonnull(cb, y, x)
    override def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode) = CodeOrdering.this.gteqNonnull(cb, y, x)
    override def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode) = CodeOrdering.this.equivNonnull(cb, y, x)
  }
}

abstract class CodeOrderingCompareConsistentWithOthers extends CodeOrdering {
  def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) < 0

  def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) <= 0

  def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) > 0

  def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y) >= 0

  def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = compareNonnull(cb, x, y).ceq(0)
}
