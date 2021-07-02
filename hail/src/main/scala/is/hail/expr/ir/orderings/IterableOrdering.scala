package is.hail.expr.ir.orderings

import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.physical.{PCode, PIndexableValue}

object IterableOrdering {

  def make(t1: SContainer, t2: SContainer, ecb: EmitClassBuilder[_]): CodeOrdering = new CodeOrdering {

    val type1: SContainer = t1
    val type2: SContainer = t2

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

    override def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
      val elemCmp = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Compare())

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

    override def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemLt = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Lt())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

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

    override def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemLtEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Lteq())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

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

    override def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemGt = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Gt())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

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

    override def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemGtEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Gteq())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

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

    override def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())
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
}
