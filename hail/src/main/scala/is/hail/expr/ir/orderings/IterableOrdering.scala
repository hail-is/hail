package is.hail.expr.ir.orderings

import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitValue}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableValue}

object IterableOrdering {

  def make(t1: SContainer, t2: SContainer, ecb: EmitClassBuilder[_]): CodeOrdering = new CodeOrdering {

    override val type1: SContainer = t1
    override val type2: SContainer = t2

    private[this] def loop(cb: EmitCodeBuilder, lhs: SIndexableValue, rhs: SIndexableValue)(
      f: (EmitValue, EmitValue) => Unit
    ): Unit = {
      val i = cb.newLocal[Int]("i")
      val lim = cb.newLocal("lim", lhs.loadLength().min(rhs.loadLength()))
      cb.for_(cb.assign(i, 0), i < lim, cb.assign(i, i + 1), {
        val left = cb.memoize(lhs.loadElement(cb, i))
        val right = cb.memoize(rhs.loadElement(cb, i))
        f(left, right)
      })
    }

    override def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
      val elemCmp = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Compare())

      val Lout = CodeLabel()
      val cmp = cb.newLocal[Int]("iterable_cmp", 0)

      val lhs = x.asIndexable
      val rhs = y.asIndexable
      loop(cb, lhs, rhs) { (lhs, rhs) =>
        cb.assign(cmp, elemCmp(cb, lhs, rhs))
        cb.if_(cmp.cne(0), cb.goto(Lout))
      }

      // if we get here, cmp is 0
      cb.assign(cmp,
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int](
          "compare", lhs.loadLength(), rhs.loadLength()))
      cb.define(Lout)
      cmp
    }

    override def _ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val elemLt = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Lt())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_lt")
      val Lout = CodeLabel()

      val lhs = x.asIndexable
      val rhs = y.asIndexable

      loop(cb, lhs, rhs) { (lhs, rhs) =>
        val lt = elemLt(cb, lhs, rhs)
        val eq = !lt && elemEq(cb, lhs, rhs)

        cb.if_(!eq, {
          cb.assign(ret, lt)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() < rhs.loadLength())
      cb.define(Lout)
      ret
    }

    override def _lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val elemLtEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Lteq())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_lteq")
      val Lout = CodeLabel()

      val lhs = x.asIndexable
      val rhs = y.asIndexable

      loop(cb, lhs, rhs) { (lhs, rhs) =>
        val lteq = elemLtEq(cb, lhs, rhs)
        val eq = elemEq(cb, lhs, rhs)

        cb.if_(!eq, {
          cb.assign(ret, lteq)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() <= rhs.loadLength)
      cb.define(Lout)
      ret
    }

    override def _gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val elemGt = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Gt())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_gt")
      val Lout = CodeLabel()

      val lhs = x.asIndexable
      val rhs = y.asIndexable
      val gt = cb.newLocal("gt", false)
      val eq = cb.newLocal("eq", true)

      loop(cb, lhs, rhs) { (lhs, rhs) =>
        val gt = elemGt(cb, lhs, rhs)
        val eq = !gt && elemEq(cb, lhs, rhs)

        cb.if_(!eq, {
          cb.assign(ret, gt)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() > rhs.loadLength())
      cb.define(Lout)
      ret
    }

    override def _gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val elemGtEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Gteq())
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())

      val ret = cb.newLocal[Boolean]("iterable_gteq")
      val Lout = CodeLabel()

      val lhs = x.asIndexable
      val rhs = y.asIndexable

      loop(cb, lhs, rhs) { (lhs, rhs) =>
        val gteq = elemGtEq(cb, lhs, rhs)
        val eq = elemEq(cb, lhs, rhs)

        cb.if_(!eq, {
          cb.assign(ret, gteq)
          cb.goto(Lout)
        })
      }

      cb.assign(ret, lhs.loadLength() >= rhs.loadLength)
      cb.define(Lout)
      ret
    }

    override def _equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val elemEq = ecb.getOrderingFunction(t1.elementType, t2.elementType, CodeOrdering.Equiv())
      val ret = cb.newLocal[Boolean]("iterable_eq", true)
      val Lout = CodeLabel()
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val lhs = x.asIndexable
      val rhs = y.asIndexable
      cb.if_(lhs.loadLength().cne(rhs.loadLength()), exitWith(false))
      loop(cb, lhs, rhs) { (lhs, rhs) =>
        cb.assign(ret, elemEq(cb, lhs, rhs))
        cb.if_(!ret, cb.goto(Lout))
      }

      cb.define(Lout)
      ret
    }
  }
}
