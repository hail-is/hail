package is.hail.expr.ir.orderings

import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue}

object BinaryOrdering {
  def make(t1: SBinary, t2: SBinary, ecb: EmitClassBuilder[_]): CodeOrdering = {

    new CodeOrderingCompareConsistentWithOthers {

      val type1: SBinary = t1
      val type2: SBinary = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
        val xv: SBinaryValue = x.asBinary
        val yv: SBinaryValue = y.asBinary
        val xlen = xv.loadLength(cb)
        val ylen = yv.loadLength(cb)
        val lim = cb.memoize[Int]((xlen < ylen).mux(xlen, ylen))
        val i = cb.newLocal[Int]("i")
        val cmp = cb.newLocal[Int]("cmp", 0)
        val Lbreak = CodeLabel()

        cb.for_(cb.assign(i, 0), i < lim, cb.assign(i, i + 1), {
          val compval = Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare",
            Code.invokeStatic1[java.lang.Byte, Byte, Int]("toUnsignedInt", xv.loadByte(cb, i)),
            Code.invokeStatic1[java.lang.Byte, Byte, Int]("toUnsignedInt", yv.loadByte(cb, i)))
          cb.assign(cmp, compval)
          cb.ifx(cmp.cne(0), cb.goto(Lbreak))
        })

        cb.define(Lbreak)
        cb.ifx(cmp.ceq(0), {
          cb.assign(cmp, Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", xlen, ylen))
        })

        cmp
      }
    }
  }
}
