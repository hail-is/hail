package is.hail.expr.ir.orderings

import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.PCode
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue}

object BinaryOrdering {
  def make(t1: SBinary, t2: SBinary, ecb: EmitClassBuilder[_]): CodeOrdering = {

    new CodeOrderingCompareConsistentWithOthers {

      val type1: SBinary = t1
      val type2: SBinary = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
        val xv: SBinaryValue = x.asBinary.memoize(cb, "xv")
        val yv: SBinaryValue = y.asBinary.memoize(cb, "yv")
        val xlen = cb.newLocal[Int]("xlen", xv.loadLength())
        val ylen = cb.newLocal[Int]("ylen", yv.loadLength())
        val lim = cb.newLocal[Int]("lim", (xlen < ylen).mux(xlen, ylen))
        val i = cb.newLocal[Int]("i", 0)
        val cmp = cb.newLocal[Int]("cmp", 0)
        val Lbreak = CodeLabel()

        cb.forLoop({}, i < lim, cb.assign(i, i + 1), {
          val compval = Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare",
            Code.invokeStatic1[java.lang.Byte, Byte, Int]("toUnsignedInt", xv.loadByte(i)),
            Code.invokeStatic1[java.lang.Byte, Byte, Int]("toUnsignedInt", yv.loadByte(i)))
          cb.assign(cmp, compval)
          cb.ifx(cmp.cne(0), cb.goto(Lbreak))
        })

        cb.define(Lbreak)
        cmp.ceq(0).mux(Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", xlen, ylen), cmp)
      }
    }
  }
}
