package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.expr.ir.agg.TakeByRVAS
import is.hail.expr.types.physical._
import is.hail.utils._
import org.testng.annotations.Test

class TakeByAggregatorSuite extends HailSuite {
  @Test def test() {
    val fb = EmitFunctionBuilder[Region, Long]

    val tba = new TakeByRVAS(PStringRequired, PInt64Optional, PArray(PStringRequired, required = false), fb)

    val stop = 10000000L
    val ret = Region.scoped { r =>
      val argR = fb.getArg[Region](1).load()
      val i = fb.newField[Long]
      val rt = tba.resultType

      fb.emit(Code(
        tba.createState,
        tba.newState,
        tba.initialize(1000),
        i := 0L,
        Code.whileLoop(i < stop,
          argR.invoke[Unit]("clear"),
          tba.seqOp(false, argR.appendBinary(i.toS.invoke[Array[Byte]]("getBytes")),
            (i & const(0xffL)).ceq(0L), -i),
          i := i + 1L
        ),
        tba.result(argR, rt)
      ))

      val o = fb.resultWithIndex()(0, r)(r)
      SafeRow.read(rt, r, o)
    }
    assert(ret == ((stop - 1) to 0 by -1)
      .iterator
      .filter(l => (l & 0xffL) != 0)
      .map(_.toString)
      .take(1000)
      .toFastIndexedSeq)
  }
}
