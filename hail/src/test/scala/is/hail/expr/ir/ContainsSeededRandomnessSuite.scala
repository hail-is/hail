package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.types.virtual.TBoolean
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class ContainsSeededRandomnessSuite extends HailSuite {
  @Test def testAnalysis() {

    val tr = TableRange(10, 10)

    val ir1 = TableMapRows(tr, InsertFields(Ref("row", tr.typ.rowType), Seq(("foo", invokeSeeded("rand_bool", 1L, TBoolean, F64(0.5))))))
    val m1 = ContainsSeededRandomness.analyze(ir1)
    assert(m1.get(ir1.newRow).contains(true))

    val ir2 = TableMapRows(tr, InsertFields(Ref("row", tr.typ.rowType), Seq(("foo", True()))))
    val m2 = ContainsSeededRandomness.analyze(ir2)
    assert(m2.get(ir2.newRow).contains(false))

    val rand = ApplySeeded("rand_bool", FastIndexedSeq(0.5), RNGStateLiteral(), seed=0, TBoolean)
    val ir3 = TableAggregate(tr, rand)
    val m3 = ContainsSeededRandomness.analyze(ir3)
    assert(m3.get(ir3.query).contains(true))
    assert(m3.get(ir3).contains(false))
  }
}
