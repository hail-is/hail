package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.types.virtual.TBoolean
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

    val ir3 = TableAggregate(tr, invokeSeeded("rand_bool", 1L, TBoolean, F64(0.5)))
    val m3 = ContainsSeededRandomness.analyze(ir3)
    assert(m3.get(ir3.query).contains(true))
    assert(m3.get(ir3).contains(false))
  }
}
