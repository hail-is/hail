package is.hail.backend.spark

import is.hail.SparkSuite
import is.hail.expr.ir
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class SparkBackendSuite extends SparkSuite {

  @Test def testRangeCount() {
    val node = ir.ApplyBinaryPrimOp(ir.Add(), ir.TableCount(ir.TableRange(10, 2)), ir.TableCount(ir.TableRange(15, 5)))
    assert(SparkBackend.execute(hc.sc, node, optimize = false) == 25)
  }

  @Test def testRangeCollect() {
    val t = ir.TableRange(10, 2)
    val row = ir.Ref("row", t.typ.rowType)
    val node = ir.TableCollect(ir.TableMapRows(t, ir.InsertFields(row, FastIndexedSeq("x" -> ir.GetField(row, "idx")))))
    assert(SparkBackend.execute(hc.sc, node, optimize = false) == Array.tabulate(10)(i => Row(i, i)).toFastIndexedSeq)
  }

  @Test def testGetGlobals() {
    val t = ir.TableRange(10, 2)
    val newGlobals = ir.InsertFields(ir.Ref("global", t.typ.globalType), FastSeq("x" -> ir.TableCollect(t)))
    val node = ir.TableGetGlobals(ir.TableMapGlobals(t, newGlobals))

    assert(SparkBackend.execute(hc.sc, node, optimize = false) == Row(Array.tabulate(10)(i => Row(i)).toFastIndexedSeq))
  }

  @Test def testCollectGlobals() {
    val t = ir.TableRange(10, 2)
    val newGlobals = ir.InsertFields(ir.Ref("global", t.typ.globalType), FastSeq("x" -> ir.TableCollect(t)))
    val node =ir.TableMapRows(
      ir.TableMapGlobals(t, newGlobals),
      ir.InsertFields(ir.Ref("row", t.typ.rowType), FastSeq("x" -> ir.GetField(ir.Ref("global", newGlobals.typ), "x"))))

    val x = Array.tabulate(10)(i => Row(i)).toFastIndexedSeq
    val expected = Array.tabulate(10)(i => Row(i, x)).toFastIndexedSeq

    assert(SparkBackend.execute(hc.sc, ir.TableCollect(node), optimize = false) == expected)
  }
}
