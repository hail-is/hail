package is.hail.nativecode

import is.hail.SparkSuite
import is.hail.cxx._
import is.hail.expr.ir
import is.hail.io.CodecSpec
import is.hail.table.Table
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class TableEmitSuite extends SparkSuite {

  @Test def testReadMapFilterWrite(): Unit = {

    val readPath = tmpDir.createTempFile("test", "ht")
    val writePath = tmpDir.createTempFile("test", "ht")

    val rt = new Table(hc, ir.TableRange(10, 2))
    rt.write(readPath)

    val read = ir.TableIR.read(hc, readPath, false, None)
    def tir(orig: ir.TableIR): ir.TableIR =
      ir.TableFilter(
        ir.TableMapRows(orig,
          ir.InsertFields(ir.Ref("row", read.typ.rowType), FastIndexedSeq("x" -> ir.I32(0)))),
        ir.True())

    val tub = new TranslationUnitBuilder()
    TableEmit(tub, tir(read)).write(tub, writePath, true, false, CodecSpec.default.toString)

    val expected = new Table(hc, tir(rt.tir))
    val result = Table.read(hc, writePath)
    assert(result.same(expected))
  }
}
