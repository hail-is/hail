package is.hail.expr.ir.lowering

import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.{Literal, ToArray, ToStream}
import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir.lowering.LowerDistributedSort.samplePartition
import is.hail.types.virtual.{TArray, TInt32, TStruct}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LowerDistributedSortSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.compileOnly

  @Test def testSamplePartition() {
    val dataKeys = IndexedSeq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    val elementType = TStruct(("key", TInt32), ("value", TInt32))
    val data1 = ToStream(Literal(TArray(elementType), dataKeys.map(x => Row(x, x * x))))
    val sampleSeq = ToStream(Literal(TArray(TInt32), IndexedSeq(0, 2, 3, 7)))

    val sampled = ToArray(samplePartition(data1, sampleSeq))

    assertEvalsTo(sampled, null)
  }
}
