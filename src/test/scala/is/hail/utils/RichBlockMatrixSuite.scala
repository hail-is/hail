package is.hail.utils

import is.hail.SparkSuite
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import is.hail.distributedmatrix.BlockMatrixIsDistributedMatrix._
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix}
import org.testng.annotations.Test

class RichBlockMatrixSuite extends SparkSuite {
  @Test def testToIndexedRowMatrixOrderedPartitioner() {
    val rangeBounds = Array(1, 4, 6)
    val numPartitions = 4

    import is.hail.sparkextras.OrderedKeyIntImplicits.orderedKey

    val opInt = OrderedPartitioner[Int, Int](rangeBounds, numPartitions)

    val bm = from(sc, SparkDenseMatrix.zeros(10, 5), 3, 2)
    
    val irm = bm.toIndexedRowMatrixOrderedPartitioner(opInt)

    val rdd = irm.rows.map { case IndexedRow(i, v) => (i.toInt, v) }

    val check = OrderedRDD(rdd, opInt)
  }
}
