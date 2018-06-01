package is.hail.vds

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.expr.types.{TLocus, TStruct}
import is.hail.table.Table
import is.hail.testUtils._
import is.hail.variant.{Locus, MatrixTable, ReferenceGenome}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.language.implicitConversions

class JoinSuite extends SparkSuite {
  @Test def testIterator() {
    val leftVariants = Array(
      Locus("1", 1),
      Locus("1", 2),
      Locus("1", 4),
      Locus("1", 4),
      Locus("1", 5),
      Locus("1", 5),
      Locus("1", 9),
      Locus("1", 13),
      Locus("1", 13),
      Locus("1", 14),
      Locus("1", 15))

    val rightVariants = Array(
      Locus("1", 1),
      Locus("1", 1),
      Locus("1", 1),
      Locus("1", 3),
      Locus("1", 4),
      Locus("1", 4),
      Locus("1", 6),
      Locus("1", 6),
      Locus("1", 8),
      Locus("1", 9),
      Locus("1", 13),
      Locus("1", 15))

    val vType = TLocus(ReferenceGenome.GRCh37)
    val leftKt = Table(hc, sc.parallelize(leftVariants.map(Row(_))), TStruct("v" -> vType)).keyBy("v")
    leftKt.typeCheck()
    val left = MatrixTable.fromRowsTable(leftKt)

    val rightKt = Table(hc, sc.parallelize(rightVariants.map(Row(_))), TStruct("v" -> vType)).keyBy("v")
    rightKt.typeCheck()
    val right = MatrixTable.fromRowsTable(rightKt)

    val localRowType = left.rvRowType

    // Inner distinct ordered join
    val jInner = left.rvd.orderedJoinDistinct(right.rvd, "inner", (_, it) => it.map(_._1), left.rvd.typ)
    val jInnerOrdRDD1 = left.rdd.join(right.rdd.distinct)

    assert(jInner.count() == jInnerOrdRDD1.count())
    assert(jInner.map { rv =>
      val r = SafeRow(localRowType, rv)
      r.getAs[Locus](0)
    }.collect() sameElements jInnerOrdRDD1.map(_._1.asInstanceOf[Row].get(0)).collect().sorted(vType.ordering.toOrdering))

    // Left distinct ordered join
    val jLeft = left.rvd.orderedJoinDistinct(right.rvd, "left", (_, it) => it.map(_._1), left.rvd.typ)
    val jLeftOrdRDD1 = left.rdd.leftOuterJoin(right.rdd.distinct)

    assert(jLeft.count() == jLeftOrdRDD1.count())
    assert(jLeft.forall(rv => rv != null))
    assert(jLeft.map { rv =>
      val r = SafeRow(localRowType, rv)
      r.getAs[Locus](0)
    }.collect() sameElements jLeftOrdRDD1.map(_._1.asInstanceOf[Row].get(0)).collect().sorted(vType.ordering.toOrdering))
  }
}
