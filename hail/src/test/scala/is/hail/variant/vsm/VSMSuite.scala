package is.hail.variant.vsm

import breeze.linalg.DenseMatrix
import is.hail.annotations._
import is.hail.check.Prop._
import is.hail.check.Parameters
import is.hail.expr.ir
import is.hail.expr.ir.{Interpret, MatrixRepartition, RepartitionStrategy, TableRead, TableRepartition}
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import is.hail.{HailSuite, TestUtils}
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.testng.annotations.Test

import scala.language.postfixOps

class VSMSuite extends HailSuite {
  @Test(enabled = false) def testVSMGenIsLinearSpaceInSizeParameter() {
    val minimumRSquareValue = 0.7

    def vsmOfSize(size: Int): MatrixTable = {
      val parameters = Parameters.default.copy(size = size, count = 1)
      MatrixTable.gen(hc, VSMSubgen.random).apply(parameters)
    }

    def spaceStatsOf[T](factory: () => T): SummaryStatistics = {
      val sampleSize = 50
      val memories = for (_ <- 0 until sampleSize) yield space(factory())._2

      val stats = new SummaryStatistics
      memories.foreach(x => stats.addValue(x.toDouble))
      stats
    }

    val sizes = 2500 to 20000 by 2500

    val statsBySize = sizes.map(size => (size, spaceStatsOf(() => vsmOfSize(size))))

    println("xs = " + sizes)
    println("mins = " + statsBySize.map { case (_, stats) => stats.getMin })
    println("maxs = " + statsBySize.map { case (_, stats) => stats.getMax })
    println("means = " + statsBySize.map { case (_, stats) => stats.getMean })

    val sr = new SimpleRegression
    statsBySize.foreach { case (size, stats) => sr.addData(size, stats.getMean) }

    println("R² = " + sr.getRSquare)

    assert(sr.getRSquare >= minimumRSquareValue,
      "The VSM generator seems non-linear because the magnitude of the R coefficient is less than 0.9")
  }

  @Test def testOverwrite() {
    val out = tmpDir.createTempFile("out", "vds")
    val vds = TestUtils.importVCF(hc, "src/test/resources/sample2.vcf")

    vds.write(out)

    TestUtils.interceptFatal("""file already exists""") {
      vds.write(out)
    }

    vds.write(out, overwrite = true)
  }

  @Test def testInvalidMetadata() {
    TestUtils.interceptFatal("metadata does not contain file version") {
      hc.readVDS("src/test/resources/0.1-1fd5cc7.vds")
    }
  }

  @Test def testFilesWithRequiredGlobals() {
    val ht = Table.read(hc, "src/test/resources/required_globals.ht").tir
    Interpret(TableRepartition(ht, 10, RepartitionStrategy.SHUFFLE), ctx).rvd.count()

    val mt = MatrixTable.read(hc, "src/test/resources/required_globals.mt").ast
    Interpret(MatrixRepartition(mt, 10, RepartitionStrategy.SHUFFLE), ctx, optimize = false).rvd.count()
  }
}
