package is.hail.variant.vsm

import breeze.linalg.DenseMatrix
import is.hail.annotations._
import is.hail.check.Prop._
import is.hail.check.Parameters
import is.hail.linalg.BlockMatrix
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import is.hail.{SparkSuite, TestUtils}
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.testng.annotations.Test

import scala.language.postfixOps

class VSMSuite extends SparkSuite {

  @Test def testWriteRead() {
    val p = forAll(MatrixTable.gen(hc, VSMSubgen.random).map(_.annotateRowsExpr("foo" -> "5"))) { vds =>
      ReferenceGenome.addReference(vds.referenceGenome)
      val f = tmpDir.createTempFile(extension = "vds")
      vds.write(f)
      val result = hc.readVDS(f).same(vds)
      ReferenceGenome.removeReference(vds.referenceGenome.name)
      result
    }

    p.check()
  }

  @Test def testSkipGenotypes() {
    val f = tmpDir.createTempFile("sample", extension = ".vds")
    hc.importVCF("src/test/resources/sample2.vcf")
      .write(f)

    assert(hc.read(f, dropCols = true)
      .filterRowsExpr("va.info.AF[0] < 0.01")
      .countRows() == 234)
  }

  @Test def testSkipDropSame() {
    val f = tmpDir.createTempFile("sample", extension = ".vds")

    hc.importVCF("src/test/resources/sample2.vcf")
      .write(f)

    assert(hc.readVDS(f, dropSamples = true)
      .same(hc.readVDS(f).dropCols()))
  }

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
    val vds = hc.importVCF("src/test/resources/sample2.vcf")

    vds.write(out)

    TestUtils.interceptFatal("""file already exists""") {
      vds.write(out)
    }

    vds.write(out, overwrite = true)
  }

  @Test def testInvalidMetadata() {
    TestUtils.interceptFatal("metadata does not contain file version") {
      hc.readVDS("src/test/resources/0.1-1fd5cc7.vds").count()
    }
  }

  @Test def testAnnotateVariantsKeyTable() {
    forAll(MatrixTable.gen(hc, VSMSubgen.random)) { vds =>
      val vds2 = vds.annotateRowsExpr("bar" -> "va")
      val kt = vds2.rowsTable()
      val resultVds = vds2.annotateRowsTable(kt, "foo")
        .annotateRowsExpr("foo" -> "va.foo.bar")
      resultVds.typecheck()
      val result = resultVds.rdd.collect()
      val (_, getFoo) = resultVds.queryVA("va.foo")
      val (_, getBar) = resultVds.queryVA("va.bar")

      result.forall { case (v, (va, gs)) =>
        getFoo(va) == getBar(va)
      }
    }.check()
  }

  @Test def testQueryGenotypes() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
    vds.aggregateEntries("AGG.map(g => g.GQ.toFloat64).hist(0d, 100d, 100)")
  }

  @Test def testReorderSamples() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
    val origOrder = Array[Annotation]("C1046::HG02024", "C1046::HG02025", "C1046::HG02026", "C1047::HG00731", "C1047::HG00732")
    val newOrder = Array[Annotation]("C1046::HG02026", "C1046::HG02024", "C1047::HG00732", "C1046::HG02025", "C1047::HG00731")

    val filteredVds = vds.filterColsList(origOrder.map(Annotation(_)).toSet)
      .indexCols("colIdx")
      .annotateRowsExpr("origGenos" -> "AGG.take(5)")
    val reorderedVds = filteredVds.reorderCols(newOrder.map(Annotation(_)))
      .annotateRowsExpr("newGenos" -> "AGG.takeBy(g => sa.colIdx, 5)")

    assert(reorderedVds.rowsTable().forall("row.origGenos == row.newGenos"))
    assert(vds.reorderCols(vds.colKeys.toArray).same(vds))
  }
  
  @Test def testWriteBlockMatrix() {
    val dirname = tmpDir.createTempFile()
    
    for {
      numSlices <- Seq(1, 2, 4, 9, 11)
      blockSize <- Seq(1, 2, 3, 4, 6, 7, 9, 10)
    } {
      val mt = hc.baldingNicholsModel(1, 6, 9, Some(numSlices), seed = blockSize + numSlices)
        .indexRows("rowIdx")
        .indexCols("colIdx")
      
      mt.selectEntries("{x: (g.GT.nNonRefAlleles().toInt64 + va.rowIdx + sa.colIdx.toInt64 + 1L).toFloat64}")
        .writeBlockMatrix(dirname, "x", blockSize)

      val data = mt.entriesTable()
          .select("{x: row.GT.nNonRefAlleles().toFloat64 + row.rowIdx.toFloat64 + row.colIdx.toFloat64 + 1.0}", None, None)
          .collect()
          .map(_.getAs[Double](0))

      val lm = new DenseMatrix[Double](6, 9, data).t // data is row major
      
      assert(BlockMatrix.read(hc, dirname).toBreezeMatrix() === lm)
    }
  }
}
