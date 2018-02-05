package is.hail.variant.vsm

import breeze.linalg.DenseMatrix
import is.hail.annotations._
import is.hail.check.Prop._
import is.hail.check.{Gen, Parameters}
import is.hail.distributedmatrix.{BlockMatrix, KeyedBlockMatrix, Keys}
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import is.hail.{SparkSuite, TestUtils}
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.collection.mutable
import scala.language.postfixOps
import scala.util.Random

class VSMSuite extends SparkSuite {
  @Test def testWriteRead() {
    val p = forAll(MatrixTable.gen(hc, VSMSubgen.random).map(_.annotateVariantsExpr("foo = 5"))) { vds =>
      GenomeReference.addReference(vds.genomeReference)
      val f = tmpDir.createTempFile(extension = "vds")
      vds.write(f)
      val result = hc.readVDS(f).same(vds)
      GenomeReference.removeReference(vds.genomeReference.name)
      result
    }

    p.check()
  }

  @Test def testSkipGenotypes() {
    val f = tmpDir.createTempFile("sample", extension = ".vds")
    hc.importVCF("src/test/resources/sample2.vcf")
      .write(f)

    assert(hc.read(f, dropSamples = true)
      .filterVariantsExpr("va.info.AF[0] < 0.01")
      .countVariants() == 234)
  }

  @Test def testSkipDropSame() {
    val f = tmpDir.createTempFile("sample", extension = ".vds")

    hc.importVCF("src/test/resources/sample2.vcf")
      .write(f)

    assert(hc.readVDS(f, dropSamples = true)
      .same(hc.readVDS(f).dropSamples()))
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
    TestUtils.interceptFatal("""invalid metadata""") {
      hc.readVDS("src/test/resources/0.1-1fd5cc7.vds").count()
    }
  }

  @Test def testAnnotateVariantsKeyTable() {
    forAll(MatrixTable.gen(hc, VSMSubgen.random)) { vds =>
      val vds2 = vds.annotateVariantsExpr("bar = va")
      val kt = vds2.rowsTable()
      val resultVds = vds2.annotateVariantsTable(kt, "foo")
        .annotateVariantsExpr("foo = va.foo.bar")
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
    vds.queryGenotypes("gs.map(g => g.GQ).hist(0, 100, 100)")
  }

  @Test def testReorderSamples() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
    val origOrder = Array[Annotation]("C1046::HG02024", "C1046::HG02025", "C1046::HG02026", "C1047::HG00731", "C1047::HG00732")
    val newOrder = Array[Annotation]("C1046::HG02026", "C1046::HG02024", "C1047::HG00732", "C1046::HG02025", "C1047::HG00731")

    val filteredVds = vds.filterSamplesList(origOrder.map(Annotation(_)).toSet)
      .indexCols("colIdx")
      .annotateVariantsExpr("origGenos = gs.take(5)")
    val reorderedVds = filteredVds.reorderSamples(newOrder.map(Annotation(_)))
      .annotateVariantsExpr("newGenos = gs.takeBy(g => sa.colIdx, 5)")

    assert(reorderedVds.rowsTable().forall("origGenos == newGenos"))
    assert(vds.reorderSamples(vds.colKeys).same(vds))
  }
  
  @Test def testWriteBlockMatrix() {
    val dirname = tmpDir.createTempFile()
    
    for {
      numSlices <- Seq(1, 2, 4, 9, 11)
      blockSize <- Seq(1, 2, 3, 4, 6, 7, 9, 10)
    } {
      val vsm = hc.baldingNicholsModel(1, 6, 9, Some(numSlices), seed = blockSize + numSlices)
        .indexRows("rowIdx")
        .indexCols("colIdx")
      vsm.writeBlockMatrix(dirname, "g.GT.gt + va.rowIdx + sa.colIdx + 1", blockSize)

      vsm.rowsTable().select("locus", "rowIdx").show(1000)
      vsm.colsTable().select("s", "colIdx").show(1000)
      val data = vsm.entriesTable()
          .select("x = GT.gt + rowIdx + 1 + colIdx.toFloat64()")
          .collect()
          .map(_.getAs[Double](0))


      val lm = new DenseMatrix[Double](6, 9, data.asInstanceOf[IndexedSeq[Double]].toArray).t // data is row major
      
      assert(BlockMatrix.read(hc, dirname).toLocalMatrix() === lm)
    }
  }
  
  @Test def testWriteKeyedBlockMatrix() {
    val dirname = tmpDir.createTempFile()
    val nSamples = 6
    val nVariants = 9
    val vsm = hc.baldingNicholsModel(1, nSamples, nVariants, Some(4))
      .indexRows("rowIdx")
      .indexCols("colIdx")

    val data = vsm
      .entriesTable()
      .select("x = GT.gt + rowIdx + 1 + colIdx.toFloat64()")
      .collect()
      .map(_.getAs[Double](0))
    val lm = new DenseMatrix[Double](nSamples, nVariants, data.asInstanceOf[IndexedSeq[Double]].toArray).t // data is row major
    val rowKeys = new Keys(TStruct("locus" -> TLocus(GenomeReference.defaultReference), "alleles" -> TArray(TString())),
      Array.tabulate(nVariants)(i => Row(Locus("1", i + 1), IndexedSeq("A", "C"))))
    val colKeys = new Keys(TStruct("s" -> TString()), Array.tabulate(nSamples)(s => Annotation(s.toString)))
    
    vsm.writeKeyedBlockMatrix(dirname, "g.GT.gt + va.rowIdx + sa.colIdx + 1", blockSize = 3)
    
    val kbm = KeyedBlockMatrix.read(hc, dirname)
    
    assert(kbm.bm.toLocalMatrix() === lm)
    kbm.rowKeys.get.assertSame(rowKeys)
    kbm.colKeys.get.assertSame(colKeys)    
  }  
}
