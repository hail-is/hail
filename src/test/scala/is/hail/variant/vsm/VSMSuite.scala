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

  @Test def testSame() {
    val vds1 = hc.importVCF("src/test/resources/sample.vcf.gz", force = true)
    val vds2 = hc.importVCF("src/test/resources/sample.vcf.gz", force = true)
    assert(vds1.same(vds2))

    val s1mdata = MatrixFileMetadata(Array("S1", "S2", "S3").map(Annotation(_)))
    val s1va1: Annotation = null
    val s1va2 = Annotation()

    val s2mdata = MatrixFileMetadata(Array("S1", "S2").map(Annotation(_)))
    val s2va1: Annotation = null
    val s2va2: Annotation = null

    val s3mdata = MatrixFileMetadata(
      Array("S1", "S2", "S3").map(Annotation(_)),
      vaSignature = TStruct(
        "inner" -> TStruct(
          "thing1" -> TString()),
        "thing2" -> TString()))
    val s3va1 = Annotation(Annotation("yes"), "yes")
    val s3va2 = Annotation(Annotation("yes"), "no")
    val s3va3 = Annotation(Annotation("no"), "yes")

    val s4mdata = MatrixFileMetadata(
      Array("S1", "S2").map(Annotation(_)),
      vaSignature = TStruct(
        "inner" -> TStruct(
          "thing1" -> TString()),
        "thing2" -> TString(),
        "dummy" -> TString()))
    val s4va1 = Annotation(Annotation("yes"), "yes", null)
    val s4va2 = Annotation(Annotation("yes"), "no", "dummy")

    assert(s1mdata != s2mdata)
    assert(s1mdata != s3mdata)
    assert(s2mdata != s3mdata)
    assert(s1mdata != s4mdata)
    assert(s2mdata != s4mdata)
    assert(s3mdata != s4mdata)

    val v1 = Variant("1", 1, "A", "T")
    val v2 = Variant("1", 2, "T", "G")
    val v3 = Variant("1", 2, "T", "A")

    val s3rdd1 = sc.parallelize(Seq((v1, (s3va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v2, (s3va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1))))))

    // differ in variant
    val s3rdd2 = sc.parallelize(Seq((v1, (s3va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v3, (s3va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1))))))

    // differ in genotype
    val s3rdd3 = sc.parallelize(Seq((v1, (s3va1,
      Iterable(Genotype(),
        Genotype(1),
        Genotype(2)))),
      (v2, (s3va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1))))))

    val s2rdd4 = sc.parallelize(Seq((v1, (s2va1,
      Iterable(Genotype(),
        Genotype(0)))),
      (v2, (s2va2, Iterable(
        Genotype(0),
        Genotype(0))))))

    // differ in number of variants
    val s3rdd5 = sc.parallelize(Seq((v1, (s3va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2))))))

    // differ in variant annotations
    val s3rdd6 = sc.parallelize(Seq((v1, (s3va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v2, (s3va3,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1))))))

    val s1rdd7 = sc.parallelize(Seq((v1, (s1va1,
      Iterable(Genotype(),
        Genotype(0),
        Genotype(2)))),
      (v2, (s1va2,
        Iterable(Genotype(0),
          Genotype(0),
          Genotype(1))))))

    val s2rdd8 = sc.parallelize(Seq((v1, (s2va1,
      Iterable(Genotype(),
        Genotype(0)))),
      (v2, (s2va2,
        Iterable(Genotype(0),
          Genotype(0))))))

    val s4rdd9 = sc.parallelize(Seq((v1, (s4va1,
      Iterable(Genotype(),
        Genotype(0)))),
      (v2, (s4va2,
        Iterable(Genotype(0),
          Genotype(0))))))

    val vdss = Array(MatrixTable.fromLegacy(hc, s3mdata, s3rdd1),
      MatrixTable.fromLegacy(hc, s3mdata, s3rdd2),
      MatrixTable.fromLegacy(hc, s3mdata, s3rdd3),
      MatrixTable.fromLegacy(hc, s2mdata, s2rdd4),
      MatrixTable.fromLegacy(hc, s3mdata, s3rdd5),
      MatrixTable.fromLegacy(hc, s3mdata, s3rdd6),
      MatrixTable.fromLegacy(hc, s1mdata, s1rdd7),
      MatrixTable.fromLegacy(hc, s2mdata, s2rdd8),
      MatrixTable.fromLegacy(hc, s4mdata, s4rdd9))

    for (vds <- vdss)
      vds.typecheck()

    for (i <- vdss.indices;
      j <- vdss.indices) {
      if (i == j)
        assert(vdss(i) == vdss(j))
      else
        assert(vdss(i) != vdss(j))
    }
  }

  @Test def testWriteRead() {
    val p = forAll(MatrixTable.gen(hc, VSMSubgen.random)) { vds =>
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
      val vds2 = vds.annotateVariantsExpr("va.bar = va")
      val kt = vds2.variantsKT()
      val resultVds = vds2.annotateVariantsTable(kt, expr = "va.foo = table.va.bar")
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
    val reorderedVds = filteredVds.reorderSamples(newOrder.map(Annotation(_)))

    def getGenotypes(vds: MatrixTable): RDD[((Variant, Annotation), Annotation)] = {
      val sampleIds = vds.stringSampleIds
      vds.typedRDD[Variant].flatMap { case (v, (_, gs)) =>
        gs.zip(sampleIds).map { case (g, s) =>
          ((v, s), g)
        }
      }
    }

    assert(getGenotypes(filteredVds).fullOuterJoin(getGenotypes(reorderedVds)).forall { case ((v, s), (g1, g2)) =>
      g1 == g2
    })

    assert(reorderedVds.stringSampleIds sameElements newOrder)

    assert(vds.reorderSamples(vds.colKeys.toArray).same(vds))

    intercept[HailException](vds.reorderSamples(newOrder))
    intercept[HailException](vds.reorderSamples(vds.colKeys.toArray ++ Array[Annotation]("foo", "bar").map(Annotation(_))))
  }
  
  @Test def testWriteBlockMatrix() {
    val dirname = tmpDir.createTempFile()
    
    for {
      numSlices <- Seq(1, 2, 4, 9, 11)
      blockSize <- Seq(1, 2, 3, 4, 6, 7, 9, 10)
    } {
      val vsm = hc.baldingNicholsModel(1, 6, 9, Some(numSlices), seed = blockSize + numSlices)      
      vsm.writeBlockMatrix(dirname, "g.GT.gt + v.start + sa.s.toInt32()", blockSize)

      val data = vsm.collect().zipWithIndex.flatMap { 
        case (row, v) => row.getAs[IndexedSeq[Row]](3).zipWithIndex.map { 
          case (gt, s) => 
            (gt.getInt(0) + (v + 1) + s).toDouble
        }
      }
      
      val lm = new DenseMatrix[Double](6, 9, data).t // data is row major
      
      assert(BlockMatrix.read(hc, dirname).toLocalMatrix() === lm)
    }
  }
  
  @Test def testWriteKeyedBlockMatrix() {
    val dirname = tmpDir.createTempFile()
    val nSamples = 6
    val nVariants = 9
    val vsm = hc.baldingNicholsModel(1, nSamples, nVariants, Some(4))      
    
    val data = vsm.collect().zipWithIndex.flatMap { 
      case (row, v) => row.getAs[IndexedSeq[Row]](3).zipWithIndex.map {
        case (gt, s) => (gt.getInt(0) + (v + 1) + s).toDouble
      }
    }
    val lm = new DenseMatrix[Double](nSamples, nVariants, data).t // data is row major
    val rowKeys = new Keys(TVariant(GenomeReference.defaultReference), Array.tabulate(nVariants)(i => Variant("1", i + 1, "A", "C")))
    val colKeys = new Keys(TStruct("s" -> TString()), Array.tabulate(nSamples)(s => Annotation(s.toString)))
    
    vsm.writeKeyedBlockMatrix(dirname, "g.GT.gt + v.start + sa.s.toInt32()", blockSize = 3)
    
    val kbm = KeyedBlockMatrix.read(hc, dirname)
    
    assert(kbm.bm.toLocalMatrix() === lm)
    kbm.rowKeys.get.assertSame(rowKeys)
    kbm.colKeys.get.assertSame(colKeys)    
  }  
}
