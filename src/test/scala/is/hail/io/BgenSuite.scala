package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.io.bgen._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

class BgenSuite extends SparkSuite {

  def getNumberOfLinesInFile(file: String): Long = {
    hadoopConf.readFile(file) { s =>
      Source.fromInputStream(s)
        .getLines()
        .length
    }.toLong
  }

  def resizeWeights(input: Array[Int], newSize: Long): ArrayUInt = {
    class Output(ab: ArrayBuilder[Int]) extends IntConsumer {
      def +=(x: Int): Unit = ab += x
      def result(): Array[Int] = ab.result()
    }

    val n = input.length
    val resized = input.map(_.toDouble / input.sum * newSize)
    val fractional = new Array[Double](n)
    val index = new Array[Int](n)
    val indexInverse = new Array[Int](n)
    val output = new Output(new ArrayBuilder[Int])

    val F = BgenWriter.computeFractional(resized, fractional)
    BgenWriter.roundWithConstantSum(resized, fractional, index, indexInverse, output, F, newSize)
    new ArrayUInt(output.result())
  }

  def verifyGPs(vds: VariantKeyDataset) {
    assert(vds.rdd.forall { case (v, (va, gs)) =>
      gs.forall { g =>
        val gp =
          if (g != null)
            g.asInstanceOf[Row].getAs[IndexedSeq[Double]](1)
          else
            null
        gp == null || (gp.forall(d => d >= 0.0 && d <= 1.0)
          && D_==(gp.sum, 1.0))
      }
    })
  }

  def isGPSame(ds1: VariantKeyDataset, ds2: VariantKeyDataset, tolerance: Double): Boolean = {
    val ds1Expanded = ds1.expandWithAll().map { case (v, va, s, sa, g) => ((v.toString, s), g) }
    val ds2Expanded = ds2.expandWithAll().map { case (v, va, s, sa, g) => ((v.toString, s), g) }
    isGPSame(ds1Expanded, ds2Expanded, tolerance)
  }

  def isGPSame(ds1: RDD[((String, Annotation), Annotation)], ds2: RDD[((String, Annotation), Annotation)], tolerance: Double): Boolean = {
    ds1.fullOuterJoin(ds2).forall { case ((v, s), (g1, g2)) =>
      g1 == g2 || {
        val r1 = g1.get.asInstanceOf[Row]
        val r2 = g2.get.asInstanceOf[Row]

        val gp1 = if (r1 != null) r1.getAs[IndexedSeq[Double]](1) else null
        val gp2 = if (r2 != null) r2.getAs[IndexedSeq[Double]](1) else null

        gp1 == gp2 || gp1.zip(gp2).forall { case (d1, d2) => math.abs(d1 - d2) <= tolerance }
      }
    }
  }

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"
    val inputs = Array(
      ("src/test/resources/example.v11.bgen", 1d / 32768),
      ("src/test/resources/example.10bits.bgen", 1d / ((1L << 10) - 1)))

    val nSamples = getNumberOfLinesInFile(sampleFile) - 2
    val nVariants = getNumberOfLinesInFile(gen)

    inputs.foreach { case (bgen, tolerance) =>
      hadoopConf.delete(bgen + ".idx", recursive = true)

      val inputs = Array(
        ("src/test/resources/example.v11.bgen", 1d / ((1L << 10) - 1))
      )

      hc.indexBgen(bgen)
      val bgenVDS = hc.importBgen(bgen, sampleFile = Some(sampleFile), nPartitions = Some(10))
        .toVKDS
        .verifyBiallelic()
      assert(bgenVDS.nSamples == nSamples && bgenVDS.countVariants() == nVariants)

      val genVDS = hc.importGen(gen, sampleFile)
        .toVKDS

      val varidBgenQuery = bgenVDS.vaSignature.query("varid")
      val varidGenQuery = genVDS.vaSignature.query("varid")

      assert(bgenVDS.metadata == genVDS.metadata)
      assert(bgenVDS.sampleIds == genVDS.sampleIds)

      val bgenAnnotations = bgenVDS.variantsAndAnnotations.map { case (v, va) => (varidBgenQuery(va), va) }
      val genAnnotations = genVDS.variantsAndAnnotations.map { case (v, va) => (varidGenQuery(va), va) }

      assert(genAnnotations.fullOuterJoin(bgenAnnotations).forall { case (varid, (va1, va2)) => va1 == va2 })

      val isSame = isGPSame(genVDS, bgenVDS, tolerance)

      val vdsFile = tmpDir.createTempFile("bgenImportWriteRead", "vds")
      bgenVDS.write(vdsFile)
      val bgenWriteVDS = hc.readGDS(vdsFile)
        .toVKDS
      assert(bgenVDS
        .filterVariantsExpr("va.rsid != \"RSID_100\"")
        .same(bgenWriteVDS.filterVariantsExpr("va.rsid != \"RSID_100\"")
        ), "Not same after write/read.")

      hadoopConf.delete(bgen + ".idx", recursive = true)

      assert(isSame)
    }
  }

  object Spec extends Properties("BGEN Import/Export") {
    val compGen = for (vds <- VariantSampleMatrix.gen(hc,
      VSMSubgen.dosage.copy(
        vGen = _ => VariantSubgen.biallelic.gen.map(v => v.copy(contig = "01")),
        sGen = _ => Gen.identifier.filter(_ != "NA")))
      .filter(_.countVariants > 0)
      .map(_.copy(wasSplit = true));
      nPartitions <- choose(1, 10))
      yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    hadoopConf.writeTextFile(sampleRenameFile)(_.write("NA\tfdsdakfasdkfla"))

    property("bgen v1.1 import") =
      forAll(compGen) { case (vds, nPartitions) =>

        verifyGPs(vds)

        val fileRoot = tmpDir.createTempFile("testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"

        vds.exportGen(fileRoot, 5)

        val localRoot = tmpDir.createLocalTempFile("testImportBgen")
        val localGenFile = localRoot + ".gen"
        val localSampleFile = localRoot + ".sample"
        val localBgenFile = localRoot + ".bgen"
        val qcToolLogFile = localRoot + ".qctool.log"

        hadoopConf.copy(genFile, localGenFile)
        hadoopConf.copy(sampleFile, localSampleFile)

        val rc = s"qctool -force -g ${ uriPath(localGenFile) } -s ${ uriPath(localSampleFile) } -og ${ uriPath(localBgenFile) } -log ${ uriPath(qcToolLogFile) }" !

        hadoopConf.copy(localBgenFile, bgenFile)

        assert(rc == 0)

        hc.indexBgen(bgenFile)
        val importedVds = hc.importBgen(bgenFile, sampleFile = Some(sampleFile), nPartitions = Some(nPartitions))
          .toVKDS

        assert(importedVds.nSamples == vds.nSamples)
        assert(importedVds.countVariants() == vds.countVariants())
        assert(importedVds.sampleIds == vds.sampleIds)

        val varidQuery = importedVds.vaSignature.query("varid")
        val rsidQuery = importedVds.vaSignature.query("rsid")

        assert(importedVds.variantsAndAnnotations.forall { case (v, va) => varidQuery(va) == v.toString && rsidQuery(va) == "." })

        val isSame = isGPSame(vds, importedVds, 3e-4)

        hadoopConf.delete(bgenFile + ".idx", recursive = true)
        isSame
      }

    val compGen2 = for (vds <- VariantSampleMatrix.gen(hc,
      VSMSubgen.dosage.copy(
        vGen = _ => VariantSubgen.random.gen.map(v => v.copy(contig = "01")),
        sGen = _ => Gen.identifier.filter(_ != "NA")))
      .filter(_.countVariants > 0)
      .map(_.copy(wasSplit = false));
      nPartitions <- choose(1, 10);
      nBitsPerProb <- choose(1, 32))
      yield (vds, nPartitions, nBitsPerProb)

    property("bgen v1.2 export/import") =
      forAll(compGen2) { case (vds, nPartitions, nBitsPerProb) =>
        verifyGPs(vds)

        val fileRoot = tmpDir.createTempFile("testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val bgenFile = fileRoot + ".bgen"

        def test(parallel: Boolean): Boolean = {
          vds.exportBGEN(fileRoot, nBitsPerProb)
          hc.indexBgen(bgenFile)

          val importedVds = hc.importBgen(bgenFile, sampleFile = Some(sampleFile), nPartitions = Some(nPartitions))
              .toVKDS
              .cache()

          assert(importedVds.nSamples == vds.nSamples)
          assert(importedVds.countVariants() == vds.countVariants())
          assert(importedVds.sampleIds == vds.sampleIds)

          val varidQuery = importedVds.vaSignature.query("varid")
          val rsidQuery = importedVds.vaSignature.query("rsid")

          assert(importedVds.variantsAndAnnotations.forall { case (v, va) => varidQuery(va) == v.toString && rsidQuery(va) == "." })

          isGPSame(vds, importedVds, 1d / ((1L << nBitsPerProb) - 1))
        }

        test(parallel = true) && test(parallel = false)
      }

    val probIteratorGen = for (
      nBitsPerProb <- Gen.choose(1, 32);
      nSamples <- Gen.choose(1, 5);
      nGenotypes <- Gen.choose(3, 5);
      totalProb = ((1L << nBitsPerProb) - 1).toUInt;
      sampleProbs <- Gen.buildableOfN[Array, Array[UInt]](nSamples, Gen.partition(nGenotypes - 1, totalProb));
      input = sampleProbs.flatten
    ) yield (nBitsPerProb, nSamples, nGenotypes, input)

    property("bgen probability iterator, array give correct result") =
      forAll(probIteratorGen) { case (nBitsPerProb, nSamples, nGenotypes, input) =>
        assert(input.length == nSamples * (nGenotypes - 1))

        val packedInputBuilder = new ArrayBuilder[Byte]
        packedInputBuilder ++= new Array[Byte](nSamples + 10)

        val bitPacker = new BitPacker(packedInputBuilder, nBitsPerProb)
        input.foreach(bitPacker += _.intRep)
        bitPacker.flush()
        val packedInput = packedInputBuilder.result()
        val probArr = new Bgen12ProbabilityArray(packedInput, nSamples, nGenotypes, nBitsPerProb)

        for (s <- 0 until nSamples)
          for (i <- 0 until (nGenotypes - 1)) {
            val expected = input(s * (nGenotypes - 1) + i)
            assert(probArr(s, i) == expected)
          }

        true
      }

    val sortIndexGen = for (
      nGenotypes <- Gen.frequency((0.5, Gen.const(3)), (0.5, Gen.choose(0, 25)));
      input <- Gen.buildableOfN[Array, Double](nGenotypes, Gen.choose(-10, 10))
    ) yield (input, nGenotypes)

    property("sorted index") =
      forAll(sortIndexGen) { case (a, n) =>
        val index = new Array[Int](n)
        val sorted = new Array[Double](n)

        BgenWriter.sortedIndex(a, index)
        index.zipWithIndex.foreach { case (sortIdx, i) => sorted(i) = a(sortIdx) }

        sorted sameElements a.sortWith { case (d1, d2) => d1 > d2 }
      }
  }

  @Test def test() {
    Spec.check()
  }

  @Test def testResizeWeights() {
    assert(resizeWeights(Array(0, 32768, 0), (1L << 32) - 1).intArrayRep sameElements Array(0, UInt(4294967295L).intRep, 0))
    assert(resizeWeights(Array(1, 1, 1), 32768).intArrayRep.toSet sameElements Set(10923, 10923, 10922))
    assert(resizeWeights(Array(2, 3, 1), 32768).intArrayRep sameElements Array(10923, 16384, 5461))
  }

  @Test def testParallelImport() {
    val bgen = "src/test/resources/parallelBgenExport.bgen"
    val sample = "src/test/resources/parallelBgenExport.sample"

    hc.indexBgen(bgen)
    hc.importBgen(bgen, Option(sample)).count()
  }

  @Test def testReIterate() {
    hc.indexBgen("src/test/resources/example.v11.bgen")
    val vds = hc.importBgen("src/test/resources/example.v11.bgen", Some("src/test/resources/example.sample"))

    assert(vds.annotateVariantsExpr("va.cr1 = gs.fraction(g => g.GT.isCalled())")
      .annotateVariantsExpr("va.cr2 = gs.fraction(g => g.GT.isCalled())")
      .variantsKT()
      .forall("va.cr1 == va.cr2"))
  }
}