package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.io.bgen.{BgenProbabilityIterator, BgenWriter, BitPacker, IntConsumer}
import is.hail.utils._
import is.hail.variant._
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
    val n = input.length
    val resized = new Array[Double](n)
    val fractional = new Array[Double](n)
    val index = new Array[Int](n)
    val output = new ArrayBuilder[Int](n) with IntConsumer

    val conversionFactor = newSize.toDouble / input.sum
    val F = BgenWriter.resizeAndComputeFractional(input, resized, fractional, conversionFactor)
    BgenWriter.roundWithConstantSum(resized, fractional, index, output, F, newSize)
    new ArrayUInt(output.result())
  }

  def isGPSame(vds1: VariantDataset, vds2: VariantDataset, tolerance: Double): Boolean = {
    val vds1Expanded = vds1.expandWithAll().map { case (v, va, s, sa, g) => ((v.toString, s), g) }
    val vds2Expanded = vds2.expandWithAll().map { case (v, va, s, sa, g) => ((v.toString, s), g) }
    isGPSame(vds1Expanded, vds2Expanded, tolerance)
  }

  def isGPSame(vds1: RDD[((String, Annotation), Genotype)], vds2: RDD[((String, Annotation), Genotype)], tolerance: Double): Boolean = {
    vds1.fullOuterJoin(vds2).forall { case ((v, s), (x, y)) =>
      (x, y) match {
        case (Some(gt1), Some(gt2)) =>
          (gt1.gp, gt2.gp) match {
            case (Some(d1), Some(d2)) => d1.zip(d2).forall { case (a, b) =>
              val isSame = math.abs(a - b) <= tolerance
              if (!isSame)
                println(s"v=$v s=$s gt1=$gt1 gt2=$gt2")
              isSame
            }
            case (None, None) => true
            case _ => false
          }
        case _ => false
      }
    }
  }

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"

    val nSamples = getNumberOfLinesInFile(sampleFile) - 2
    val nVariants = getNumberOfLinesInFile(gen)

    val inputs = Array(
      ("src/test/resources/example.v11.bgen", 1e-6),
      ("src/test/resources/example.10bits.bgen", 1d / ((1L << 10) - 1))
    )

    inputs.foreach { case (bgen, tolerance) =>
      hadoopConf.delete(bgen + ".idx", recursive = true)

      hc.indexBgen(bgen)
      val bgenVDS = hc.importBgen(bgen, sampleFile = Some(sampleFile), nPartitions = Some(10))
      assert(bgenVDS.nSamples == nSamples && bgenVDS.countVariants() == nVariants)

      val genVDS = hc.importGen(gen, sampleFile)

      val varidBgenQuery = bgenVDS.vaSignature.query("varid")
      val varidGenQuery = genVDS.vaSignature.query("varid")

      assert(bgenVDS.metadata == genVDS.metadata)
      assert(bgenVDS.sampleIds == genVDS.sampleIds)

      val bgenAnnotations = bgenVDS.variantsAndAnnotations.map { case (v, va) => (varidBgenQuery(va), va) }
      val genAnnotations = genVDS.variantsAndAnnotations.map { case (v, va) => (varidGenQuery(va), va) }

      assert(genAnnotations.fullOuterJoin(bgenAnnotations).forall { case (varid, (va1, va2)) => va1 == va2 })

      val genExpanded = genVDS.expandWithAll().map { case (v, va, s, sa, g) => ((varidGenQuery(va).asInstanceOf[String], s), g) }
      val bgenExpanded = bgenVDS.expandWithAll().map { case (v, va, s, sa, g) => ((varidBgenQuery(va).asInstanceOf[String], s), g) }

      val isSame = isGPSame(genExpanded, bgenExpanded, tolerance)

      hadoopConf.delete(bgen + ".idx", recursive = true)

      assert(isSame)
    }
  }

  object TestBgen extends Properties("BGEN Import/Export") {
    val bitPackedIteratorGen = for {
      v <- Gen.buildableOf[Array, Double](Gen.choose(0d, 1d)).resize(10)
      nBitsPerProb <- Gen.choose(1, 32)
    } yield (v, nBitsPerProb)

    val compGen1 = for {
      vds <- VariantSampleMatrix.gen(hc,
        VSMSubgen.dosageGenotype.copy(vGen = VariantSubgen.biallelic.gen.map(v => v.copy(contig = "01")),
          sampleIdGen = Gen.distinctBuildableOf[Array, String](Gen.identifier.filter(_ != "NA"))))
        .filter(_.countVariants > 0)
        .map(_.copy(wasSplit = true))
      nPartitions <- choose(1, 10)
    } yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    hadoopConf.writeTextFile(sampleRenameFile)(_.write("NA\tfdsdakfasdkfla"))

    property("bgen v1.1 import") =
      forAll(compGen1) { case (vds, nPartitions) =>

        assert(vds.rdd.forall { case (v, (va, gs)) =>
          gs.flatMap(_.gp).flatten.forall(d => d >= 0.0 && d <= 1.0)
        })

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

        val isSame = isGPSame(vds, importedVds, 3e-4)

        hadoopConf.delete(bgenFile + ".idx", recursive = true)
        isSame
      }

    val compGen2 = for {vds <- VariantSampleMatrix.gen(hc,
      VSMSubgen.dosageGenotype.copy(vGen = VariantSubgen.biallelic.gen,
        sampleIdGen = Gen.distinctBuildableOf[Array, String](Gen.identifier.filter(_ != "NA"))))
      .map(vds => vds.splitMulti())
      nBitsPerProb <- choose(1, 32)
    } yield (vds, nBitsPerProb)

    property("bgen v1.2 export/import") =
      forAll(compGen2) { case (vds, nBitsPerProb) =>
        val fileRoot = tmpDir.createTempFile("testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val bgenFile = fileRoot + ".bgen"

        vds.exportBgen(fileRoot, nBitsPerProb)
        hc.indexBgen(bgenFile)
        val importedVds = hc.importBgen(bgenFile, sampleFile = Some(sampleFile), nPartitions = Some(10))
        val isSame = isGPSame(vds, importedVds, 1d / ((1L << nBitsPerProb) - 1))

        val fileRootParallel = tmpDir.createTempFile("testImportBgenParallel")
        vds.exportBgen(fileRootParallel, nBitsPerProb, parallel = true)
        hc.indexBgen(fileRootParallel + ".bgen/part-*")
        val bgenFilesParallel = hadoopConf.glob(fileRootParallel + ".bgen/part-*").map(_.getPath.toString).filterNot(_.endsWith(".idx"))
        val sampleFileParallel = fileRootParallel + ".sample"
        val parallelVds = hc.importBgens(bgenFilesParallel, Option(sampleFileParallel), nPartitions = Some(10))
        val isSameParallel = isGPSame(vds, parallelVds, 1d / ((1L << nBitsPerProb) - 1))

        isSame && isSameParallel
      }

    val probIteratorGen = for {
      nBitsPerProb <- Gen.choose(1, 32)
      nProbabilities <- Gen.choose(1, 20)
      size = ((1L << nBitsPerProb) - 1).toUInt
      input <- Gen.partition(nProbabilities, size)
    } yield (nBitsPerProb, input)

    property("bgen probability iterator gives correct result") =
      forAll(probIteratorGen) { case (nBitsPerProb, input) =>
        val packedInput = new ArrayBuilder[Byte]
        val bitPacker = new BitPacker(packedInput, nBitsPerProb)
        input.foreach(bitPacker += _.intRep)
        bitPacker.flush()

        val probIterator = new BgenProbabilityIterator(new ByteArrayReader(packedInput.result()), nBitsPerProb)
        val result = new Array[UInt](input.length)

        for (i <- input.indices) {
          if (probIterator.hasNext) {
            result(i) = probIterator.next()
          } else {
            fatal(s"Should have at least ${ input.length } probabilities. Found i=$i")
          }
        }

        input sameElements result
      }

    property("sorted index") =
      forAll(Gen.buildableOfN[Array, Double](3, Gen.choose(-10, 10))) { a =>
        val index = new Array[Int](3)
        val sorted = new Array[Double](3)
        BgenWriter.sortedIndex(a, index)
        index.zipWithIndex.foreach { case (sortedIdx, origIdx) => sorted(sortedIdx) = a(origIdx) }
        sorted sameElements a.sortWith { case (d1, d2) => d1 > d2 }
      }
  }

  @Test def test() {
    TestBgen.check()
  }

  @Test def testResizeWeights() {
    assert(resizeWeights(Array(0, 32768, 0), (1L << 32) - 1).intArrayRep sameElements Array(0, UInt(4294967295L).intRep, 0))
    assert(resizeWeights(Array(1, 1, 1), 32768).intArrayRep sameElements Array(10923, 10923, 10922))
    assert(resizeWeights(Array(2, 3, 1), 32768).intArrayRep sameElements Array(10923, 16384, 5461))
  }

  @Test def testParallelImport() {
    val bgen = "src/test/resources/parallelBgenExport.bgen"
    val sample = "src/test/resources/parallelBgenExport.sample"

    hc.indexBgen(bgen)
    hc.importBgen(bgen, Option(sample)).count()
  }
}