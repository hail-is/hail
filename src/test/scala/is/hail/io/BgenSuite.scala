package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.io.bgen.{BgenProbabilityIterator, BgenWriter}
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
    val convFactor = ((1L << 32) - 1) / 32768d
    assert(BgenWriter.resizeProbInts(Array(0, 32768, 0), convFactor) sameElements Array(UInt(0), UInt(4294967295L), UInt(0)))

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
      VSMSubgen.dosageGenotype.copy(vGen = VariantSubgen.random.gen,
        sampleIdGen = Gen.distinctBuildableOf[Array, String](Gen.identifier.filter(_ != "NA")))).filter(_.countVariants > 0)
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

        hadoopConf.delete(bgenFile + ".idx", recursive = true)

        isSame
      }

    val probIteratorGen = for {
      nBitsPerProb <- Gen.choose(1, 32)
      nProbabilities <- Gen.choose(1, 20)
      size = ((1L << nBitsPerProb) - 1).toUInt
      input <- Gen.partition(nProbabilities, size)
    } yield (nBitsPerProb, input)

    property("bgen probability iterator gives correct result") =
      forAll(probIteratorGen) { case (nBitsPerProb, input) =>
        val packedInput = BgenWriter.bitPack(input, nBitsPerProb)
        val probIterator = new BgenProbabilityIterator(new ByteArrayReader(packedInput), nBitsPerProb)
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
  }

  @Test def test() {
    TestBgen.check()
  }

  @Test def testParallelImport() {
    val bgen = "src/test/resources/parallelBgenExport.bgen"
    val sample = "src/test/resources/parallelBgenExport.sample"

    hc.indexBgen(bgen)
    hc.importBgen(bgen, Option(sample)).count()
  }
}