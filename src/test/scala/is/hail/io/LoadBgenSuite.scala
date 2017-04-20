package is.hail.io

import is.hail.SparkSuite
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.io.bgen.BgenProbabilityIterator
import is.hail.utils._
import is.hail.variant._
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

class LoadBgenSuite extends SparkSuite {

  def getNumberOfLinesInFile(file: String): Long = {
    hadoopConf.readFile(file) { s =>
      Source.fromInputStream(s)
        .getLines()
        .length
    }.toLong
  }

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"
    val inputs = Array(
      ("src/test/resources/example.v11.bgen", 1e-6),
      ("src/test/resources/example.10bits.bgen", 1d / (math.pow(2, 10) - 1))
    )

    def testBgen(bgen: String, tolerance: Double = 1e-6): Unit = {
      hadoopConf.delete(bgen + ".idx", recursive = true)

      val nSamples = getNumberOfLinesInFile(sampleFile) - 2
      val nVariants = getNumberOfLinesInFile(gen)

      hc.indexBgen(bgen)
      val bgenVDS = hc.importBgen(bgen, sampleFile = Some(sampleFile), nPartitions = Some(10))
      assert(bgenVDS.nSamples == nSamples && bgenVDS.countVariants() == nVariants)

      val genVDS = hc.importGen(gen, sampleFile)

      val varidBgenQuery = bgenVDS.vaSignature.query("varid")
      val rsidBgenQuery = bgenVDS.vaSignature.query("rsid")

      val varidGenQuery = genVDS.vaSignature.query("varid")
      val rsidGenQuery = bgenVDS.vaSignature.query("rsid")

      assert(bgenVDS.metadata == genVDS.metadata)
      assert(bgenVDS.sampleIds == genVDS.sampleIds)

      val bgenAnnotations = bgenVDS.variantsAndAnnotations.map { case (v, va) => (varidBgenQuery(va), va) }
      val genAnnotations = genVDS.variantsAndAnnotations.map { case (v, va) => (varidGenQuery(va), va) }

      assert(genAnnotations.fullOuterJoin(bgenAnnotations).forall { case (varid, (va1, va2)) => va1 == va2 })

      val bgenFull = bgenVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((varidBgenQuery(va), s), gt) }
      val genFull = genVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((varidGenQuery(va), s), gt) }

      val isSame = genFull.fullOuterJoin(bgenFull)
        .collect()
        .forall { case ((v, i), (gt1, gt2)) =>
          (gt1, gt2) match {
            case (Some(x), Some(y)) =>
              (x.dosage, y.dosage) match {
                case (Some(dos1), Some(dos2)) => dos1.zip(dos2).forall { case (d1, d2) => math.abs(d1 - d2) <= tolerance }
                case (None, None) => true
                case _ => false
              }
            case _ => false
          }
        }

      hadoopConf.delete(bgen + ".idx", recursive = true)

      assert(isSame)
    }

    inputs.foreach { case (file, tolerance) => testBgen(file, tolerance) }
  }

  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds <- VariantSampleMatrix.gen(hc,
      VSMSubgen.dosage.copy(vGen = VariantSubgen.biallelic.gen.map(v => v.copy(contig = "01")),
        sampleIdGen = Gen.distinctBuildableOf[IndexedSeq, String](Gen.identifier.filter(_ != "NA"))))
      .filter(_.countVariants > 0)
      .map(_.copy(wasSplit = true));
      nPartitions <- choose(1, 10))
      yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    hadoopConf.writeTextFile(sampleRenameFile)(_.write("NA\tfdsdakfasdkfla"))

    property("import generates same output as export") =
      forAll(compGen) { case (vds, nPartitions) =>

        assert(vds.rdd.forall { case (v, (va, gs)) =>
          gs.flatMap(_.dosage).flatten.forall(d => d >= 0.0 && d <= 1.0)
        })

        val fileRoot = tmpDir.createTempFile("testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"

        vds.exportGen(fileRoot)

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
          .cache()

        assert(importedVds.nSamples == vds.nSamples)
        assert(importedVds.countVariants() == vds.countVariants())
        assert(importedVds.sampleIds == vds.sampleIds)

        val importedFull = importedVds.expandWithAll().map { case (v, va, s, sa, g) => ((v, s), g) }
        val originalFull = vds.expandWithAll().map { case (v, va, s, sa, g) => ((v, s), g) }

        originalFull.fullOuterJoin(importedFull).forall { case ((v, i), (g1, g2)) =>

          val r = g1 == g2 ||
            g1.get.dosage.get.zip(g2.get.dosage.get)
              .forall { case (d1, d2) => math.abs(d1 - d2) < 1e-4 }

          if (!r)
            println(g1, g2)

          r
        }
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check()
  }

  object TestProbIterator extends Properties("ImportBGEN") {
    val bitPackedIteratorGen = for (
      v <- Gen.buildableOf[Array, Double](Gen.choose(0d, 1d)).resize(10);
      nBitsPerProb <- Gen.choose(1, 32)) yield (v, nBitsPerProb)

    property("bgen probability iterator gives correct result") =
      forAll(Gen.buildableOf[Array, Double](Gen.choose(0d, 1d)).resize(10)) { v =>
        val probs = v.map(_ / v.sum)

        (1 to 32).forall { nBitsPerProb =>

          val probInts = probs.map(_ * (math.pow(2, nBitsPerProb) - 1))
          val floorInts = probInts.map(i => (i, math.floor(i)))
          val totalFractional = floorInts.map(f => f._1 - f._2).sum.toInt
          val input = floorInts.sortBy(i => -(i._1 - i._2)).zipWithIndex.map { case (r, i) =>
            if (i < totalFractional)
              (i, math.ceil(r._1))
            else
              (i, math.floor(r._1))
          }.map(_._2.toInt)

          val expectedNBytes = math.ceil((probs.length * nBitsPerProb).toDouble / 8).toInt
          val packedInput = Array.ofDim[Byte](expectedNBytes)

          val bitMask = ~0L >>> (64 - nBitsPerProb)
          assert(bitMask == (math.pow(2, nBitsPerProb) - 1).toLong)

          var byteIndex = 0
          var data = 0L
          var dataSize = 0

          input.foreach { i =>
            data |= ((i.toLong & bitMask) << dataSize)
            dataSize += nBitsPerProb

            while (dataSize >= 8) {
              packedInput(byteIndex) = (data & 0xffL).toByte
              data = data >>> 8
              dataSize -= 8
              byteIndex += 1
            }
          }

          if (dataSize > 0)
            packedInput(byteIndex) = (data & 0xffL).toByte

          val probIterator = new BgenProbabilityIterator(packedInput, nBitsPerProb)
          val result = Array.ofDim[Int](probs.length)

          for (i <- probs.indices) {
            if (probIterator.hasNext) {
              result(i) = probIterator.next()
            } else {
              fatal(s"Should have at least ${ probs.length } probabilities. Found i=$i")
            }
          }

          result sameElements input
        }
      }
  }

  @Test def testBgenProbabilityIterator() {
    TestProbIterator.check()
  }

}