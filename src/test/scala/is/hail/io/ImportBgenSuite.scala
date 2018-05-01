package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.io.bgen.BGen12ProbabilityArray
import is.hail.io.gen.ExportGen
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

class BgenProbabilityIterator(input: ByteArrayReader, nBitsPerProb: Int) extends HailIterator[UInt] {
  val bitMask = (1L << nBitsPerProb) - 1
  var data = 0L
  var dataSize = 0

  override def next(): UInt = {
    while (dataSize < nBitsPerProb && input.hasNext()) {
      data |= ((input.read() & 0xffL) << dataSize)
      dataSize += 8
    }
    assert(dataSize >= nBitsPerProb, s"Data size `$dataSize' less than nBitsPerProb `$nBitsPerProb'.")

    val result = data & bitMask
    dataSize -= nBitsPerProb
    data = data >>> nBitsPerProb
    result.toUInt
  }

  override def hasNext: Boolean = input.hasNext() || (dataSize >= nBitsPerProb)
}

class ImportBgenSuite extends SparkSuite {
  private val contigRecoding = Some((1 to 9).map(i => s"0$i" -> i.toString).toMap)

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

    val genMT = hc.importGen(gen, sampleFile, contigRecoding = contigRecoding)

    val inputs = Array(
      ("src/test/resources/example.v11.bgen", 1d / 32768),
      ("src/test/resources/example.10bits.bgen", 1d / ((1L << 10) - 1)),
      ("src/test/resources/example.8bits.bgen", 1d / 255))

    assert(inputs.forall { case (bgenFile, tolerance) =>
      hadoopConf.delete(bgenFile + ".idx", recursive = true)
      hc.indexBgen(bgenFile)
      val bgenMT = hc.importBgen(bgenFile, sampleFile = Some(sampleFile), includeGT = true, includeGP = true,
        includeDosage = false, nPartitions = Some(10), contigRecoding = contigRecoding)
      val isSame = genMT.same(bgenMT, tolerance, absolute = true)
      hadoopConf.delete(bgenFile + ".idx", recursive = true)
      isSame
    })
  }

  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds <- MatrixTable.gen(hc,
      VSMSubgen.callAndProbabilities.copy(
        // qctool recodes other GRCh37 contigs as "NA"
        vGen = _ => VariantSubgen.plinkCompatibleBiallelic(ReferenceGenome.defaultReference).genLocusAlleles,
        sGen = _ => Gen.identifier.filter(_ != "NA")))
      .filter(_.countRows > 0);
      nPartitions <- choose(1, 10))
      yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    hadoopConf.writeTextFile(sampleRenameFile)(_.write("NA\tfdsdakfasdkfla"))

    property("import generates same output as export") =
      forAll(compGen) { case (vds, nPartitions) =>

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

        val fileRoot = tmpDir.createTempFile("testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"

        TestUtils.exportGen(vds, fileRoot, 5)

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
        val importedVds = hc.importBgen(bgenFile, sampleFile = Some(sampleFile),
          includeGT = true, includeGP = true, includeDosage = false, nPartitions = Some(nPartitions),
          contigRecoding = contigRecoding)

        assert(importedVds.numCols == vds.numCols)
        assert(importedVds.countRows() == vds.countRows())
        assert(importedVds.stringSampleIds == vds.stringSampleIds)

        val varidQuery = importedVds.rowType.query("varid")
        val rsidQuery = importedVds.rowType.query("rsid")

        assert(importedVds.variantsAndAnnotations.forall { case (v, va) => varidQuery(va) == v.toString && rsidQuery(va) == "." })

        val importedFull = importedVds.expandWithAll().map { case (v, va, s, sa, g) => ((v, s), g) }
        val originalFull = vds.expandWithAll().map { case (v, va, s, sa, g) => ((v, s), g) }

        originalFull.fullOuterJoin(importedFull).forall { case ((v, i), (g1, g2)) =>
          g1 == g2 || {
            val r1 = g1.get.asInstanceOf[Row]
            val r2 = g2.get.asInstanceOf[Row]
            val gp1 = if (r1 != null) r1.getAs[IndexedSeq[Double]](1) else null
            val gp2 = if (r2 != null) r2.getAs[IndexedSeq[Double]](1) else null
            gp1 == gp2 || gp1.zip(gp2).forall { case (d1, d2) => math.abs(d1 - d2) < 1e-4 }
          }
        }
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check()
  }

  def bitPack(input: Array[UInt], nSamples: Int, nBitsPerProb: Int): Array[Byte] = {
    val expectedProbBytes = (input.length * nBitsPerProb + 7) / 8
    val packedInput = new Array[Byte](nSamples + 10 + expectedProbBytes)

    val bitMask = (1L << nBitsPerProb) - 1
    var byteIndex = 0
    var data = 0L
    var dataSize = 0

    input.foreach { i =>
      data |= ((i.toLong & bitMask) << dataSize)
      dataSize += nBitsPerProb

      while (dataSize >= 8) {
        packedInput(nSamples + 10 + byteIndex) = (data & 0xffL).toByte
        data = data >>> 8
        dataSize -= 8
        byteIndex += 1
      }
    }

    if (dataSize > 0)
      packedInput(nSamples + 10 + byteIndex) = (data & 0xffL).toByte

    packedInput
  }

  object TestProbIterator extends Properties("ImportBGEN") {
    val probIteratorGen = for (
      nBitsPerProb <- Gen.choose(1, 32);
      nSamples <- Gen.choose(1, 5);
      nGenotypes <- Gen.choose(3, 5);
      nProbabilities = nSamples * (nGenotypes - 1);
      totalProb = ((1L << nBitsPerProb) - 1).toUInt;
      sampleProbs <- Gen.buildableOfN[Array](nSamples, Gen.partition(nGenotypes - 1, totalProb));
      input = sampleProbs.flatten
    ) yield (nBitsPerProb, nSamples, nGenotypes, input)

    property("bgen probability iterator, array give correct result") =
      forAll(probIteratorGen) { case (nBitsPerProb, nSamples, nGenotypes, input) =>
        assert(input.length == nSamples * (nGenotypes - 1))
        val packedInput = bitPack(input, nSamples, nBitsPerProb)
        val reader = new ByteArrayReader(packedInput)
        reader.seek(nSamples + 10)
        val probIter = new BgenProbabilityIterator(reader, nBitsPerProb)
        val probArr = new BGen12ProbabilityArray(packedInput, nSamples, nGenotypes, nBitsPerProb)

        for (s <- 0 until nSamples)
          for (i <- 0 until (nGenotypes - 1)) {
            val expected = input(s * (nGenotypes - 1) + i)
            assert(probIter.hasNext)
            assert(probIter.next() == expected)
            assert(probArr(s, i) == expected)
          }
        assert(!probIter.hasNext)

        true
      }
  }

  @Test def testBgenProbabilityIterator() {
    TestProbIterator.check()
  }

  @Test def testParallelImport() {
    val bgen = "src/test/resources/parallelBgenExport.bgen"
    val sample = "src/test/resources/parallelBgenExport.sample"

    hc.indexBgen(bgen)
    hc.importBgen(bgen, Option(sample), includeGT = true, includeGP = true, includeDosage = false).count()
  }

  @Test def testReIterate() {
    hc.indexBgen("src/test/resources/example.v11.bgen")
    val vds = hc.importBgen("src/test/resources/example.v11.bgen", Some("src/test/resources/example.sample"),
      includeGT = true, includeGP = true, includeDosage = false, contigRecoding = contigRecoding)

    assert(vds.annotateRowsExpr(("cr1", "AGG.fraction(g => isDefined(g.GT))"))
      .annotateRowsExpr(("cr2", "AGG.fraction(g => isDefined(g.GT))"))
      .rowsTable()
      .forall("row.cr1 == row.cr2"))
  }
  
  @Test def testDosage() {
    for (bgen <- Array("src/test/resources/example.8bits.bgen",
                       "src/test/resources/example.10bits.bgen",
                       "src/test/resources/example.v11.bgen")) {
      if (!hadoopConf.exists(bgen + ".idx"))
        hc.indexBgen(bgen)
      
      val vds = hc.importBgen(bgen, includeGT = false, includeGP = true, includeDosage = true, contigRecoding = contigRecoding)
        .filterRowsExpr("va.locus.position == 2000")

      val dosages = vds.aggregateEntries("AGG.map(g => g.dosage).collect()")._1
        .asInstanceOf[IndexedSeq[java.lang.Double]]

      val dosagesFromGP = vds.aggregateEntries("AGG.map(g => dosage(g.GP)).collect()")._1
        .asInstanceOf[IndexedSeq[java.lang.Double]]
      
      assert(dosages.length == 500 && dosagesFromGP.length == 500)
      assert(dosages.count(_ == null) > 0)
      assert((dosages, dosagesFromGP).zipped.forall { case (x, y) => (x == null && y == null) || D_==(x, y) })
    }
  }
}
