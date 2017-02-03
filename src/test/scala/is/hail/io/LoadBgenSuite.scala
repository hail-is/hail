package is.hail.io

import is.hail.utils._
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.driver._
import is.hail.variant._
import is.hail.SparkSuite
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
    val bgen = "src/test/resources/example.v11.bgen"

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

    val bgenAnnotations = bgenVDS.variantsAndAnnotations.map { case (v, va) => (varidBgenQuery(va).get, va) }
    val genAnnotations = genVDS.variantsAndAnnotations.map { case (v, va) => (varidGenQuery(va).get, va) }

    assert(genAnnotations.fullOuterJoin(bgenAnnotations).forall { case (varid, (va1, va2)) => if (va1 == va2) true else false })

    val bgenFull = bgenVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((varidBgenQuery(va).get, s), gt) }
    val genFull = genVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((varidGenQuery(va).get, s), gt) }

    genFull.fullOuterJoin(bgenFull)
      .collect()
      .foreach { case ((v, i), (gt1, gt2)) =>
        assert(gt1 == gt2)
      }

    hadoopConf.delete(bgen + ".idx", recursive = true)
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

        val importedVariants = importedVds.variants
        val origVariants = vds.variants

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
}