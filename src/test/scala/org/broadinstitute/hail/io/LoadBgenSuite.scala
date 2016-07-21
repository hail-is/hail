package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.{Gen, Properties}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

class LoadBgenSuite extends SparkSuite {

  def getNumberOfLinesInFile(file: String): Long = {
    readFile(file, sc.hadoopConfiguration) { s =>
      Source.fromInputStream(s)
        .getLines()
        .length
    }.toLong
  }

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"
    val bgen = "src/test/resources/example.v11.bgen"
    val fileRoot = tmpDir.createTempFile(prefix = "exampleInfoScoreTest")
    val qcToolLogFile = fileRoot + ".qctool.log"
    val statsFile = fileRoot + ".stats"
    val qcToolPath = "qctool"

    hadoopDelete(bgen + ".idx", sc.hadoopConfiguration, recursive = true)

    val nSamples = getNumberOfLinesInFile(sampleFile) - 2
    val nVariants = getNumberOfLinesInFile(gen)

    var s = State(sc, sqlContext, null)
    s = IndexBGEN.run(s, Array(bgen))
    s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", "10", bgen))
    assert(s.vds.nSamples == nSamples && s.vds.nVariants == nVariants)

    val genVDS = ImportGEN.run(s, Array("-s", sampleFile, gen)).vds
    val bgenVDS = s.vds

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

    hadoopDelete(bgen + ".idx", sc.hadoopConfiguration, recursive = true)
  }

  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds <- VariantSampleMatrix.gen(sc,
      VSMSubgen.dosage.copy(vGen = VSMSubgen.dosage.vGen.map(v => v.copy(contig = "01")))) if vds.nVariants != 0;
      nPartitions <- choose(1, 10))
      yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    writeTextFile(sampleRenameFile, sc.hadoopConfiguration)(_.write("NA\tfdsdakfasdkfla"))

    property("import generates same output as export") =
      forAll(compGen) { case (vds, nPartitions) =>

        assert(vds.rdd.forall { case (v, (va, gs)) =>
          gs.flatMap(_.dosage).flatten.forall(d => d >= 0.0 && d <= 1.0)
        })

        val fileRoot = tmpDir.createTempFile(prefix = "testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"
        val qcToolLogFile = fileRoot + ".qctool.log"
        val statsFile = fileRoot + ".stats"

        hadoopDelete(bgenFile + ".idx", sc.hadoopConfiguration, recursive = true)
        hadoopDelete(bgenFile, sc.hadoopConfiguration, recursive = true)
        hadoopDelete(genFile, sc.hadoopConfiguration, recursive = true)
        hadoopDelete(sampleFile, sc.hadoopConfiguration, recursive = true)
        hadoopDelete(qcToolLogFile, sc.hadoopConfiguration, recursive = true)

        var s = State(sc, sqlContext, vds)
        s = SplitMulti.run(s, Array[String]())
        s = RenameSamples.run(s, Array("-i", sampleRenameFile))

        val origVds = s.vds

        s = ExportGEN.run(s, Array("-o", fileRoot))

        val rc = s"qctool -force -g $genFile -s $sampleFile -og $bgenFile -log $qcToolLogFile" !

        assert(rc == 0)

        var q = IndexBGEN.run(State(sc, sqlContext, null), Array(bgenFile))
        q = ImportBGEN.run(State(sc, sqlContext, null), Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))

        val importedVds = q.vds

        assert(importedVds.nSamples == origVds.nSamples)
        assert(importedVds.nVariants == origVds.nVariants)
        assert(importedVds.sampleIds == origVds.sampleIds)

        val importedVariants = importedVds.variants
        val origVariants = origVds.variants

        val importedFull = importedVds.expandWithAll().map { case (v, va, s, sa, g) => ((v, s), g) }
        val originalFull = origVds.expandWithAll().map { case (v, va, s, sa, g) => ((v, s), g) }

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