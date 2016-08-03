package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.{FatalException, SparkSuite}
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

    hadoopDelete(bgen + ".idx", sc.hadoopConfiguration, true)

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

    hadoopDelete(bgen + ".idx", sc.hadoopConfiguration, true)
  }

  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.dosage);
                       nPartitions: Int <- choose(1, 10)) yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    writeTextFile(sampleRenameFile, sc.hadoopConfiguration) { case w =>
      w.write("NA\tfdsdakfasdkfla")
    }

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        val vdsRemapped = vds.copy(rdd = vds.rdd.map { case (v, (va, gs)) => (v.copy(contig = "01"), (va, gs)) })

        assert(vdsRemapped.rdd.forall { case (v, (va, gs)) =>
          gs.forall { case g =>
            g.dosage.forall(dx =>
              dx.forall { case d =>
                d >= 0.0 && d <= 1.0
              }
            )
          }
        })

        val fileRoot = tmpDir.createTempFile(prefix = "testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"
        val qcToolLogFile = fileRoot + ".qctool.log"
        val statsFile = fileRoot + ".stats"
        val qcToolPath = "qctool"

        hadoopDelete(bgenFile + ".idx", sc.hadoopConfiguration, true)
        hadoopDelete(bgenFile, sc.hadoopConfiguration, true)
        hadoopDelete(genFile, sc.hadoopConfiguration, true)
        hadoopDelete(sampleFile, sc.hadoopConfiguration, true)
        hadoopDelete(qcToolLogFile, sc.hadoopConfiguration, true)

        var s = State(sc, sqlContext, vdsRemapped)
        s = SplitMulti.run(s, Array[String]())
        s = RenameSamples.run(s, Array("-i", sampleRenameFile))

        val origVds = s.vds

        s = ExportGEN.run(s, Array("-o", fileRoot))

        s"sh src/test/resources/runExternalToolQuiet.sh $qcToolPath -force -g $genFile -s $sampleFile -og $bgenFile -log $qcToolLogFile" !

        if (vds.nVariants == 0)
          try {
            s = IndexBGEN.run(s, Array("-n", nPartitions.toString, bgenFile))
            s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
            false
          } catch {
            case e: FatalException => true
            case _: Throwable => false
          }
        else {
          var q = IndexBGEN.run(State(sc, sqlContext, null), Array(bgenFile))
          q = ImportBGEN.run(State(sc, sqlContext, null), Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))

          val importedVds = q.vds

          assert(importedVds.nSamples == origVds.nSamples)
          assert(importedVds.nVariants == origVds.nVariants)
          assert(importedVds.sampleIds == origVds.sampleIds)

          val importedVariants = importedVds.variants
          val origVariants = origVds.variants

          val importedFull = importedVds.expandWithAll().map { case (v, va, s, sa, gt) => ((v, s), gt) }
          val originalFull = origVds.expandWithAll().map { case (v, va, s, sa, gt) => ((v, s), gt) }

          originalFull.fullOuterJoin(importedFull).forall { case ((v, i), (gt1, gt2)) =>
            if (gt1 == gt2)
              true
            else {
              (gt1, gt2) match {
                case (None, None) => true
                case (None, Some(y)) =>
                  println(s"ERROR: gts not equal; orig=${gt1} hail=${gt2}")
                  false
                case (Some(x), None) =>
                  println(s"ERROR: gts not equal; orig=${gt1} hail=${gt2}")
                  false
                case (Some(x), Some(y)) =>

                  if (x.dosage.get.zip(y.dosage.get).forall{case (dx, dy) => math.abs(dx - dy) <= 3e-4}) {
                    if (x.gt == y.gt)
                      true
                    else {
                      val maxDosageY = y.dosage.get.max
                      if (y.dosage.get.count(d => d == maxDosageY) > 1 && y.dosage.get(x.gt.get) == maxDosageY) {
                        println(s"WARN: gts unequal because no max dosage in imported dosage; orig=${gt1} hail=${gt2}")
                        true
                      } else if (x.gt.isEmpty) {
                        println(s"WARN: gts unequal because original gt was None; orig=${gt1} hail=${gt2}")
                        true
                      } else {
                        println(s"ERROR: gts not equal but dosages equal; orig=${gt1} hail=${gt2}")
                        false
                      }
                    }
                  } else {
                    println(s"ERROR: dosages not equal; orig=${gt1} hail=${gt2}")
                    false
                  }
              }
            }
          }
        }
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check(100, 50, Option(20))
  }
}