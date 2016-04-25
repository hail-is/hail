package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.bgen.BgenLoader
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

    hadoopDelete(bgen + ".idx", sc.hadoopConfiguration, true)

    val nSamples = getNumberOfLinesInFile(sampleFile) - 2
    val nVariants = getNumberOfLinesInFile(gen)

    var s = State(sc, sqlContext, null)
    s = IndexBGEN.run(s, Array(bgen))
    s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", "10", bgen))
    assert(s.vds.nSamples == nSamples && s.vds.nVariants == nVariants)

    val genVDS = GenLoader(gen, sampleFile, sc)
    val bgenVDS = s.vds
    val genVariantsAnnotations = genVDS.variantsAndAnnotations
    val bgenVariantsAnnotations = bgenVDS.variantsAndAnnotations

    val bgenQuery = bgenVDS.vaSignature.query("varid")
    val genQuery = genVDS.vaSignature.query("varid")
    val bgenFull = bgenVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((bgenQuery(va).get, s), gt) }
    val genFull = genVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((genQuery(va).get, s), gt) }

    assert(bgenVDS.metadata == genVDS.metadata)
    assert(bgenVDS.sampleIds == genVDS.sampleIds)
    assert(bgenVariantsAnnotations.collect() sameElements genVariantsAnnotations.collect())
    genFull.fullOuterJoin(bgenFull)
      .collect()
      .foreach { case ((v, i), (gt1, gt2)) =>
        assert(gt1 == gt2)
      }
  }


  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _);
                       nPartitions: Int <- choose(1, 10)) yield (vds, nPartitions)

    writeTextFile("/tmp/sample_rename.txt", sc.hadoopConfiguration) { case w =>
      w.write("NA\tfdsdakfasdkfla")
    }

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        val vdsRemapped = vds.copy(rdd = vds.rdd.map { case (v, va, gs) => (v.copy(contig = "01"), va, gs) })

        val fileRoot = "/tmp/testGenWriter"
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"
        val qcToolLogFile = fileRoot + ".qctool.log"
        val qcToolPath = "qctool"

        hadoopDelete(bgenFile + ".idx", sc.hadoopConfiguration, true)
        hadoopDelete(bgenFile, sc.hadoopConfiguration, true)
        hadoopDelete(genFile, sc.hadoopConfiguration, true)
        hadoopDelete(sampleFile, sc.hadoopConfiguration, true)
        hadoopDelete(qcToolLogFile, sc.hadoopConfiguration, true)

        var s = State(sc, sqlContext, vdsRemapped)
        s = SplitMulti.run(s, Array[String]())
        s = RenameSamples.run(s, Array("-i", "/tmp/sample_rename.txt"))

        val origVds = s.vds

        GenWriter(fileRoot, s.vds, sc)

        s"src/test/resources/runExternalToolQuiet.sh $qcToolPath -force -g $genFile -s $sampleFile -og $bgenFile -log $qcToolLogFile" !

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

          val result = originalFull.fullOuterJoin(importedFull).map { case ((v, i), (gt1, gt2)) =>
            val gt1x = gt1 match {
              case Some(x) =>
                var newPl = x.pl.getOrElse(Array(0, 0, 0)).map { i => math.min(i, BgenLoader.MAX_PL) }
                val newGt = BgenUtils.parseGenotype(newPl)
                newPl = if (newGt == -1) null else newPl
                Some(x.copy(gt = Option(newGt), ad = None, dp = None, gq = None, pl = Option(newPl)))
              case None => None
            }

            if (gt1x == gt2)
              true
            else {
              if (gt1x.isDefined && gt2.isDefined)
                gt1x.get.pl.getOrElse(Array(BgenLoader.MAX_PL, BgenLoader.MAX_PL, BgenLoader.MAX_PL))
                  .zip(gt2.get.pl.getOrElse(Array(BgenLoader.MAX_PL, BgenLoader.MAX_PL, BgenLoader.MAX_PL)))
                  .forall { case (pl1, pl2) =>
                    if (math.abs(pl1 - pl2) <= 2) {
                      true
                    } else {
                      println(s"$v $i $gt1x $gt2")
                      false
                    }
                  }
              else {
                println(s"$v $i $gt1x $gt2")
                false
              }
            }
          }.fold(true)(_ && _)

          result
        }
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check(100, 100)
  }
}

object BgenUtils {
  def parseGenotype(pls: Array[Int]): Int = {
    if (pls(0) == 0 && pls(1) == 0
      || pls(0) == 0 && pls(2) == 0
      || pls(1) == 0 && pls(2) == 0)
      -1
    else {
      if (pls(0) == 0)
        0
      else if (pls(1) == 0)
        1
      else if (pls(2) == 0)
        2
      else
        -1
    }
  }
}