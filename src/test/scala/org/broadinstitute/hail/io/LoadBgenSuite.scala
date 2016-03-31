package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Gen._
import scala.io.Source
import sys.process._
import scala.language.postfixOps

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

    val nSamples = getNumberOfLinesInFile(sampleFile) - 2
    val nVariants = getNumberOfLinesInFile(gen)

    var s = State(sc, sqlContext, null)
    s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", "10", bgen))
    assert(s.vds.nSamples == nSamples && s.vds.nVariants == nVariants)

    val genVDS = GenLoader(gen, sampleFile, sc)
    val bgenVDS = s.vds
    val genVariantsAnnotations = genVDS.variantsAndAnnotations
    val bgenVariantsAnnotations = bgenVDS.variantsAndAnnotations
    val bgenSampleIds = s.vds.sampleIds
    val genSampleIds = genVDS.sampleIds
    val bgenFull = bgenVDS.expandWithAnnotation().map{case (v, va, i, gt) => ((va.get("varid").toString,bgenSampleIds(i)),gt)}
    val genFull = genVDS.expandWithAnnotation().map{case (v, va, i, gt) => ((va.get("varid").toString,genSampleIds(i)),gt)}

    assert(bgenVDS.metadata == genVDS.metadata)
    assert(bgenVDS.sampleIds == genVDS.sampleIds)
    assert(bgenVariantsAnnotations.collect() sameElements genVariantsAnnotations.collect())
    assert(genFull.fullOuterJoin(bgenFull).map{case ((v,i),(gt1,gt2)) => gt1 == gt2}.fold(true)(_ && _))
  }


  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _);
                       nPartitions: Int <- choose(1, 20)) yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        val vdsRemapped = vds.copy(rdd = vds.rdd.map{case (v, va, gs) => (v.copy(contig="01"), va, gs)})

        val fileRoot = "/tmp/testGenWriter"
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"
        val qcToolLogFile = fileRoot + ".qctool.log"
        val qcToolPath = "/Users/jigold/Downloads/qctool_v1.4-osx/qctool" //FIXME: remove path

        hadoopDelete(fileRoot + "*", sc.hadoopConfiguration, true)
        hadoopDelete(bgenFile + ".idx", sc.hadoopConfiguration, true)
        hadoopDelete(genFile, sc.hadoopConfiguration, true)
        hadoopDelete(sampleFile, sc.hadoopConfiguration, true)
        hadoopDelete(qcToolLogFile, sc.hadoopConfiguration, true)

        var s = State(sc, sqlContext, vdsRemapped)
        s = SplitMulti.run(s, Array[String]())

        val origVds = s.vds

        GenWriter(fileRoot, s.vds, sc)

        s"src/test/resources/runExternalToolQuiet.sh $qcToolPath -force -g $genFile -s $sampleFile -og $bgenFile -log $qcToolLogFile" !

        if (vds.nVariants == 0)
          try {
            s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
            false
          } catch {
            case e:FatalException => true
            case _: Throwable => false
          }
        else {
          val q = ImportBGEN.run(State(sc, sqlContext, null), Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
          val importedVds = q.vds

          assert(importedVds.nSamples == origVds.nSamples)
          assert(importedVds.nVariants == origVds.nVariants)
          assert(importedVds.sampleIds == origVds.sampleIds)

          val importedVariants = importedVds.variants
          val origVariants = origVds.variants

          val importedSampleIds = importedVds.sampleIds
          val originalSampleIds = origVds.sampleIds

          val importedFull = importedVds.expandWithAnnotation().map{case (v, va, i, gt) => ((v, importedSampleIds(i)), gt)}
          val originalFull = origVds.expandWithAnnotation().map{case (v, va, i, gt) => ((v, originalSampleIds(i)), gt)}

          val result = originalFull.fullOuterJoin(importedFull).map{ case ((v, i), (gt1, gt2)) =>
            val gt1x = gt1 match {
              case Some(x) => {
                var newPl = x.pl.getOrElse(Array(0,0,0)).map{i => math.min(i, 48)}
                val newGt = BgenLoader.parseGenotype(newPl)
                newPl = if (newGt == -1) null else newPl
                Some(x.copy(gt = Option(newGt), ad = None, dp = None, gq = None, pl = Option(newPl)))
              }
              case None => None
            }
            gt1x == gt2
          }.fold(true)(_ && _)

          result
        }
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check()
  }
}