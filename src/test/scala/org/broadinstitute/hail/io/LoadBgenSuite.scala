package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
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

  /*  @Test def testMultipleUKBioBank10Variants() {
    val bgen = "/Users/jigold/ukbiobank_test10.chr*.bgen"
    val sampleFile = "/Users/jigold/ukbiobank_test10.sample"

    var s = State(sc, sqlContext, null)

    s = ImportBGEN.run(s,Array("-s",sampleFile,"-n","10", bgen))
    s = SplitMulti.run(s,Array[String]())
    s = ExportPlink.run(s, Array("-o","/tmp/testUkBiobank10var_hail"))
  }*/

  /* @Test def testGavinExample() {
    val gen = "/Users/jigold/bgen_test/example.gen"
    val sampleFile = "/Users/jigold/bgen_test/example.sample"
    val bgen = "/Users/jigold/bgen_test/example.v11.bgen"

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
    //println(genFull.fullOuterJoin(bgenFull).filter{case ((v,i),(gt1,gt2)) => gt1 != gt2}.map{case ((v,i),(gt1,gt2)) => (v, i, gt1, gt2)}.take(20).mkString("\n"))
  }*/

  /*  @Test def testParse() {
    val pAA = 0.0
    val pAB = 0.0
    val pBB = 0.0
    val foo = (pAA, pAB, pBB) match {
      case (0.0, 0.0, 0.0) => -1
      case (0.0, x, 0.0) => 1
      case (x, 0.0, 0.0) => 0
      case (0.0, 0.0, x) => 2
    }

  }*/

  /*  def isPlEqualDosage(dosages: Array[Double]): Boolean = {
    val ints = GenLoader.convertPPsToInt(dosages)
    val pls = BgenLoader.phredScalePPs(ints(0), ints(1), ints(2))

  }*/

/*  @Test def failedRandomBgen(){
    val bgenFile = "/Users/jigold/bgen_test/jackie_example1.bgen"
    val sampleFile = "/Users/jigold/bgen_test/jackie_example1.sample"
    var s = State(sc, sqlContext, null)
    s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", "5", bgenFile))
  }*/

  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _);
                       nPartitions: Int <- choose(1, 20)) yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>
        val fileRoot = "/tmp/testGenWriter"
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"
        val qcToolLogFile = fileRoot + ".qctool.log"
        val qcTool = "/Users/jigold/Downloads/qctool_v1.4-osx/qctool"

        hadoopDelete(fileRoot + "*", sc.hadoopConfiguration, true)
        hadoopDelete(bgenFile + ".idx", sc.hadoopConfiguration, true)

        var s = State(sc, sqlContext, vds)
        s = SplitMulti.run(s, Array[String]())

        val origVds = s.vds
        println(s"nPartitions=$nPartitions nSamples=${origVds.nSamples} nVariants=${origVds.nVariants}")

        GenWriter(fileRoot, s.vds, sc)
        s"$qcTool -force -g $genFile -s $sampleFile -og $bgenFile -log $qcToolLogFile" !

        if (vds.nSamples == 0 || vds.nVariants == 0)
          try {
            s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
            false
          } catch {
            case e:FatalException => true
            case _ => false
          }
        else {
          val q = ImportBGEN.run(State(sc, sqlContext, null), Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
          val importedVds = q.vds

          assert(importedVds.nSamples == origVds.nSamples)
          assert(importedVds.nVariants == origVds.nVariants)
          assert(importedVds.sampleIds == origVds.sampleIds)
          val importedVariants = importedVds.variants.collect()
          val origVariants = origVds.variants.collect()
          println(importedVariants.take(5).mkString(","))
          println(origVariants.take(5).mkString(","))
          assert(importedVds.variants.collect() sameElements origVds.variants.collect())
        }

        true
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check()
  }
}