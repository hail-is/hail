package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
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

  @Test def testMultipleUKBioBank10Variants() {
    val bgen = "/Users/jigold/ukbiobank_test10.chr*.bgen"
    val sampleFile = "/Users/jigold/ukbiobank_test10.sample"

    var s = State(sc, sqlContext, null)

    s = ImportBGEN.run(s,Array("-s",sampleFile,"-n","10", bgen))
    s = SplitMulti.run(s,Array[String]())
    s = ExportPlink.run(s, Array("-o","/tmp/testUkBiobank10var_hail"))
  }

  @Test def testGavinExample() {
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
  }

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

  @Test def testPhredTable() {
    val table = BgenLoader.phredConversionTable.zipWithIndex
    println(table.take(10).mkString(","))
    println(table.takeRight(10).mkString(","))
  }
}