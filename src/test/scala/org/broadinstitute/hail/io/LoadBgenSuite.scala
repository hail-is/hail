package org.broadinstitute.hail.io

import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Gen._
import sys.process._
import scala.language.postfixOps

class LoadBgenSuite extends SparkSuite {

//  object Spec extends Properties("ImportBGEN") {
//    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _);
//                       nPartitions: Int <- choose(1, BgenLoader.expectedSize(vds.nSamples, vds.nVariants).toInt.min(50))) yield (vds, nPartitions)
//
//    property("import generates same output as export") =
//      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>
//
//        println(s"nPartitions:$nPartitions nSamples:${vds.nSamples} nVariants:${vds.nVariants}")
//        var s = State(sc, sqlContext, vds)
//
//        s = SplitMulti.run(s, Array[String]())
//        s = ExportVCF.run(s, Array("-o","/tmp/truth.vcf"))
//
//        val exitcode = "/Users/jigold/Downloads/qctool_v1.4-osx/qctool -g /tmp/truth.vcf -og /tmp/truth.bgen" !
//
//        s = ImportBGEN.run(s, Array("/tmp/truth.bgen","-n",nPartitions.toString))
//
//        true
//
//      }
//  }
//
//  @Test def testBGENrandom() {
//    //Spec.check()
//  }

  @Test def test() {
    val munge1a = BgenLoader.mungeIndel("13:104613359:A_AAT", "R", "I")
    val munge1b = BgenLoader.mungeIndel("13:104613359:A_AAT", "I", "R")
    println("From 13:104613359:A_AAT, R, I => %s, %s".format(munge1a._1, munge1a._2))
    println("From 13:104613359:A_AAT, I, R => %s, %s".format(munge1b._1, munge1b._2))
    val munge2a = BgenLoader.mungeIndel("13:104620197:TGAA_", "R", "D")
    val munge2b = BgenLoader.mungeIndel("13:104620197:TGAA_", "D", "R")
    println("From 13:104620197:TGAA_, R, D => %s, %s".format(munge2a._1, munge2a._2))
    println("From 13:104620197:TGAA_, D, R => %s, %s".format(munge2b._1, munge2b._2))
//    val loader = BgenLoader("src/test/resources/11_36760324_51539425.bgen", sc)
//    val loader = BgenLoader("src/test/resources/biggerM.bgen", sc)
//    val loader = BgenLoader("src/test/resources/indelgen.bgen", sc)
//    val loader = BgenLoader("src/test/resources/bigM.bgen", sc)
  }

  @Test def testImportBGEN() {
    //val bgen = "src/test/resources/example.chr22.reference.bgen"
    val bgen = "/Users/jigold/testgen.bgen"
    var s = State(sc, sqlContext, null)
    s = ImportBGEN.run(s, Array(bgen,"-n","1"))
  }

  @Test def testRandomlyGeneratedGenFile() {
    //val bgen = "src/test/resources/example.chr22.reference.bgen"
    val bgen = "/Users/jigold/testgen.bgen"
    val sampleFile = "/Users/jigold/testgen.sample"
    var s = State(sc, sqlContext, null)
    s = ImportBGEN.run(s, Array(bgen,"-s",sampleFile,"-n","5"))
    s = SplitMulti.run(s,Array[String]())
    s = ExportPlink.run(s, Array("-o","/tmp/testgen_hail"))

    val exitCode = s"/Users/jigold/plink --bgen $bgen --sample $sampleFile --const-fid --keep-allele-order --make-bed --out /tmp/testgen_plink" !

    val diffCodeBed = s"diff /tmp/testgen_hail.bed /tmp/testgen_plink.bed" !
    val diffCodeBim = s"diff /tmp/testgen_hail.bim /tmp/testgen_plink.bim" !
    val diffCodeFam = s"diff /tmp/testgen_hail.fam /tmp/testgen_plink.fam" !

    assert(diffCodeBed == 0 && diffCodeBim == 0 && diffCodeFam == 0)
  }

}