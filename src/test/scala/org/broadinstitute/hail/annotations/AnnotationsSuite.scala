package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.{Genotype, IntervalList, Variant}
import org.scalacheck.Gen
import org.testng.annotations.Test
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.methods.FilterVariantCondition

class AnnotationsSuite extends SparkSuite {

  def getFunction(cond: String, vas: AnnotationSignatures): (Variant, AnnotationData) => Boolean = {
    cond match {
      case c: String =>
        try {
          val cf = new FilterVariantCondition(c, vas)
          cf.typeCheck()
          cf.apply
        } catch {
          case e: scala.tools.reflect.ToolBoxError =>
            /* e.message looks like:
               reflective compilation has failed:

               ';' expected but '.' found. */
            fatal("parse error in condition: " + e.message.split("\n").last)
        }
    }
  }

  @Test def test() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)

//    val vds = LoadVCF(sc, "src/test/resources/linearRegression.vcf")
    val vas = vds.metadata.variantAnnotationSignatures


    //FIXME involve every type of thing we can generate, involve options... look at vcf spec 4.2 ...
    val vTotal = vds.nVariants

    val cond1 = "true"
    val p1 = getFunction(cond1, vas)
    assert(vds.filterVariants(p1).nVariants == vTotal)


    val cond2 = "va.info.FS == 0"
    val p2 = getFunction(cond2, vas)
    assert(vds.filterVariants(p2).nVariants == 132)

    val cond3 = "va.info.HWP == 1"
    val p3 = getFunction(cond3, vas)
    assert(vds.filterVariants(p3).nVariants == 159)

    val state2 = VariantQC.run(state, Array("--store"))
//    state2.vds.metadata.variantAnnotationSignatures.maps.foreach{case (k,m) =>
    // m.foreach {case (k2,ss) => println(k2 + " " + ss.conversion)} }
//    state2.vds.rdd.map { case (v,va,gs) => va }
//      .collect()
//      .apply(1)
//      .maps("qc").foreach(println(_))
    println(FilterVariants.run(state2, Array("--keep", "-c", "(va.qc.MAF.isDefined && va.qc.MAF.get > 0.05)"))
      .vds
      .nVariants)

//    FilterGenotypes.run(state2, Array("--keep", "-c", "g.dp > 100")).vds
//      .rdd
//      .map { case (v, va, gs)  => (v, va, gs.toArray) }
//      .collect()(0)._3.foreach(println)
    ExportVariants.run(state2, Array("--output",
      "src/test/resources/sample.vcf.exportVariants", "-c", "v.contig,v.start,va.qc.rHetHomVar,va.qc.MAF,va.qc.dpMean"))
    val state3 = SampleQC.run(state2, Array("--store"))
    ExportSamples.run(state3, Array("--output", "src/test/resources/sample.vcf.exportSamples", "-c", "s.id,sa.qc.dpMean,sa.qc.nHet"))
//    assert({val nV = .vds.nVariants;
//      println(s"nV = $nV"); nV > 0})
  }
}
