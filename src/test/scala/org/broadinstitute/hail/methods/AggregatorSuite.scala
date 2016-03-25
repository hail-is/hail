package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.Genotype
import org.testng.annotations.Test

import scala.collection.mutable.ArrayBuffer

class AggregatorSuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    s = s.copy(vds = s.vds.copy(rdd = s.vds.rdd.filter { case (v, va, gs) => v.start == 16050036 }))
    s = AnnotateVariants.run(s, Array("-c", "va.countOver20 = gs.count(g.gq >= 20)"))
    val q = s.vds.queryVA("countOver20")
    s.vds.rdd.collect()
      .foreach {
        case (v, va, gs) =>
          val vaq = q(va)
          val gsC = gs.count(g => g.gq.isDefined && g.gq.get >= 20)
          assert(vaq == Option(gsC))
      }

    import org.broadinstitute.hail.expr
    val a = new ArrayBuffer[Any]()
    val a2 = new ArrayBuffer[Any]()
    val a3 = new ArrayBuffer[expr.Aggregator]()
    val cond = "sa.a = if (g.isHomVar) g.gq else 0"
    val cond2 = "sa.a = if (g.isHomVar) {g.gq} else {0}"
    val cond3 = "sa.a = g.isHomVar"
    val f = expr.Parser.parseAnnotationArgs(Map("g" ->(0, expr.TGenotype)), null, a, a2, a3, cond)
    val f2 = expr.Parser.parseAnnotationArgs(Map("g" ->(0, expr.TGenotype)), null, a, a2, a3, cond2)
    val f3 = expr.Parser.parseAnnotationArgs(Map("g" ->(0, expr.TGenotype)), null, a, a2, a3, cond3)
    println(f.head)
    println(f2.head)
    println(f3.head)
    val genotypes = vds.rdd.take(1).head._3.take(20).toArray
    val g1 = genotypes(0)
    println(genotypes.mkString("\n - "))

    //    s = AnnotateSamples.run(s, Array("-c", "sa.fractionOver20 = gs.fraction(g.gq >= 20), sa.hetmeangq = gs.sum(if (g.isHet) g.gq else 0).toDouble / gs.count(g.isHet)"))
    //    s = AnnotateSamples.run(s, Array("-c", "sa.fractionOver20 = gs.fraction(g.gq >= 20), sa.hetmeangq = gs.sum(if (g.isHet) g.gq else 0).toDouble / gs.count(g.isHet)"))
    //    s.vds.sampleAnnotations.foreach { x =>
    //      println(Annotation.printAnnotation(x, 2))
    //    }
  }
}
