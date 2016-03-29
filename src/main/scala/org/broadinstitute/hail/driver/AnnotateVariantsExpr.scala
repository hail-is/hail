package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object AnnotateVariantsExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants/expr"

  override def hidden = true

  def description = "Annotate variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "gs" ->(2, TGenotypeStream))
    val symTab2 = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)
    )
    val a = new ArrayBuffer[Any]()
    val a2 = new ArrayBuffer[Any]()
    val a3 = new ArrayBuffer[Aggregator]()

    for (_ <- symTab)
      a += null
    for (_ <- symTab2)
      a2 += null
    val parsed = expr.Parser.parseAnnotationArgs(symTab, symTab2, a, a2, a3, cond)


    val keyedSignatures = parsed.map { case (ids, t, f) =>
      if (ids.head != "va")
        fatal(s"expect 'va[.identifier]+', got ${ids.mkString(".")}")
      (ids.tail, t)
    }

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val computations = parsed.map(_._3)

    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertVA(signature, ids)
      inserterBuilder += i
      v.copy(vaSignature = s)
    }

    val inserters = inserterBuilder.result()

    val sampleInfoBc = vds.sparkContext.broadcast(
      vds.localSamples.map(vds.sampleAnnotations)
        .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))

    val annotated = vdsAddedSigs.mapAnnotations { case (v, va, gs) =>
      a(0) = v
      a(1) = va
      a(2) = gs
      if (a3.nonEmpty) {
        val gsQueries = a3.toArray.map(_._1())
        gs.iterator
          .zip(sampleInfoBc.value.iterator)
          .foreach {
            case (g, (sa, s)) =>
              a2(0) = v
              a2(1) = va
              a2(2) = s
              a2(3) = sa
              a2(4) = g
              a3.iterator.zipWithIndex
                .foreach {
                  case ((zv, so, co), i) =>
                    gsQueries(i) = so(gsQueries(i))
                }
          }
        gsQueries.iterator.zipWithIndex
          .foreach { case (res, i) =>
            a2(5 + i) = res
          }
      }

      computations.indices.foreach { i =>
        a(1) = inserters(i).apply(a(1), Option(computations(i)()))
      }
      a(1): Annotation
    }
    state.copy(vds = annotated)
  }

}
