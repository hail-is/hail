package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, Inserter}
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators.SampleTSVAnnotator
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object AnnotateSamplesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation condition")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples/expr"

  def description = "Annotate samples in current dataset"

  override def hidden = true

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition
    val symTab = Map(
      "s" ->(0, expr.TSample),
      "sa" ->(1, vds.saSignature),
      "gs" -> (2, expr.TGenotypeStream))

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
      if (ids.head != "sa")
        fatal(s"expect 'sa[.identifier]+', got ${ids.mkString(".")}")
      (ids.tail, t)
    }
    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
      val (s, i) = v.insertSA(signature, ids)
      inserterBuilder += i
      v.copy(saSignature = s)
    }

    val computations = parsed.map(_._3)
    val inserters = inserterBuilder.result()

    val doAggregates = a3.nonEmpty
    val aggregatorArray = if (doAggregates) {
      val a3arr = a3.toArray
      val sampleInfoBc = vds.sparkContext.broadcast(vds.localSamples
        .map(vds.sampleIds)
        .map(Sample)
        .zip(vds.localSamples.map(vds.sampleAnnotations)))
      vds.rdd.aggregate(Array.fill[Array[Any]](vds.nLocalSamples)(a3arr.map(_._1())))({ case (arr, (v, va, gs)) =>
        gs.iterator
          .zipWithIndex
          .foreach { case (g, i) =>
            a2(0) = v
            a2(1) = va
            a2(2) = sampleInfoBc.value(i)._1
            a2(3) = sampleInfoBc.value(i)._2
            a2(4) = g

            a3arr.iterator
              .zipWithIndex
              .foreach { case ((zv, seqOp, combOp), j) =>
                val iArray = arr(i)
                iArray(j) = seqOp(iArray(j))
              }
          }
        //            println(arr.map(subarr => "(" + subarr.mkString(",") + ")").mkString("|"))
        arr
      }, { case (arr1, arr2) =>
        val combOp = a3arr.map(_._3)
        arr1.iterator
          .zip(arr2.iterator)
          .map { case (ai1, ai2) =>
            ai1.iterator
              .zip(ai2.iterator)
              .zip(combOp.iterator)
              .map { case ((ij1, ij2), c) => c(ij1, ij2) }
              .toArray
          }
          .toArray
      })
    } else null

    val newAnnotations = vdsAddedSigs.sampleAnnotations.zipWithIndex.map { case (sa, i) =>
      a(0) = Sample(vds.sampleIds(i))
      a(1) = sa
      a(2) = 0 //FIXME placeholder?

      if (doAggregates) {
        aggregatorArray(i).iterator.zipWithIndex
          .foreach { case (value, j) =>
            a2(5 + j) = value }
      }

      val queries = computations.map(_ ())
      queries.indices.foreach { i =>
        a(1) = inserters(i).apply(
          a(1),
          Option(queries(i)))
      }
      a(1): Annotation
    }
    val annotated = vdsAddedSigs.copy(sampleAnnotations = newAnnotations)
    state.copy(vds = annotated)
  }
}
