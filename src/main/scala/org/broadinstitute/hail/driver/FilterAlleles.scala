
package org.broadinstitute.hail.driver

import scala.math.min
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{EvalContext, Parser, TArray, TBoolean, TGenotype, TInt, TSample, TVariant}
import org.broadinstitute.hail.methods.Filter
import org.broadinstitute.hail.variant.{AltAllele, GTPair, Genotype, Variant}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.{GenTraversableOnce, mutable}
import scala.reflect.ClassTag

object FilterAlleles extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter expression involving v (variant), va (variant annotations), and aIndex (allele index)",
      metaVar = "COND_EXPR")
    var condition: String = _

    @Args4jOption(required = true, name = "-a", aliases = Array("--annotation"),
      usage = "Annotation modifying expression involving v (new variant), va (old variant annotations), and aIndices (maps from new to old indices)",
      metaVar = "ANNO_EXPR")
    var annotation: String = _
  }

  override def newOptions: Options = new Options

  override def name: String = "filteralleles"

  override def description: String = "Filter alleles in current dataset using the Hail expression language"

  override def supportsMultiallelic: Boolean = true

  override def requiresVDS: Boolean = true

  override protected def run(state: State, options: Options): State = {
    if (state.vds.wasSplit)
      warn("this VDS was already split; this module was designed to handle multi-allelics, perhaps you should use filtervariants instead.")

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val conditionEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, state.vds.vaSignature),
      "aIndex" -> (2, TInt)))
    val conditionE = Parser.parse[Boolean](options.condition, conditionEC, TBoolean)

    val annotationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, state.vds.vaSignature),
      "aIndices" -> (2, TArray(TInt))))
    val (types, generators) = Parser.parseAnnotationArgs(options.annotation, annotationEC, Annotation.VARIANT_HEAD)
    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = types.foldLeft(state.vds.vaSignature) { case (vas, (path, signature)) =>
      val (newVas, i) = vas.insert(signature, path)
      inserterBuilder += i
      newVas
    }
    val inserters = inserterBuilder.result()

    val keep = options.keep

    def filterAllelesInVariant(v: Variant, va: Annotation): Option[(Variant, IndexedSeq[Int], Array[Int])] = {
      var alive = 0
      val oldToNew = new Array[Int](v.nAlleles)
      for (aai <- v.altAlleles.indices) {
        val index = aai + 1
        conditionEC.setAll(v, va, index)
        oldToNew(index) =
          if (Filter.keepThis(conditionE(), keep)) {
            alive += 1
            alive
          } else
            0
      }

      if (alive == 0)
        None
      else {
        val newToOld = oldToNew.iterator
            .zipWithIndex
            .filter { case (newIdx, oldIdx) => oldIdx == 0 || newIdx != 0 }
            .map(_._2)
            .toArray

        val altAlleles = oldToNew.iterator
            .zipWithIndex
            .filter { case (newIdx, _) => newIdx != 0 }
            .map { case (_, idx) => v.altAlleles(idx-1) }
            .toArray

        Some((v.copy(altAlleles = altAlleles), newToOld : IndexedSeq[Int], oldToNew))
      }
    }

    def updateAnnotation(v: Variant, va: Annotation, newToOld: IndexedSeq[Int]): Annotation = {
      annotationEC.setAll(v, va, newToOld)
      generators.zip(inserters).foldLeft(va) { case (va, (fn, inserter)) => inserter(va, fn()) }
    }

    def updateGenotypes(gs: Iterable[Genotype], oldToNew: Array[Int], newCount: Int): Iterable[Genotype] = {
      def downcodeGtPair(gt: GTPair): GTPair =
        GTPair.fromNonNormalized(oldToNew(gt.j), oldToNew(gt.k))
      def downcodeGt(gt: Int): Int =
        Genotype.gtIndex(downcodeGtPair(Genotype.gtPair(gt)))

      def downcodeAd(ad: Array[Int]): Array[Int] = {
        coalesce(ad)(newCount, (_, alleleIndex) => oldToNew(alleleIndex), 0) { (oldDepth, depth) =>
          oldDepth + depth
        }
      }

      def downcodePx(px: Array[Int]): Array[Int] = {
        coalesce(px)(Utils.triangle(newCount), (_, gt) => downcodeGt(gt), Int.MaxValue) { (oldNll, nll) =>
          min(oldNll, nll)
        }
      }

      def downcodeGenotype(g: Genotype): Genotype = {
        val px = g.px.map(downcodePx)
        val gq = px.map(Genotype.gqFromPL)
        Genotype(g.gt.map(downcodeGt),
          g.ad.map(downcodeAd),
          g.dp,
          gq,
          px,
          g.fakeRef,
          g.isDosage
        )
      }

      gs.map(downcodeGenotype)
    }

    def updateOrFilterRow(v: Variant, va: Annotation, gs: Iterable[Genotype]): Option[(Variant, (Annotation, Iterable[Genotype]))] =
      filterAllelesInVariant(v, va).map { case (newV, newToOld, oldToNew) =>
        val newVa = updateAnnotation(v, va, newToOld)
        val newGs = updateGenotypes(gs, oldToNew, newToOld.length)
        (newV, (newVa, newGs))
      }

    val newVds = state.vds
      .flatMapVariants { case (v, va, gs) => updateOrFilterRow(v, va, gs) }

    state.copy(vds = newVds)
  }
}