
package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{EvalContext, Parser, TArray, TBoolean, TInt, TVariant}
import org.broadinstitute.hail.methods.Filter
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{GTPair, Genotype, GenotypeType, Variant}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.math.min

object FilterAlleles extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

    @Args4jOption(required = false, name = "--downcode", usage = "If set, downcodes the PL and AD." +
      "Genotype and GQ are set based on the resulting PLs.")
    var downcode: Boolean = false

    @Args4jOption(required = false, name = "--subset", usage = "If set, subsets the PL and AD." +
      "Genotype and GQ are set based on the resulting PLs.")
    var subset: Boolean = false

    @Args4jOption(required = false, name = "--annotate-all-variants", usage = "If set, annotates all variants" +
      "with the -a condition, otherwise only variants where an allele was filtered get annotated.")
    var annotateAll: Boolean = false

    @Args4jOption(required = false, name = "--filterAlteredGenotypes", usage = "If set, any genotype call that would change due" +
      " to filtering an allele would be set to missig instead.")
    var filterAlteredGenotypes: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter expression involving v (variant), va (variant annotations), and aIndex (allele index)",
      metaVar = "COND_EXPR")
    var condition: String = _

    @Args4jOption(required = false, name = "-a", aliases = Array("--annotation"),
      usage = "Annotation modifying expression involving v (new variant), va (old variant annotations), and aIndices (maps from new to old indices)",
      metaVar = "ANNO_EXPR")
    var annotation: String = "va = va"
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

    if (!(options.downcode ^ options.subset))
      fatal("either `--downcode' or `--subset' required, but not both")

    val conditionEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, state.vds.vaSignature),
      "aIndex" -> (2, TInt)))
    val conditionE = Parser.parseTypedExpr[Boolean](options.condition, conditionEC)

    val annotationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, state.vds.vaSignature),
      "aIndices" -> (2, TArray(TInt))))
    val (paths, types, f) = Parser.parseAnnotationExprs(options.annotation, annotationEC, Some(Annotation.VARIANT_HEAD))
    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(state.vds.vaSignature) { case (vas, (path, signature)) =>
      val (newVas, i) = vas.insert(signature, path)
      inserterBuilder += i
      newVas
    }
    val inserters = inserterBuilder.result()

    val keep = options.keep
    val downcode = options.downcode
    val filterAlteredGenotypes = options.filterAlteredGenotypes
    val annotateAll = options.annotateAll

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
      f().zip(inserters).foldLeft(va) { case (va, (v, inserter)) => inserter(va, v) }
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
        coalesce(px)(triangle(newCount), (_, gt) => downcodeGt(gt), Int.MaxValue) { (oldNll, nll) =>
          min(oldNll, nll)
        }
      }

      def downcodeGenotype(g: Genotype): Genotype = {
        val px = g.px.map(downcodePx)
        g.copy(gt = g.gt.map(downcodeGt),
          ad = g.ad.map(downcodeAd),
          gq = px.map(Genotype.gqFromPL),
          px = px
        )
      }

      def subsetPx(px: Array[Int]): Array[Int] = {
        val (newPx,minP) = px.zipWithIndex
          .filter({
            case (p, i) =>
              val gTPair = Genotype.gtPair(i)
              (gTPair.j == 0 || oldToNew(gTPair.j) != 0) && (gTPair.k ==0 || oldToNew(gTPair.k) != 0)
          })
          .foldLeft((Array.fill(triangle(newCount))(0),Int.MaxValue))({
            case((newPx,minP),(p,i)) =>
              newPx(downcodeGt(i)) = p
              (newPx,min(p,minP))
          })

        newPx.map(_ - minP)
      }

      def subsetGenotype(g: Genotype) : Genotype = {
        val px  = g.px.map(subsetPx)
        g.copy(
          gt = px.map(_.zipWithIndex.min._2),
          ad = g.ad.map(_.zipWithIndex.filter({case(d,i) => i ==0 || oldToNew(i) != 0}).map(_._1)),
          gq = px.map(Genotype.gqFromPL),
          px = px
        )
      }

      gs.map({
        g =>
          val newG = if(downcode) downcodeGenotype(g) else subsetGenotype(g)
          if(filterAlteredGenotypes && newG.gt != g.gt)
            newG.copy(gt = None)
          else
            newG
      })
    }

    def updateOrFilterRow(v: Variant, va: Annotation, gs: Iterable[Genotype]): Option[(Variant, (Annotation, Iterable[Genotype]))] =
      filterAllelesInVariant(v, va).map { case (newV, newToOld, oldToNew) =>
        if(newV == v)
          if(annotateAll) (v,(updateAnnotation(v, va, newToOld),gs))
          else (v, (va, gs))
        else {
          val newGs = updateGenotypes(gs, oldToNew, newToOld.length)
          (newV, (updateAnnotation(v, va, newToOld), newGs))
        }
      }

    val newVds = state.vds
      .flatMapVariants { case (v, va, gs) => updateOrFilterRow(v, va, gs) }
      .copy(vaSignature = finalType)

    state.copy(vds = newVds)
  }
}