package is.hail.methods

import is.hail.annotations.{Annotation, Inserter}
import is.hail.expr.{EvalContext, Parser, TArray, TInt, TVariant}
import is.hail.sparkextras.OrderedRDD
import is.hail.utils._
import is.hail.variant.{GTPair, Genotype, Variant, VariantDataset}

import scala.collection.mutable
import scala.math.min

object FilterAlleles {

  def apply(vds: VariantDataset, filterExpr: String, annotationExpr: String = "va = va",
    filterAlteredGenotypes: Boolean = false, keep: Boolean = true,
    subset: Boolean = true, maxShift: Int = 100, keepStar: Boolean = false): VariantDataset = {

    if (vds.wasSplit)
      warn("this VDS was already split; this module was designed to handle multi-allelics, perhaps you should use filtervariants instead.")

    val conditionEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "aIndex" -> (2, TInt)))
    val conditionE = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, conditionEC)

    val annotationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "aIndices" -> (2, TArray(TInt))))
    val (paths, types, f) = Parser.parseAnnotationExprs(annotationExpr, annotationEC, Some(Annotation.VARIANT_HEAD))
    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.vaSignature) { case (vas, (path, signature)) =>
      val (newVas, i) = vas.insert(signature, path)
      inserterBuilder += i
      newVas
    }
    val inserters = inserterBuilder.result()

    def filterAllelesInVariant(v: Variant, va: Annotation): Option[(Variant, IndexedSeq[Int], Array[Int])] = {
      var alive = 0
      val oldToNew = new Array[Int](v.nAlleles)
      for (aai <- v.altAlleles.indices) {
        val index = aai + 1
        conditionEC.setAll(v, va, index)
        oldToNew(index) =
          if (Filter.boxedKeepThis(conditionE(), keep)) {
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
          .map { case (_, idx) => v.altAlleles(idx - 1) }
          .toArray

        Some((v.copy(altAlleles = altAlleles).minRep, newToOld: IndexedSeq[Int], oldToNew))
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
        val px = Genotype.px(g).map(downcodePx)
        g.copy(gt = Genotype.gt(g).map(downcodeGt),
          ad = Genotype.ad(g).map(downcodeAd),
          gq = px.map(Genotype.gqFromPL),
          px = px
        )
      }

      def subsetPx(px: Array[Int]): Array[Int] = {
        val (newPx, minP) = px.zipWithIndex
          .filter({
            case (p, i) =>
              val gTPair = Genotype.gtPair(i)
              (gTPair.j == 0 || oldToNew(gTPair.j) != 0) && (gTPair.k == 0 || oldToNew(gTPair.k) != 0)
          })
          .foldLeft((Array.fill(triangle(newCount))(0), Int.MaxValue))({
            case ((newPx, minP), (p, i)) =>
              newPx(downcodeGt(i)) = p
              (newPx, min(p, minP))
          })

        newPx.map(_ - minP)
      }

      def subsetGenotype(g: Genotype): Genotype = {
        if (g == null)
          null
        else {
          val px = Genotype.px(g).map(subsetPx)
          g.copy(
            gt = px.map(_.zipWithIndex.min._2),
            ad = Genotype.ad(g).map(_.zipWithIndex.filter({ case (d, i) => i == 0 || oldToNew(i) != 0 }).map(_._1)),
            gq = px.map(Genotype.gqFromPL),
            px = px)
        }
      }

      gs.map({
        g =>
          val newG = if (subset) subsetGenotype(g) else downcodeGenotype(g)
          if (filterAlteredGenotypes && Genotype.gt(newG) != Genotype.gt(g))
            newG.copy(gt = None)
          else
            newG
      })
    }

    def updateOrFilterRow(v: Variant, va: Annotation, gs: Iterable[Genotype],
      f: (Variant) => Boolean): Option[(Variant, (Annotation, Iterable[Genotype]))] =
      filterAllelesInVariant(v, va)
        .filter { case (v, _, _) => f(v) }
        .map { case (newV, newToOld, oldToNew) =>
          val newVa = updateAnnotation(v, va, newToOld)
          if (newV == v)
            (v, (newVa, gs))
          else {
            val newGs = updateGenotypes(gs, oldToNew, newToOld.length)
            (newV, (newVa, newGs))
          }
        }.filter { case (v, (va, gs)) => keepStar || !(v.isBiallelic && v.altAllele.isStar) }


    val partitionerBc = vds.sparkContext.broadcast(vds.rdd.orderedPartitioner)

    val shuffledVariants = vds.rdd.mapPartitionsWithIndex { case (i, it) =>
      it.flatMap { case (v, (va, gs)) =>
        updateOrFilterRow(v, va, gs,
          f = (v: Variant) => partitionerBc.value.getPartition(v) != i)
      }
    }.orderedRepartitionBy(vds.rdd.orderedPartitioner)

    val localMaxShift = maxShift
    val staticVariants = vds.rdd.mapPartitionsWithIndex { case (i, it) =>
      LocalVariantSortIterator(it.flatMap { case (v, (va, gs)) =>
        updateOrFilterRow(v, va, gs,
          f = (v: Variant) => partitionerBc.value.getPartition(v) == i)
      }, localMaxShift)
    }

    val newRDD = OrderedRDD.partitionedSortedUnion(staticVariants, shuffledVariants, vds.rdd.orderedPartitioner)

    vds.copy(rdd = newRDD, vaSignature = finalType)
  }
}
