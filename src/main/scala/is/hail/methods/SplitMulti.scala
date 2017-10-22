package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.sparkextras.OrderedRDD
import is.hail.utils._
import is.hail.variant.{Genotype, GenotypeBuilder, Variant, VariantDataset}

object SplitMulti {

  def splitGT(gt: Int, i: Int): Int = {
    val p = Genotype.gtPair(gt)
    (if (p.j == i) 1 else 0) +
      (if (p.k == i) 1 else 0)
  }

  def split(v: Variant,
    va: Annotation,
    it: Iterable[Genotype],
    propagateGQ: Boolean,
    keepStar: Boolean,
    insertSplitAnnots: (Annotation, Int, Boolean) => Annotation,
    f: (Variant) => Boolean): Iterator[(Variant, (Annotation, Iterable[Genotype]))] = {

    if (v.isBiallelic) {
      val minrep = v.minRep
      if (f(minrep))
        return Iterator((minrep, (insertSplitAnnots(va, 1, false), it)))
      else
        return Iterator()
    }

    val splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(keepStar || !_._1.isStar)
      .map { case (aa, aai) => (Variant(v.contig, v.start, v.ref, Array(aa)).minRep, aai + 1) }
      .filter { case (sv, _) => f(sv) }
      .toArray

    if (splitVariants.isEmpty)
      return Iterator()

    val splitGenotypeBuilders = splitVariants.map { case (sv, _) => new GenotypeBuilder(sv.nAlleles) }
    val splitGenotypeStreamBuilders = splitVariants.map { case (sv, _) => new ArrayBuilder[Genotype]() }

    for (g <- it) {
      val gadsum = Genotype.ad(g).map(gadx => (gadx, gadx.sum))

      // svj corresponds to the ith allele of v
      for (((svj, i), j) <- splitVariants.iterator.zipWithIndex) {
        val gb = splitGenotypeBuilders(j)
        gb.clear()

        if (g == null)
          gb.setMissing()
        else {
          Genotype.gt(g).foreach { ggtx =>
            val gtx = splitGT(ggtx, i)
            gb.setGT(gtx)

            val p = Genotype.gtPair(ggtx)
            if (gtx != p.nNonRefAlleles)
              gb.setFakeRef()
          }

          gadsum.foreach { case (gadx, sum) =>
            // what bcftools does
            // Array(gadx(0), gadx(i))
            gb.setAD(Array(sum - gadx(i), gadx(i)))
          }

          Genotype.dp(g).foreach { dpx => gb.setDP(dpx) }

          if (propagateGQ)
            Genotype.gq(g).foreach { gqx => gb.setGQ(gqx) }

          Genotype.pl(g).foreach { gplx =>
            val plx = gplx.iterator.zipWithIndex
              .map { case (p, k) => (splitGT(k, i), p) }
              .reduceByKeyToArray(3, Int.MaxValue)(_ min _)
            gb.setPX(plx)

            if (!propagateGQ) {
              val gq = Genotype.gqFromPL(plx)
              gb.setGQ(gq)
            }
          }
        }

        splitGenotypeStreamBuilders(j) += gb.result()
      }
    }

    splitVariants.iterator
      .zip(splitGenotypeStreamBuilders.iterator)
      .map { case ((v, ind), gsb) =>
        (v, (insertSplitAnnots(va, ind, true), gsb.result()))
      }
  }

  def splitNumber(str: String): String =
    if (str == "A" || str == "R" || str == "G")
      "."
    else str

  def apply(vds: VariantDataset, propagateGQ: Boolean = false, keepStar: Boolean = false,
    maxShift: Int = 100): VariantDataset = {

    if (vds.wasSplit) {
      warn("called redundant split on an already split VDS")
      return vds
    }

    val (vas2, insertIndex) = vds.vaSignature.insert(TInt32, "aIndex")
    val (vas3, insertSplit) = vas2.insert(TBoolean, "wasSplit")

    val vas4 = vas3.getAsOption[TStruct]("info").map { s =>
      val updatedInfoSignature = TStruct(s.fields.map { f =>
        f.attrs.get("Number").map(splitNumber) match {
          case Some(n) => f.copy(attrs = f.attrs + ("Number" -> n))
          case None => f
        }
      })
      val (newSignature, _) = vas3.insert(updatedInfoSignature, "info")
      newSignature
    }.getOrElse(vas3)

    val partitionerBc = vds.sparkContext.broadcast(vds.rdd.orderedPartitioner)

    val shuffledVariants = vds.rdd.mapPartitionsWithIndex { case (i, it) =>
      it.flatMap { case (v, (va, gs)) =>
        split(v, va, gs,
          propagateGQ = propagateGQ,
          keepStar = keepStar,
          insertSplitAnnots = { (va, index, wasSplit) =>
            insertSplit(insertIndex(va, index), wasSplit)
          },
          f = (v: Variant) => partitionerBc.value.getPartition(v) != i)
      }
    }.orderedRepartitionBy(vds.rdd.orderedPartitioner)

    val localMaxShift = maxShift
    val staticVariants = vds.rdd.mapPartitionsWithIndex { case (i, it) =>
      LocalVariantSortIterator(it.flatMap { case (v, (va, gs)) =>
        split(v, va, gs,
          propagateGQ = propagateGQ,
          keepStar = keepStar,
          insertSplitAnnots = { (va, index, wasSplit) =>
            insertSplit(insertIndex(va, index), wasSplit)
          },
          f = (v: Variant) => partitionerBc.value.getPartition(v) == i)
      }, localMaxShift)
    }

    val newRDD = OrderedRDD.partitionedSortedUnion(staticVariants, shuffledVariants, vds.rdd.orderedPartitioner)

    vds.copy(rdd = newRDD, vaSignature = vas4, wasSplit = true)
  }
}
