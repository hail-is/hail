package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object MultiSplit extends Command {
  def name = "multisplit"

  def description = "Split multi-allelic sites in the current dataset"

  class Options extends BaseOptions

  def newOptions = new Options

  def splitGT(gt: Int, i: Int): Int = {
    val p = Genotype.gtPair(gt)
    (if (p.j == i) 1 else 0) +
      (if (p.k == i) 1 else 0)
  }

  def split(v: Variant, va: Annotations, it: Iterable[Genotype]): Iterator[(Variant, Annotations, Iterable[Genotype])] = {
    if (v.isBiallelic)
      return Iterator((v, va + ("wasSplit", false), it))

    val splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(_._1.alt != "*")
      .map { case (aa, i) =>
        (Variant(v.contig, v.start, v.ref, aa.alt), i + 1)
      }.toArray

    val splitGenotypeBuilders = splitVariants.map { case (sv, _) => new GenotypeBuilder(sv) }
    val splitGenotypeStreamBuilders = splitVariants.map { case (sv, _) => new GenotypeStreamBuilder(sv, true) }

    for (g <- it) {

      val gadsum = g.ad.map(gadx => (gadx, gadx.sum))

      // svj corresponds to the ith allele of v
      for (((svj, i), j) <- splitVariants.iterator.zipWithIndex) {
        val gb = splitGenotypeBuilders(j)

        gb.clear()
        g.gt.foreach { ggtx =>
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

        g.dp.foreach { dpx => gb.setDP(dpx) }

        g.pl.foreach { gplx =>
          val plx = gplx.iterator.zipWithIndex
            .map { case (p, k) => (splitGT(k, i), p) }
            .reduceByKeyToArray(3, Int.MaxValue)(_ min _)
          assert(!plx.contains(Int.MaxValue))
          gb.setPL(plx)
        }


        splitGenotypeStreamBuilders(j).write(gb)
      }
    }

    splitVariants.iterator.map(_._1)
      .zip(splitGenotypeStreamBuilders.iterator)
      .map { case (v, gsb) =>
        (v, va + ("wasSplit", true), gsb.result())
      }
  }

  def run(state: State, options: Options): State = {
    val newVDS = state.vds.copy[Genotype](
      metadata = state.vds.metadata.addVariantAnnotationSignatures("wasSplit", "Boolean"),
      rdd = state.vds.rdd.flatMap[(Variant, Annotations, Iterable[Genotype])]((split _).tupled))
    state.copy(vds = newVDS)
  }
}
