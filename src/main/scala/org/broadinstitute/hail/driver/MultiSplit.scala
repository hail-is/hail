package org.broadinstitute.hail.driver

import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object MultiSplit extends Command {
  def name = "multisplit"

  def description = "Split multi-allelic sites in the current dataset"

  class Options extends BaseOptions

  def newOptions = new Options

  def splitGT(gt: Int, i: Int): Int = {
    val (j, k) = Genotype.gtPair(gt)
    (if (j == i) 1 else 0) +
      (if (k == i) 1 else 0)
  }

  def split(v: Variant, it: Iterable[Genotype]): Iterator[(Variant, Iterable[Genotype])] = {
    if (v.isBiallelic)
      return Iterator((v, it))

    val splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(_._1.alt != "*")
      .map { case (aa, i) =>
        (Variant(v.contig, v.start, v.ref, aa.alt), i + 1)
      }.toArray

    // FIXME compress?
    val splitGenotypeBs = splitVariants.map { case (sv, _) => new GenotypeStreamBuilder(sv, true) }

    for (g <- it) {

      val gadsum = g.ad.map(gadx => (gadx, gadx.sum))

      // svj corresponds to the ith allele of v
      for (((svj, i), j) <- splitVariants.iterator.zipWithIndex) {

        val gt = g.gt.map(ggtx => splitGT(ggtx, i))
        val ad: Option[IndexedSeq[Int]] = gadsum.map { case (gadx, sum) =>
          // what bcftools does
          // Array(gadx(0), gadx(i))
          Array(sum - gadx(i), gadx(i))
        }

        val pl: Option[IndexedSeq[Int]] = g.pl.map { gplx =>
          val plx = gplx.iterator.zipWithIndex
            .map { case (p, k) => (splitGT(k, i), p) }
            .reduceByKeyToArray(3, Int.MaxValue)(_ min _)
          assert(!plx.contains(Int.MaxValue))
          plx
        }

        // FIXME
        splitGenotypeBs(j) += Genotype(gt, ad, g.dp, pl)
      }
    }

    splitVariants.iterator.map(_._1).zip(splitGenotypeBs.iterator.map(_.result()))
  }

  def run(state: State, options: Options): State = {
    val newVDS = state.vds.copy[Genotype](rdd =
      state.vds.rdd.flatMap[(Variant, Iterable[Genotype])]((split _).tupled))
    state.copy(vds = newVDS)
  }
}
