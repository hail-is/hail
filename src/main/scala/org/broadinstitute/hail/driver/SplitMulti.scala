package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{TBoolean, TInt, TStruct}
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object SplitMulti extends Command {
  def name = "splitmulti"

  def description = "Split multi-allelic sites in the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--propagate-gq", usage = "Propagate GQ instead of computing from PL")
    var propagateGQ: Boolean = false

    @Args4jOption(required = false, name = "--no-compress", usage = "Don't compress genotype streams")
    var noCompress: Boolean = false

  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def splitGT(gt: Int, i: Int): Int = {
    val p = Genotype.gtPair(gt)
    (if (p.j == i) 1 else 0) +
      (if (p.k == i) 1 else 0)
  }

  def minRep(start: Int, ref: String, alt: String): (Int, String, String) = {
    require(ref != alt)
    var newStart = start

    var refe = ref.length
    var alte = alt.length
    while (refe > 1
      && alte > 1
      && ref(refe - 1) == alt(alte - 1)) {
      refe -= 1
      alte -= 1
    }

    var refs = 0
    var alts = 0
    while (ref(refs) == alt(alts)
      && refs + 1 < refe
      && alts + 1 < alte) {
      newStart += 1
      refs += 1
      alts += 1
    }

    assert(refs < refe && alts < alte)
    (newStart, ref.substring(refs, refe), alt.substring(alts, alte))
  }

  def split(v: Variant,
    va: Annotation,
    it: Iterable[Genotype],
    propagateGQ: Boolean,
    compress: Boolean,
    isDosage: Boolean,
    insertSplitAnnots: (Annotation, Int, Boolean) => Annotation): Iterator[(Variant, (Annotation, Iterable[Genotype]))] = {

    if (v.isBiallelic)
      return Iterator((v, (insertSplitAnnots(va, 0, false), it)))

    val splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(_._1.alt != "*")
      .map { case (aa, i) =>
        val (newStart, newRef, newAlt) = minRep(v.start, v.ref, aa.alt)

        (Variant(v.contig, newStart, newRef, newAlt), i + 1)
      }.toArray.sorted

    val splitGenotypeBuilders = splitVariants.map { case (sv, _) => new GenotypeBuilder(sv.nAlleles, isDosage) }
    val splitGenotypeStreamBuilders = splitVariants.map { case (sv, _) => new GenotypeStreamBuilder(sv.nAlleles, isDosage, compress) }

    for (g <- it) {

      val gadsum = g.ad.map(gadx => (gadx, gadx.sum))

      // svj corresponds to the ith allele of v
      for (((svj, i), j) <- splitVariants.iterator.zipWithIndex) {
        val gb = splitGenotypeBuilders(j)
        gb.clear()

        if (!isDosage) {
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

          if (propagateGQ)
            g.gq.foreach { gqx => gb.setGQ(gqx) }

          g.pl.foreach { gplx =>
            val plx = gplx.iterator.zipWithIndex
              .map { case (p, k) => (splitGT(k, i), p) }
              .reduceByKeyToArray(3, Int.MaxValue)(_ min _)
            gb.setPX(plx)

            if (!propagateGQ) {
              val gq = Genotype.gqFromPL(plx)
              gb.setGQ(gq)
            }
          }
        } else {
          val newpx = g.dosage.map { gdx =>
            val dx = gdx.iterator.zipWithIndex
              .map { case (d, k) => (splitGT(k, i), d) }
              .reduceByKeyToArray(3, 0.0)(_ + _)

            val px = Genotype.weightsToLinear(dx)
            gb.setPX(px)
            px
          }

          val newgt = newpx
            .flatMap { px => Genotype.gtFromLinear(px) }
            .getOrElse(-1)

          if (newgt != -1)
            gb.setGT(newgt)

          g.gt.foreach { gtx =>
            val p = Genotype.gtPair(gtx)
            if (newgt != p.nNonRefAlleles && newgt != -1)
              gb.setFakeRef()
          }
        }

        splitGenotypeStreamBuilders(j).write(gb)
      }
    }

    splitVariants.iterator
      .zip(splitGenotypeStreamBuilders.iterator)
      .map { case ((v, ind), gsb) =>
        (v, (insertSplitAnnots(va, ind - 1, true), gsb.result()))
      }
  }

  def splitNumber(str: String): String =
    if (str == "A" || str == "R" || str == "G")
      "."
    else str

  def run(state: State, options: Options): State = {
    val vds = state.vds
    if (vds.wasSplit) {
      warn("called redundant `splitmulti' on an already split VDS")
      return state
    }

    val propagateGQ = options.propagateGQ
    val noCompress = options.noCompress
    val isDosage = vds.isDosage

    val (vas2, insertIndex) = vds.vaSignature.insert(TInt, "aIndex")
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

    val newVDS = state.vds.copy[Genotype](
      wasSplit = true,
      vaSignature = vas4,
      rdd = vds.rdd.mapPartitions[(Variant, (Annotation, Iterable[Genotype]))]({ it =>
        it.flatMap { case (v, (va, gs)) =>
          split(v, va, gs,
            propagateGQ = propagateGQ,
            compress = !noCompress,
            isDosage = isDosage, { (va, index, wasSplit) =>
              insertSplit(insertIndex(va, Some(index)), Some(wasSplit))
            })
        }
      }, preservesPartitioning = true)
        .toOrderedRDD(_.locus))
    state.copy(vds = newVDS)
  }
}
