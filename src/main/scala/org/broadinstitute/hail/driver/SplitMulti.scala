package org.broadinstitute.hail.driver

import org.apache.spark.sql.Row
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._
import scala.collection.mutable

object SplitMulti extends Command {
  def name = "splitmulti"

  def description = "Split multi-allelic sites in the current dataset"

  override def supportsMultiallelic = true

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--propagate-gq", usage = "Propagate GQ instead of computing from PL")
    var propagateGQ: Boolean = false
  }

  def newOptions = new Options

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
    splitF: (Annotation, Int) => Annotation,
    insertF: Inserter): Iterator[(Variant, Annotation, Iterable[Genotype])] = {
    if (v.isBiallelic)
      return Iterator((v, insertF(va, Some(false)), it))

    val splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(_._1.alt != "*")
      .map { case (aa, i) =>
        val (newStart, newRef, newAlt) = minRep(v.start, v.ref, aa.alt)

        (Variant(v.contig, newStart, newRef, newAlt), i + 1)
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

        if (propagateGQ)
          g.gq.foreach { gqx => gb.setGQ(gqx) }

        g.pl.foreach { gplx =>
          val plx = gplx.iterator.zipWithIndex
            .map { case (p, k) => (splitGT(k, i), p) }
            .reduceByKeyToArray(3, Int.MaxValue)(_ min _)
          gb.setPL(plx)

          if (!propagateGQ) {
            val gq = Genotype.gqFromPL(plx)
            gb.setGQ(gq)
          }
        }

        splitGenotypeStreamBuilders(j).write(gb)
      }
    }

    splitVariants.iterator
      .zip(splitGenotypeStreamBuilders.iterator)
      .map { case ((v, ind), gsb) =>
        (v, insertF(splitF(va, ind), Some(true)), gsb.result())
      }
  }


  def splitAnnotations(base: Signature): (Any, Int) => Any = {
    base match {
      case struct: StructSignature =>
        struct.m.get("info") match {
          case Some((index: Int, infoSigs: StructSignature)) =>
            val functions: Array[((Any, Int) => Any)] = infoSigs.m
              .toArray
              .sortBy { case (key, (i, sig)) => i }
              .map { case (key, (i, sig)) => sig match {
                case vcfSig: VCFSignature if vcfSig.number == "A" =>
                  (a: Any, ind: Int) => if (a == null)
                    null
                  else
                    Array(a.asInstanceOf[mutable.WrappedArray[_]]
                      .apply(i))
                case vcfSig: VCFSignature if vcfSig.number == "R" =>
                  (a: Any, ind: Int) =>
                    if (a == null)
                      null
                    else {
                      val arr = a.asInstanceOf[mutable.WrappedArray[_]]
                      Array(arr(0), arr(ind))
                    }
                case vcfSig: VCFSignature if vcfSig.number == "G" =>
                  (a: Any, ind: Int) =>
                    if (a == null)
                      null
                    else {
                      val arr = a.asInstanceOf[mutable.WrappedArray[_]]
                      Array(arr(0), arr(triangle(ind + 1) + 1), arr(triangle(ind + 2) - 1))
                    }
                case _ => (a: Any, ind: Int) => a
              }
              }
            (ad: Any, alleleIndex: Int) =>
              val adArr = ad.asInstanceOf[Row].toSeq.toArray
              val infoR = adArr(index).asInstanceOf[Row]
              adArr(index) = Row.fromSeq(
                functions.zipWithIndex
                  .map { case (f, i) =>
                    f(infoR.get(i), alleleIndex)
                  })
              Row.fromSeq(adArr)
          case _ => (ad, index) => ad
        }
      case sig => (a, i) => a
    }
  }

  def run(state: State, options: Options): State = {
    val localPropagateGQ = options.propagateGQ
    val splitF = splitAnnotations(state.vds.metadata.variantAnnotationSignatures)
    val (newSigs, f) = state.vds.metadata.variantAnnotationSignatures.insert(List("wasSplit"),
      SimpleSignature(expr.TBoolean))
    val newVDS = state.vds.copy[Genotype](
      metadata = state.vds.metadata
        .copy(wasSplit = true, variantAnnotationSignatures = newSigs),
      rdd = state.vds.rdd.flatMap[(Variant, Annotation, Iterable[Genotype])] { case (v, va, it) =>
        split(v, va, it, localPropagateGQ, splitF, f)
      })
    state.copy(vds = newVDS)
  }
}
