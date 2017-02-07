package is.hail.driver

import is.hail.utils._
import is.hail.annotations._
import is.hail.expr.{TBoolean, TInt, TStruct}
import is.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object SplitMulti extends Command {
  def name = "splitmulti"

  def description = "Split multi-allelic sites in the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--propagate-gq", usage = "Propagate GQ instead of computing from PL")
    var propagateGQ: Boolean = false

    @Args4jOption(required = false, name = "--no-compress", usage = "Don't compress genotype streams")
    var noCompress: Boolean = false

    @Args4jOption(required = false, name = "--keep-star-alleles", usage = "Do not filter * alleles")
    var keepStar: Boolean = false

    @Args4jOption(required = false, name = "--max-shift", usage = "Max position shift after min-rep")
    var maxShift: Int = 100
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def splitGT(gt: Int, i: Int): Int = {
    val p = Genotype.gtPair(gt)
    (if (p.j == i) 1 else 0) +
      (if (p.k == i) 1 else 0)
  }

  def split(v: Variant,
    va: Annotation,
    it: Iterable[Genotype],
    propagateGQ: Boolean,
    compress: Boolean,
    isDosage: Boolean,
    keepStar: Boolean,
    insertSplitAnnots: (Annotation, Int, Boolean) => Annotation): Iterator[(Variant, (Annotation, Iterable[Genotype]))] = {

    if (v.isBiallelic)
      return Iterator((v, (insertSplitAnnots(va, 1, false), it)))

    val splitVariants = v.altAlleles.iterator.zipWithIndex
      .filter(keepStar || _._1.alt != "*")
      .map { case (aa, aai) =>
        (Variant(v.contig, v.start, v.ref, Array(aa)).minrep, aai + 1)
      }.toArray

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
        (v, (insertSplitAnnots(va, ind, true), gsb.result()))
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
    val keepStar = options.keepStar

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

    val newVDS = state.vds.copy(
      wasSplit = true,
      vaSignature = vas4,
      rdd = vds.rdd.flatMap { case (v, (va, gs)) =>
        split(v, va, gs,
          propagateGQ = propagateGQ,
          compress = !noCompress,
          keepStar = keepStar,
          isDosage = isDosage,
          insertSplitAnnots = { (va, index, wasSplit) =>
            insertSplit(insertIndex(va, Some(index)), Some(wasSplit))
          })
      }
        .map { case (v, (va, gs)) =>
          (v, (va, gs.toGenotypeStream(v, isDosage, compress = !noCompress): Iterable[Genotype]))
        }
        .smartShuffleAndSort(vds.rdd.orderedPartitioner, options.maxShift))

    state.copy(vds = newVDS)
  }
}
