package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.utils._

object Contig {
  def gen(rg: ReferenceGenome): Gen[(String, Int)] = Gen.oneOfSeq(rg.lengths.toSeq)
}

object VariantMethods {
  def locusAllelesToString(locus: Locus, alleles: IndexedSeq[String]): String =
    s"$locus:${ alleles(0) }:${ alleles.tail.mkString(",") }"
}

object VariantSubgen {
  def random(rg: ReferenceGenome): VariantSubgen = VariantSubgen(
    contigGen = Contig.gen(rg),
    nAllelesGen = Gen.frequency((5, Gen.const(2)), (1, Gen.choose(2, 10))),
    refGen = genDNAString,
    altGen = Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))))

  def plinkCompatible(rg: ReferenceGenome): VariantSubgen = {
    val r = random(rg)
    val compatible = (1 until 22).map(_.toString).toSet
    r.copy(
      contigGen = r.contigGen.filter { case (contig, len) =>
        compatible.contains(contig)
      })
  }

  def biallelic(rg: ReferenceGenome): VariantSubgen = random(rg).copy(nAllelesGen = Gen.const(2))

  def plinkCompatibleBiallelic(rg: ReferenceGenome): VariantSubgen =
    plinkCompatible(rg).copy(nAllelesGen = Gen.const(2))
}

case class VariantSubgen(
  contigGen: Gen[(String, Int)],
  nAllelesGen: Gen[Int],
  refGen: Gen[String],
  altGen: Gen[String]) {

  def genLocusAlleles: Gen[Annotation] =
    for {
      (contig, length) <- contigGen
      start <- Gen.choose(1, length)
      nAlleles <- nAllelesGen
      ref <- refGen
      altAlleles <- Gen.distinctBuildableOfN[Array](
        nAlleles - 1,
        altGen)
        .filter(!_.contains(ref))
    } yield
      Annotation(Locus(contig, start), (ref +: altAlleles).toFastIndexedSeq)
}
