package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.utils._

object Contig {
  def gen(rg: ReferenceGenome): Gen[(String, Int)] = Gen.oneOfSeq(rg.lengths.toSeq)
}

object VariantMethods {

  def parse(str: String, rg: ReferenceGenome): (Locus, IndexedSeq[String]) = {
    val elts = str.split(":")
    val size = elts.length
    if (size < 4)
      fatal(s"Invalid string for Variant. Expecting contig:pos:ref:alt1,alt2 -- found '$str'.")

    val contig = elts.take(size - 3).mkString(":")
    (Locus(contig, elts(size - 3).toInt, rg), elts(size - 2) +: elts(size - 1).split(","))
  }

  def locusAllelesToString(locus: Locus, alleles: IndexedSeq[String]): String =
    s"$locus:${alleles(0)}:${alleles.tail.mkString(",")}"

  def minRep(locus: Locus, alleles: IndexedSeq[String]): (Locus, IndexedSeq[String]) = {
    if (alleles.isEmpty)
      fatal(s"min_rep: expect at least one allele, found no alleles")
    if (alleles.contains(null))
      fatal(s"min_rep: found null allele at locus $locus")

    val ref = alleles(0)

    val altAlleles = alleles.tail

    if (ref.length == 1)
      (locus, alleles)
    else if (altAlleles.forall(a => a == "*"))
      (locus, ref.substring(0, 1) +: altAlleles)
    else {
      val alts = altAlleles.filter(a => a != "*")
      require(!alts.contains(ref))

      val min_length = math.min(ref.length, alts.map(x => x.length).min)
      var ne = 0

      while (
        ne < min_length - 1
        && alts.forall(x => ref(ref.length - ne - 1) == x(x.length - ne - 1))
      )
        ne += 1

      var ns = 0
      while (
        ns < min_length - ne - 1
        && alts.forall(x => ref(ns) == x(ns))
      )
        ns += 1

      if (ne + ns == 0)
        (locus, alleles)
      else {
        assert(ns < ref.length - ne && alts.forall(x => ns < x.length - ne))
        (
          Locus(locus.contig, locus.position + ns),
          ref.substring(ns, ref.length - ne) +:
            altAlleles.map(a => if (a == "*") a else a.substring(ns, a.length - ne)).toArray,
        )
      }
    }
  }
}

object VariantSubgen {
  def random(rg: ReferenceGenome): VariantSubgen = VariantSubgen(
    contigGen = Contig.gen(rg),
    nAllelesGen = Gen.frequency((5, Gen.const(2)), (1, Gen.choose(2, 10))),
    refGen = genDNAString,
    altGen = Gen.frequency((10, genDNAString), (1, Gen.const("*"))),
  )

  def plinkCompatible(rg: ReferenceGenome): VariantSubgen = {
    val r = random(rg)
    val compatible = (1 until 22).map(_.toString).toSet
    r.copy(
      contigGen = r.contigGen.filter { case (contig, _) =>
        compatible.contains(contig)
      }
    )
  }

  def biallelic(rg: ReferenceGenome): VariantSubgen = random(rg).copy(nAllelesGen = Gen.const(2))

  def plinkCompatibleBiallelic(rg: ReferenceGenome): VariantSubgen =
    plinkCompatible(rg).copy(nAllelesGen = Gen.const(2))
}

case class VariantSubgen(
  contigGen: Gen[(String, Int)],
  nAllelesGen: Gen[Int],
  refGen: Gen[String],
  altGen: Gen[String],
) {

  def genLocusAlleles: Gen[Annotation] =
    for {
      (contig, length) <- contigGen
      start <- Gen.choose(1, length)
      nAlleles <- nAllelesGen
      ref <- refGen
      altAlleles <- Gen.distinctBuildableOfN[Array](
        nAlleles - 1,
        altGen,
      )
        .filter(!_.contains(ref))
    } yield Annotation(Locus(contig, start), (ref +: altAlleles).toFastSeq)
}
