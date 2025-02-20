package is.hail.check

import is.hail.check.GenMore.distinctContainerOfN
import is.hail.variant.Sex.Sex
import is.hail.variant.{Locus, ReferenceGenome, Sex}
import org.scalacheck.Gen.{choose, containerOf, identifier, oneOf, pick, size}
import org.scalacheck.{Arbitrary, Gen}

trait GenVariantInstances {

  def genSex: Gen[Sex] =
    oneOf(Sex.Male, Sex.Female)

  def genContig(rg: ReferenceGenome): Gen[(String, Int)] =
    oneOf(rg.lengths)

  def genLocus(rg: ReferenceGenome): Gen[Locus] =
    for {
      (contig, length) <- genContig(rg)
      position <- Gen.choose(1, length)
    } yield Locus(contig, position, rg)

  def genReferenceGenome: Gen[ReferenceGenome] =
    for {
      name <- identifier
      n <- size
      contigs <- distinctContainerOfN[Array, String](n, Gen.identifier)
      lengths <- distinctContainerOfN[Array, Int](n, choose(1000000, 500000000))
      rg = ReferenceGenome(name, contigs, contigs.zip(lengths).toMap)

      nX <- choose(0, n)
      xContigs <- pick(nX, contigs).map(_.toSet)

      nY <- choose(0, n - nX)
      yContigs <- pick(nY, contigs.toSet - xContigs).map(_.toSet)

      nM <- choose(0, n - nX - nY)
      mtContigs <- pick(nM, contigs.toSet - (xContigs ++ yContigs)).map(_.toSet)

      par <- containerOf[Array, (Locus, Locus)] {
        for {
          contig <- oneOf(xContigs + yContigs)
          lo <- choose(1, rg.lengths(contig))
          hi <- choose(lo + 1, rg.lengths(contig))
        } yield (Locus(contig, lo, rg), Locus(contig, hi, rg))
      }

    } yield rg.copy(
      xContigs = xContigs,
      yContigs = yContigs,
      mtContigs = mtContigs,
      parInput = par,
    )

  implicit lazy val arbReferenceGenome: Arbitrary[ReferenceGenome] =
    Arbitrary(genReferenceGenome)

  implicit lazy val arbLocus: Arbitrary[Locus] =
    Arbitrary(genReferenceGenome.flatMap(genLocus))
}
