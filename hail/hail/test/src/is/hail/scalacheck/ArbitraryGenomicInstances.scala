package is.hail.scalacheck

import is.hail.annotations.Annotation
import is.hail.utils.{toRichIterable, triangle, uniqueMaxIndex}
import is.hail.variant._
import is.hail.variant.Genotype.gqFromPL
import is.hail.variant.Sex.Sex

import org.apache.spark.sql.Row
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen._

private[scalacheck] trait ArbitraryGenomicInstances {

  lazy val genBase: Gen[Char] =
    oneOf('A', 'C', 'T', 'G')

  lazy val genDnaString: Gen[String] =
    stringOfN(12, genBase)

  implicit lazy val arbSex: Arbitrary[Sex] =
    oneOf(Sex.Male, Sex.Female)

  def genReferenceGenome(genContigs: Int => Gen[Array[String]]): Gen[ReferenceGenome] =
    for {
      name <- identifier
      n <- posNum[Int]
      contigs <- genContigs(n)
      lengths <- containerOfN[Array, Int](n, choose(1000000, 500000000))
      rg = ReferenceGenome(name, contigs, contigs.zip(lengths).toMap)

      nx <- choose(0, n)
      xContigs <- disjointSetOfN(nx, contigs)

      ny <- choose(0, n - nx)
      yContigs <- disjointSetOfN(ny, contigs.toSet -- xContigs)

      nm <- choose(0, n - nx - ny)
      mtContigs <- disjointSetOfN(nm, contigs.toSet -- (xContigs ++ yContigs))

      par <-
        if (nx + ny > 0)
          distinctContainerOf[Array, (Locus, Locus)] {
            for {
              contig <- oneOf(xContigs ++ yContigs)
              lo <- choose(1, rg.lengths(contig))
              hi <- choose(lo + 1, rg.lengths(contig))
            } yield (Locus(contig, lo, rg), Locus(contig, hi, rg))
          }
        else
          const(Array.empty[(Locus, Locus)])

    } yield rg.copy(
      xContigs = xContigs,
      yContigs = yContigs,
      mtContigs = mtContigs,
      parInput = par,
    )

  lazy val genPlinkCompatibleReferenceGenome: Gen[ReferenceGenome] =
    atMost(21)(_ => genReferenceGenome(n => const(Array.tabulate(n)(i => (i + 1).toString))))

  implicit lazy val arbReferenceGenome: Arbitrary[ReferenceGenome] =
    oneOf(
      genReferenceGenome(n => distinctContainerOfN[Array, String](n, identifier)),
      genPlinkCompatibleReferenceGenome,
    )

  def genContig(rg: ReferenceGenome): Gen[(String, Int)] =
    oneOf(rg.lengths)

  def genLocus(rg: ReferenceGenome): Gen[Locus] =
    for {
      (contig, length) <- genContig(rg)
      position <- Gen.choose(1, length)
    } yield Locus(contig, position, rg)

  implicit lazy val arbLocus: Arbitrary[Locus] =
    arbitrary[ReferenceGenome] flatMap genLocus

  def genCall(
    nAlleles: Int,
    genPloidy: Gen[Int] = choose(0, 2),
    genPhased: Gen[Boolean] = prob(0.5),
  ): Gen[Call] =
    for {
      ploidy <- genPloidy
      phased <- genPhased
      alleles <- containerOfN[Array, Int](ploidy, Gen.choose(0, nAlleles - 1))
    } yield CallN(alleles, phased)

  def genUnphasedDiploid(nAlleles: Int): Gen[Call] =
    genCall(nAlleles, const(2), const(false))

  def genPhasedDiploid(nAlleles: Int): Gen[Call] =
    genCall(nAlleles, const(2), const(true))

  lazy val genCall: Gen[Call] =
    choose(2, 5) flatMap { nAlleles => genCall(nAlleles) }

  def genExtremeGenotype(nAlleles: Int): Gen[Annotation] =
    nullable {
      val m = Int.MaxValue / (nAlleles + 1)
      val nGenotypes = triangle(nAlleles)
      for {
        c <- option(genUnphasedDiploid(nAlleles))
        ad <- option(containerOfN[Array, Int](nAlleles, choose(0, m)))
        dp <- option(choose(0, m))
        gq <- option(choose(0, 10000))
        pl <- oneOf(
          option(containerOfN[Array, Int](nGenotypes, choose(0, m))),
          option(containerOfN[Array, Int](nGenotypes, choose(0, 100))),
        )
      } yield {
        c.foreach(c => pl.foreach(pla => pla(Call.unphasedDiploidGtIndex(c)) = 0))
        pl.foreach { pla =>
          val m = pla.min
          var i = 0
          while (i < pla.length) {
            pla(i) -= m
            i += 1
          }
        }

        Annotation(
          c.orNull,
          ad.map(a => a: IndexedSeq[Int]).orNull,
          dp.map(_ + ad.map(_.sum).getOrElse(0)).orNull,
          gq.orNull,
          pl.map(a => a: IndexedSeq[Int]).orNull,
        )
      }
    }

  def genRealisticGenotype(nAlleles: Int): Gen[Annotation] =
    nullable {
      for {
        alleleFrequencies <- containerOfN[IndexedSeq, Double](nAlleles, choose(1e-6, 1d))

        c <- option {
          val min = alleleFrequencies.min
          val weights = alleleFrequencies.map(f => (f / min).toInt)
          val freq = frequency((weights, (0 until nAlleles).map(const)).zipped.toSeq: _*)
          zip(freq, freq).map { case (gti, gtj) => Call2(gti, gtj) }
        }

        ad <- option(containerOfN[Array, Int](nAlleles, choose(0, 50)))
        dp <- choose(0, 30).map(d => ad.map(o => o.sum + d))
        pl <- option {
          val nGenotypes = triangle(nAlleles)
          containerOfN[Array, Int](nGenotypes, choose(0, 1000)).map { arr =>
            c match {
              case Some(x) =>
                arr(Call.unphasedDiploidGtIndex(x).toInt) = 0
                arr
              case None =>
                val min = arr.min
                arr.map(_ - min)
            }
          }
        }
        gq <- choose(-30, 30).map(i => pl.map(pls => math.max(0, gqFromPL(pls) + i)))
      } yield Annotation(
        c.orNull,
        ad.map(a => a: IndexedSeq[Int]).orNull,
        dp.orNull,
        gq.orNull,
        pl.map(a => a: IndexedSeq[Int]).orNull,
      )
    }

  def genGenericCallAndProbabilitiesGenotype(nAlleles: Int): Gen[Annotation] =
    nullable {
      val nGenotypes = triangle(nAlleles)
      for (gp <- option(partition(nGenotypes, 32768)))
        yield {
          val c =
            gp.flatMap(a => Option(uniqueMaxIndex(a))).map(Call2.fromUnphasedDiploidGtIndex(_))
          Row(
            c.orNull,
            gp.map(gpx => gpx.map(p => p.toDouble / 32768): IndexedSeq[Double]).orNull,
          )
        }
    }

  private[this] def genLocusAlleles(
    genContig: Gen[(String, Int)],
    genNAlleles: Gen[Int],
    genRef: Gen[String],
    genAlt: Gen[String],
  ): Gen[Annotation] =
    for {
      (contig, length) <- genContig
      start <- Gen.choose(1, length)
      nAlleles <- genNAlleles
      altAlleles <- distinctContainerOfN[Array, String](nAlleles - 1, genAlt)
      ref <- genRef
      if !altAlleles.contains(ref)
    } yield Annotation(Locus(contig, start), (ref +: altAlleles).toFastSeq)

  def genRandomLocusAlleles(rg: ReferenceGenome): Gen[Annotation] =
    genLocusAlleles(
      genContig = genContig(rg),
      genNAlleles = frequency(5 -> const(2), 1 -> choose(2, 10)),
      genRef = genDnaString,
      genAlt = frequency(10 -> genDnaString, 1 -> const("*")),
    )

  def genBiallelic(rg: ReferenceGenome): Gen[Annotation] =
    genLocusAlleles(
      genContig = genContig(rg),
      genNAlleles = const(2),
      genRef = genDnaString,
      genAlt = Gen.frequency(10 -> genDnaString, 1 -> const("*")),
    )
}
