package is.hail.types.physical

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager
import is.hail.check.Arbitrary.arbitrary
import is.hail.check.{Arbitrary, Gen}
import is.hail.types.virtual.{
  Field, TArray, TBoolean, TCall, TDict, TFloat32, TFloat64, TInt32, TInt64, TInterval, TLocus,
  TSet, TString, TStruct, TTuple, Type,
}
import is.hail.utils.{Interval, genDNAString, triangle, uniqueMaxIndex}
import is.hail.variant.Call.{
  alleleByIndex, allelePair, alleles, isPhased, ploidy, unphasedDiploidGtIndex,
}
import is.hail.variant.Genotype.gqFromPL
import is.hail.variant.{AllelePair, Call, Call2, CallN, Locus, ReferenceGenome}
import org.apache.spark.sql.Row

import scala.annotation.switch

trait GenInstances {

  def genScalar(required: Boolean): Gen[PType] =
    Gen.oneOf(
      PBoolean(required),
      PInt32(required),
      PInt64(required),
      PFloat32(required),
      PFloat64(required),
      PCanonicalString(required),
      PCanonicalCall(required),
    )

  val genOptionalScalar: Gen[PType] = genScalar(false)

  val genRequiredScalar: Gen[PType] = genScalar(true)

  def genComplexType(required: Boolean): Gen[PType] = {
    val rgDependents = ReferenceGenome.hailReferences.toArray.map(PCanonicalLocus(_, required))
    val others = Array(PCanonicalCall(required))
    Gen.oneOfSeq(rgDependents ++ others)
  }

  def genFields(required: Boolean, genFieldType: Gen[PType]): Gen[Array[PField]] =
    Gen.buildableOf[Array](
      Gen.zip(Gen.identifier, genFieldType)
    )
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields =>
        fields
          .iterator
          .zipWithIndex
          .map { case ((k, t), i) => PField(k, t, i) }
          .toArray
      )

  def preGenStruct(required: Boolean, genFieldType: Gen[PType]): Gen[PStruct] =
    for (fields <- genFields(required, genFieldType)) yield PCanonicalStruct(fields, required)

  def preGenTuple(required: Boolean, genFieldType: Gen[PType]): Gen[PTuple] =
    for (fields <- genFields(required, genFieldType))
      yield PCanonicalTuple(required, fields.map(_.typ): _*)

  private val defaultRequiredGenRatio = 0.2

  def genStruct: Gen[PStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(preGenStruct(_, genArb))

  val genOptionalStruct: Gen[PType] = preGenStruct(required = false, genArb)

  val genRequiredStruct: Gen[PType] = preGenStruct(required = true, genArb)

  val genInsertableStruct: Gen[PStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(required =>
    if (required)
      preGenStruct(required = true, genArb)
    else
      preGenStruct(required = false, genOptional)
  )

  def genSized(size: Int, required: Boolean, genPStruct: Gen[PStruct]): Gen[PType] =
    if (size < 1)
      Gen.const(PCanonicalStruct.empty(required))
    else if (size < 2)
      genScalar(required)
    else {
      Gen.frequency(
        (4, genScalar(required)),
        (1, genComplexType(required)),
        (
          1,
          genArb.map {
            PCanonicalArray(_)
          },
        ),
        (
          1,
          genArb.map {
            PCanonicalSet(_)
          },
        ),
        (
          1,
          genArb.map {
            PCanonicalInterval(_)
          },
        ),
        (1, preGenTuple(required, genArb)),
        (1, Gen.zip(genRequired, genArb).map { case (k, v) => PCanonicalDict(k, v) }),
        (1, genPStruct.resize(size)),
      )
    }

  def preGenArb(required: Boolean, genStruct: Gen[PStruct] = genStruct): Gen[PType] =
    Gen.sized(genSized(_, required, genStruct))

  def genArb: Gen[PType] = Gen.coin(0.2).flatMap(preGenArb(_))

  val genOptional: Gen[PType] = preGenArb(required = false)

  val genRequired: Gen[PType] = preGenArb(required = true)

  val genInsertable: Gen[PStruct] = genInsertableStruct

  implicit def arbType: Arbitrary[PType] = Arbitrary(genArb)

  implicit val arbPArray: Arbitrary[PArray] =
    Gen {
      for {
        elem <- arbitrary[PType]
        required <- arbitrary[Boolean]
      } yield PCanonicalArray(elem, required)

      def genNonmissingValue(sm: HailStateManager): Gen[IndexedSeq[Annotation]] =
        Gen.buildableOf[Array](elementType.genValue(sm)).map(x => x: IndexedSeq[Annotation])
    }

  object Contig {
    def gen(rg: ReferenceGenome): Gen[(String, Int)] = Gen.oneOfSeq(rg.lengths.toSeq)
  }

  object Locus {
    def gen(rg: ReferenceGenome): Gen[Locus] =
      for {
        (contig, length) <- Contig.gen(rg)
        pos <- Gen.choose(1, length)
      } yield Locus(contig, pos)
  }

  object Call {
    def check(c: Call, nAlleles: Int): Unit = {
      (ploidy(c): @switch) match {
        case 0 =>
        case 1 =>
          val a = alleleByIndex(c, 0)
          assert(a >= 0 && a < nAlleles)
        case 2 =>
          val nGenotypes = triangle(nAlleles)
          val udtn =
            if (isPhased(c)) {
              val p = allelePair(c)
              unphasedDiploidGtIndex(Call2(AllelePair.j(p), AllelePair.k(p)))
            } else
              unphasedDiploidGtIndex(c)
          assert(
            udtn < nGenotypes,
            s"Invalid call found '${c.toString}' for number of alleles equal to '$nAlleles'.",
          )
        case _ =>
          alleles(c).foreach(a => assert(a >= 0 && a < nAlleles))
      }
    }

    def gen(
      nAlleles: Int,
      ploidyGen: Gen[Int] = Gen.choose(0, 2),
      phasedGen: Gen[Boolean] = Gen.nextCoin(0.5),
    ): Gen[Call] = for {
      ploidy <- ploidyGen
      phased <- phasedGen
      alleles <- Gen.buildableOfN[Array](ploidy, Gen.choose(0, nAlleles - 1))
    } yield {
      val c = CallN(alleles, phased)
      check(c, nAlleles)
      c
    }

    def genUnphasedDiploid(nAlleles: Int): Gen[Call] = gen(nAlleles, Gen.const(2), Gen.const(false))

    def genPhasedDiploid(nAlleles: Int): Gen[Call] = gen(nAlleles, Gen.const(2), Gen.const(true))

    def genNonmissingValue: Gen[Call] = for {
      nAlleles <- Gen.choose(2, 5)
      c <- gen(nAlleles)
    } yield {
      check(c, nAlleles)
      c
    }
  }

  object Genotype {
    def genExtremeNonmissing(nAlleles: Int): Gen[Annotation] = {
      val m = Int.MaxValue / (nAlleles + 1)
      val nGenotypes = triangle(nAlleles)
      val gg = for {
        c: Option[Call] <- Gen.option(Call.genUnphasedDiploid(nAlleles))
        ad <- Gen.option(Gen.buildableOfN[Array](nAlleles, Gen.choose(0, m)))
        dp <- Gen.option(Gen.choose(0, m))
        gq <- Gen.option(Gen.choose(0, 10000))
        pl <- Gen.oneOfGen(
          Gen.option(Gen.buildableOfN[Array](nGenotypes, Gen.choose(0, m))),
          Gen.option(Gen.buildableOfN[Array](nGenotypes, Gen.choose(0, 100))),
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
        val g = Annotation(
          c.orNull,
          ad.map(a => a: IndexedSeq[Int]).orNull,
          dp.map(_ + ad.map(_.sum).getOrElse(0)).orNull,
          gq.orNull,
          pl.map(a => a: IndexedSeq[Int]).orNull,
        )
        g
      }
      gg
    }

    def genExtreme(nAlleles: Int): Gen[Annotation] =
      Gen.frequency(
        (100, genExtremeNonmissing(nAlleles)),
        (1, Gen.const(null)),
      )

    def genRealisticNonmissing(nAlleles: Int): Gen[Annotation] = {
      val nGenotypes = triangle(nAlleles)
      val gg = for {
        callRate <- Gen.choose(0d, 1d)
        alleleFrequencies <-
          Gen.buildableOfN[Array](nAlleles, Gen.choose(1e-6, 1d)) // avoid divison by 0
            .map { rawWeights =>
              val sum = rawWeights.sum
              rawWeights.map(_ / sum)
            }
        c <- Gen.option(
          Gen.zip(
            Gen.chooseWithWeights(alleleFrequencies),
            Gen.chooseWithWeights(alleleFrequencies),
          )
            .map { case (gti, gtj) => Call2(gti, gtj) },
          callRate,
        )
        ad <- Gen.option(Gen.buildableOfN[Array](nAlleles, Gen.choose(0, 50)))
        dp <- Gen.choose(0, 30).map(d => ad.map(o => o.sum + d))
        pl <- Gen.option(Gen.buildableOfN[Array](nGenotypes, Gen.choose(0, 1000)).map { arr =>
          c match {
            case Some(x) =>
              arr(Call.unphasedDiploidGtIndex(x)) = 0
              arr
            case None =>
              val min = arr.min
              arr.map(_ - min)
          }
        })
        gq <- Gen.choose(-30, 30).map(i => pl.map(pls => math.max(0, gqFromPL(pls) + i)))
      } yield Annotation(c.orNull, ad.map(a => a: IndexedSeq[Int]).orNull, dp.orNull, gq.orNull, pl.map(a => a: IndexedSeq[Int]).orNull)
      gg
    }

    def genRealistic(nAlleles: Int): Gen[Annotation] =
      Gen.frequency(
        (100, genRealisticNonmissing(nAlleles)),
        (1, Gen.const(null)),
      )

    def genGenericCallAndProbabilitiesGenotype(nAlleles: Int): Gen[Annotation] = {
      val nGenotypes = triangle(nAlleles)
      val gg = for (gp <- Gen.option(Gen.partition(nGenotypes, 32768))) yield {
        val c = gp.flatMap(a => Option(uniqueMaxIndex(a))).map(Call2.fromUnphasedDiploidGtIndex(_))
        Row(
          c.orNull,
          gp.map(gpx => gpx.map(p => p.toDouble / 32768): IndexedSeq[Double]).orNull,
        )
      }
      Gen.frequency(
        (100, gg),
        (1, Gen.const(null)),
      )
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

  object ReferenceGenome {
    def gen: Gen[ReferenceGenome] =
      for {
        name <- Gen.identifier.filter(!ReferenceGenome.hailReferences.contains(_))
        nContigs <- Gen.choose(3, 10)
        contigs <- Gen.distinctBuildableOfN[Array](nContigs, Gen.identifier)
        lengths <- Gen.buildableOfN[Array](nContigs, Gen.choose(1000000, 500000000))
        contigsIndex = contigs.zip(lengths).toMap
        xContig <- Gen.oneOfSeq(contigs)
        parXA <- Gen.choose(0, contigsIndex(xContig))
        parXB <- Gen.choose(0, contigsIndex(xContig))
        yContig <- Gen.oneOfSeq(contigs) if yContig != xContig
        parYA <- Gen.choose(0, contigsIndex(yContig))
        parYB <- Gen.choose(0, contigsIndex(yContig))
        mtContig <- Gen.oneOfSeq(contigs) if mtContig != xContig && mtContig != yContig
      } yield ReferenceGenome(
        name,
        contigs,
        contigs.zip(lengths).toMap,
        Set(xContig),
        Set(yContig),
        Set(mtContig),
        Array(
          (Locus(xContig, math.min(parXA, parXB)), Locus(xContig, math.max(parXA, parXB))),
          (Locus(yContig, math.min(parYA, parYB)), Locus(yContig, math.max(parYA, parYB))),
        ),
      )
  }

  object Type {
    def genScalar(): Gen[Type] =
      Gen.oneOf(TBoolean, TInt32, TInt64, TFloat32,
        TFloat64, TString, TCall)

    def genComplexType(): Gen[Type] = {
      val rgDependents = ReferenceGenome.hailReferences.toArray.map(TLocus(_))
      val others = Array(TCall)
      Gen.oneOfSeq(rgDependents ++ others)
    }

    def genFields(genFieldType: Gen[Type]): Gen[Array[Field]] = {
      Gen.buildableOf[Array](
        Gen.zip(Gen.identifier, genFieldType)
      )
        .filter(fields => fields.map(_._1).areDistinct())
        .map(fields =>
          fields
            .iterator
            .zipWithIndex
            .map { case ((k, t), i) => Field(k, t, i) }
            .toArray
        )
    }

    def preGenStruct(genFieldType: Gen[Type]): Gen[TStruct] =
      for (fields <- genFields(genFieldType)) yield TStruct(fields)

    def preGenTuple(genFieldType: Gen[Type]): Gen[TTuple] =
      for (fields <- genFields(genFieldType)) yield TTuple(fields.map(_.typ): _*)

    private val defaultRequiredGenRatio = 0.2

    def genStruct: Gen[TStruct] =
      Gen.coin(defaultRequiredGenRatio).flatMap(c => preGenStruct(genArb))

    def genSized(size: Int, genTStruct: Gen[TStruct]): Gen[Type] =
      if (size < 1)
        Gen.const(TStruct.empty)
      else if (size < 2)
        genScalar()
      else {
        Gen.frequency(
          (4, genScalar()),
          (1, genComplexType()),
          (
            1,
            genArb.map {
              TArray(_)
            },
          ),
          (
            1,
            genArb.map {
              TSet(_)
            },
          ),
          (
            1,
            genArb.map {
              TInterval(_)
            },
          ),
          (1, preGenTuple(genArb)),
          (1, Gen.zip(genRequired, genArb).map { case (k, v) => TDict(k, v) }),
          (1, genTStruct.resize(size)),
        )
      }

    def preGenArb(genStruct: Gen[TStruct] = genStruct): Gen[Type] =
      Gen.sized(genSized(_, genStruct))

    def genArb: Gen[Type] = preGenArb()

    val genOptional: Gen[Type] = preGenArb()

    val genRequired: Gen[Type] = preGenArb()

    def genWithValue(sm: HailStateManager): Gen[(Type, Annotation)] = for {
      s <- Gen.size
      // prefer smaller type and bigger values
      fraction <- Gen.choose(0.1, 0.3)
      x = (fraction * s).toInt
      y = s - x
      t <- Type.genStruct.resize(x)
      v <- t.genValue(sm).resize(y)
    } yield (t, v)

    implicit def arbType: Arbitrary[Type] =
      Arbitrary(genArb)
  }

  object PBaseStruct {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      if (types.isEmpty) {
        Gen.const(Annotation.empty)
      } else
        Gen.uniformSequence(types.map(t => t.genValue(sm))).map(a => Annotation(a: _*))
  }

  object PDict {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Gen.buildableOf2[Map](Gen.zip(keyType.genValue(sm), valueType.genValue(sm)))
  }

  object PSet {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Gen.buildableOf[Set](elementType.genValue(sm))
  }

  object TInterval {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Interval.gen(pointType.ordering(sm), pointType.genValue(sm))
  }

  trait Type {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation]

    def genValue(sm: HailStateManager): Gen[Annotation] =
      Gen.nextCoin(0.05).flatMap(isEmpty =>
        if (isEmpty) Gen.const(null) else genNonmissingValue(sm)
      )
  }

  trait TRNGState {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???
  }

  trait TUnion {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???
  }

  trait TVariable {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???
  }

  trait TVoid {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???
  }

  trait TArray {
    def genNonmissingValue(sm: HailStateManager): Gen[IndexedSeq[Annotation]] =
      Gen.buildableOf[Array](elementType.genValue(sm)).map(x => x: IndexedSeq[Annotation])
  }

  trait TBaseStruct {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      if (types.isEmpty) {
        Gen.const(Annotation.empty)
      } else
        Gen.size.flatMap(fuel =>
          if (types.length > fuel)
            Gen.uniformSequence(types.map(t => Gen.const(null))).map(a => Annotation(a: _*))
          else
            Gen.uniformSequence(types.map(t => t.genValue(sm))).map(a => Annotation(a: _*))
        )
  }

  trait TBinary {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Gen.buildableOf(arbitrary[Byte])
  }

  trait TBoolean {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      arbitrary[Boolean]
  }

  trait TCall {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Call.genNonmissingValue
  }

  trait Foo {
    def genValue(sm: HailStateManager): Gen[Annotation] =
      if (required) genNonmissingValue(sm)
      else Gen.nextCoin(0.05).flatMap(isEmpty =>
        if (isEmpty) Gen.const(null) else genNonmissingValue(sm)
      )

    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      virtualType.genNonmissingValue(sm)
  }

  trait TDict {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Gen.buildableOf2[Map](Gen.zip(keyType.genValue(sm), valueType.genValue(sm)))
  }

  trait TFloat32 {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      arbitrary[Float]
  }

  trait TFloat64 {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      arbitrary[Double]
  }

  trait TInt32 {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      arbitrary[Int]
  }

  trait TInt64 {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      arbitrary[Long]
  }

  trait TLocus {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Locus.gen(sm.referenceGenomes(rgName))
  }

  trait TNDArray {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???
  }

  trait TSet {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      Gen.buildableOf[Set](elementType.genValue(sm))
  }

  trait TStream {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
      throw new UnsupportedOperationException("Streams don't have associated annotations.")
  }

  trait TString {
    def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = arbitrary[String]
  }

  trait Interval {
    def gen[P](pord: ExtendedOrdering, pgen: Gen[P]): Gen[Interval] =
      Gen.zip(pgen, pgen, Gen.coin(), Gen.coin())
        .filter { case (x, y, s, e) => pord.compare(x, y) != 0 || (s && e) }
        .map { case (x, y, s, e) =>
          if (pord.compare(x, y) < 0)
            Interval(x, y, s, e)
          else
            Interval(y, x, s, e)
        }
  }

  object utils {
    def genBase: Gen[Char] = Gen.oneOf('A', 'C', 'T', 'G')

    def genDNAString: Gen[String] = Gen.stringOf(genBase)
      .resize(12)
      .filter(s => !s.isEmpty)
  }

}
