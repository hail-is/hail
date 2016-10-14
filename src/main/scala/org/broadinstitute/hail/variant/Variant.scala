package org.broadinstitute.hail.variant

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.check.{Arbitrary, Gen}
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.sparkextras.OrderedKey
import org.json4s._

import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

object Contig {
  val standardContigs = (1 to 23).map(_.toString) ++ IndexedSeq("X", "Y", "MT")
  val standardContigIdx = standardContigs.zipWithIndex.toMap

  def compare(lhs: String, rhs: String): Int = {
    (standardContigIdx.get(lhs), standardContigIdx.get(rhs)) match {
      case (Some(i), Some(j)) => i.compare(j)
      case (Some(_), None) => -1
      case (None, Some(_)) => 1
      case (None, None) => lhs.compare(rhs)
    }
  }
}

object AltAlleleType extends Enumeration {
  type AltAlleleType = Value
  // FIXME add "*"
  val SNP, MNP, Insertion, Deletion, Complex = Value
}

object CopyState extends Enumeration {
  type CopyState = Value
  val Auto, HemiX, HemiY = Value
}

object AltAllele {
  val schema: StructType = StructType(Array(
    StructField("ref", StringType, nullable = false),
    StructField("alt", StringType, nullable = false)))

  val t: TStruct = TStruct("ref" -> TString,
    "alt" -> TString)

  def fromRow(r: Row): AltAllele =
    AltAllele(r.getString(0), r.getString(1))

  def gen(ref: String): Gen[AltAllele] =
    for (alt <- Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))) if alt != ref)
      yield AltAllele(ref, alt)

  def gen: Gen[AltAllele] =
    for (ref <- genDNAString;
      alt <- genDNAString if alt != ref)
      yield AltAllele(ref, alt)
}

case class AltAllele(ref: String,
  alt: String) {
  require(ref != alt, "ref was equal to alt")
  require(!ref.isEmpty, "ref was an empty string")
  require(!alt.isEmpty, "alt was an empty string")

  import AltAlleleType._

  def altAlleleType: AltAlleleType = {
    if (ref.length == 1 && alt.length == 1)
      SNP
    else if (ref.length == alt.length)
      if (nMismatch == 1)
        SNP
      else
        MNP
    else if (alt.startsWith(ref))
      Insertion
    else if (ref.startsWith(alt))
      Deletion
    else
      Complex
  }

  def isSNP: Boolean = (ref.length == 1 && alt.length == 1) ||
    (ref.length == alt.length && nMismatch == 1)

  def isMNP: Boolean = ref.length > 1 &&
    ref.length == alt.length &&
    nMismatch > 1

  def isInsertion: Boolean = ref.length < alt.length && alt.startsWith(ref)

  def isDeletion: Boolean = alt.length < ref.length && ref.startsWith(alt)

  def isIndel: Boolean = isInsertion || isDeletion

  def isComplex: Boolean = ref.length != alt.length && !isInsertion && !isDeletion

  def isTransition: Boolean = isSNP && {
    val (refChar, altChar) = strippedSNP
    (refChar == 'A' && altChar == 'G') || (refChar == 'G' && altChar == 'A') ||
      (refChar == 'C' && altChar == 'T') || (refChar == 'T' && altChar == 'C')
  }

  def isTransversion: Boolean = isSNP && !isTransition

  def nMismatch: Int = {
    require(ref.length == alt.length, s"invalid nMismatch call on ref `${ref}' and alt `${alt}'")
    (ref, alt).zipped.map((a, b) => if (a == b) 0 else 1).sum
  }

  def strippedSNP: (Char, Char) = {
    require(isSNP, "called strippedSNP on non-SNP")
    (ref, alt).zipped.dropWhile { case (a, b) => a == b }.head
  }

  def toJSON: JValue = JObject(
    ("ref", JString(ref)),
    ("alt", JString(alt))
  )

  override def toString: String = s"$ref/$alt"
}

object Variant {
  def apply(contig: String,
    start: Int,
    ref: String,
    alt: String): Variant = Variant(contig, start, ref, Array(AltAllele(ref, alt)))

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String]): Variant =
    Variant(contig, start, ref, alts.map(alt => AltAllele(ref, alt)))

  def nGenotypes(nAlleles: Int): Int = {
    require(nAlleles > 0, s"called nGenotypes with invalid number of alternates: $nAlleles")
    nAlleles * (nAlleles + 1) / 2
  }

  def gen: Gen[Variant] = VariantSubgen.random.gen

  implicit def arbVariant: Arbitrary[Variant] = Arbitrary(gen)

  val schema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("start", IntegerType, nullable = false),
      StructField("ref", StringType, nullable = false),
      StructField("altAlleles", ArrayType(AltAllele.schema, containsNull = false),
        nullable = false)))

  val t: TStruct =
    TStruct("contig" -> TString,
      "start" -> TInt,
      "ref" -> TString,
      "altAlleles" -> TArray(AltAllele.t))

  def fromRow(r: Row) =
    Variant(r.getAs[String](0),
      r.getAs[Int](1),
      r.getAs[String](2),
      r.getSeq[Row](3)
        .map(s => AltAllele.fromRow(s))
        .toArray)

  implicit def orderedKey: OrderedKey[Locus, Variant] =
    new OrderedKey[Locus, Variant] {
      def project(key: Variant): Locus = key.locus

      def kOrd: Ordering[Variant] = implicitly[Ordering[Variant]]

      def pkOrd: Ordering[Locus] = implicitly[Ordering[Locus]]

      def kct: ClassTag[Variant] = implicitly[ClassTag[Variant]]

      def pkct: ClassTag[Locus] = implicitly[ClassTag[Locus]]
    }

  def variantUnitRdd(sc: SparkContext, input: String): RDD[(Variant, Unit)] =
    sc.textFileLines(input)
      .map {
        _.map { line =>
          val fields = line.split(":")
          if (fields.length != 4)
            fatal("invalid variant: expect `CHR:POS:REF:ALT1,ALT2,...,ALTN'")
          val ref = fields(2)
          (Variant(fields(0),
            fields(1).toInt,
            ref,
            fields(3).split(",").map(alt => AltAllele(ref, alt))), ())
        }.value
      }
}

object VariantSubgen {
  val random = VariantSubgen(
    contigGen = Gen.identifier,
    startGen = Gen.posInt,
    nAllelesGen = Gen.frequency((5, Gen.const(2)), (1, Gen.choose(2, 10))),
    refGen = genDNAString,
    altGen = Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))))

  val plinkCompatible = random.copy(
    contigGen = Gen.choose(1, 22).map(_.toString)
  )

  val biallelic = random.copy(nAllelesGen = Gen.const(2))
}

case class VariantSubgen(
  contigGen: Gen[String],
  startGen: Gen[Int],
  nAllelesGen: Gen[Int],
  refGen: Gen[String],
  altGen: Gen[String]) {

  def gen: Gen[Variant] =
    for (contig <- contigGen;
      start <- startGen;
      nAlleles <- nAllelesGen;
      ref <- refGen;
      altAlleles <- Gen.distinctBuildableOfN[Array, String](
        nAlleles,
        altGen)
        .filter(!_.contains(ref))) yield
      Variant(contig, start, ref, altAlleles.tail.map(alt => AltAllele(ref, alt)))
}

case class Variant(contig: String,
  start: Int,
  ref: String,
  altAlleles: IndexedSeq[AltAllele]) extends Ordered[Variant] {

  /* The position is 1-based. Telomeres are indicated by using positions 0 or N+1, where N is the length of the
       corresponding chromosome or contig. See the VCF spec, v4.2, section 1.4.1. */
  require(start >= 0, s"invalid variant: negative position: `${this.toString}'")
  require(!ref.isEmpty, s"invalid variant: empty contig: `${this.toString}'")

  def nAltAlleles: Int = altAlleles.length

  def isBiallelic: Boolean = nAltAlleles == 1

  // FIXME altAllele, alt to be deprecated
  def altAllele: AltAllele = {
    require(isBiallelic, "called altAllele on a non-biallelic variant")
    altAlleles(0)
  }

  def alt: String = altAllele.alt

  def nAlleles: Int = 1 + nAltAlleles

  def allele(i: Int): String = if (i == 0)
    ref
  else
    altAlleles(i - 1).alt

  def nGenotypes = Variant.nGenotypes(nAlleles)

  def locus: Locus = Locus(contig, start)

  def isAutosomalOrPseudoAutosomal: Boolean =
    (!isMitochondrial && !isX && !isY) || inXPar || inYPar

  def isMitochondrial = contig.toUpperCase == "M" || contig.toUpperCase == "MT" || contig == "26"

  // PAR regions of sex chromosomes: https://en.wikipedia.org/wiki/Pseudoautosomal_region
  // Boundaries for build GRCh37: http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/
  def inXParPos: Boolean = (60001 <= start && start <= 2699520) || (154931044 <= start && start <= 155260560)
  def inYParPos: Boolean = (10001 <= start && start <= 2649520) || (59034050 <= start && start <= 59363566)

  // FIXME: will replace with contig == "X" etc once bgen/plink support is merged and conversion is handled by import
  def inXPar: Boolean = isX && inXParPos
  def inYPar: Boolean = isY && inYParPos

  def inXNonPar: Boolean = isX && !inXParPos
  def inYNonPar: Boolean = isY && !inYParPos

  private def isX: Boolean = contig == "x" || contig == "X" || contig == "23" || contig == "25"
  private def isY: Boolean = contig == "y" || contig == "Y" || contig == "24"

  import CopyState._

  def copyState(sex: Sex.Sex): CopyState =
    if (sex == Sex.Male)
      if (inXNonPar)
        HemiX
      else if (inYNonPar)
        HemiY
      else
        Auto
    else
      Auto

  def compare(that: Variant): Int = {
    var c = Contig.compare(contig, that.contig)
    if (c != 0)
      return c

    c = start.compare(that.start)
    if (c != 0)
      return c

    c = ref.compare(that.ref)
    if (c != 0)
      return c

    c = nAltAlleles.compare(that.nAltAlleles)
    if (c != 0)
      return c

    var i = 0
    while (i < altAlleles.length) {
      c = altAlleles(i).alt.compare(that.altAlleles(i).alt)
      if (c != 0)
        return c
      i += 1
    }

    0
  }

  override def toString: String =
    s"$contig:$start:$ref:${altAlleles.map(_.alt).mkString(",")}"

  def toRow = {
    Row.fromSeq(Array(
      contig,
      start,
      ref,
      altAlleles.map { a => Row.fromSeq(Array(a.ref, a.alt)) }))
  }

  def toJSON: JValue = JObject(
    ("contig", JString(contig)),
    ("start", JInt(start)),
    ("ref", JString(ref)),
    ("altAlleles", JArray(altAlleles.map(_.toJSON).toList))
  )
}