package is.hail.variant

import is.hail.check.{Arbitrary, Gen}
import is.hail.expr._
import is.hail.sparkextras.OrderedKey
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

import scala.collection.JavaConverters._
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

object Contig {
  def gen: Gen[Contig] = for {
    name <- Gen.identifier
    length <- Gen.posInt
  } yield Contig(name, length)
}

case class Contig(name: String, length: Int) {
  assert(length > 0, s"Contig length must be greater than 0. Contig `$name' has length equal to $length.")
  def toJSON: JValue = JObject(("name", JString(name)), ("length", JInt(length)))
}

object AltAlleleType extends Enumeration {
  type AltAlleleType = Value
  val SNP, MNP, Insertion, Deletion, Complex, Star = Value
}

object CopyState extends Enumeration {
  type CopyState = Value
  val Auto, HemiX, HemiY = Value
}

object AltAllele {
  def sparkSchema: StructType = StructType(Array(
    StructField("ref", StringType, nullable = false),
    StructField("alt", StringType, nullable = false)))

  val expandedType: TStruct = TStruct("ref" -> TString,
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

  implicit def altAlleleOrder: Ordering[AltAllele] = new Ordering[AltAllele] {
    def compare(x: AltAllele, y: AltAllele): Int = x.compare(y)
  }
}

case class AltAllele(ref: String,
  alt: String) {
  require(ref != alt, "ref was equal to alt")
  require(!ref.isEmpty, "ref was an empty string")
  require(!alt.isEmpty, "alt was an empty string")

  import AltAlleleType._

  def altAlleleType: AltAlleleType = {
    if (isSNP)
      SNP
    else if (isInsertion)
      Insertion
    else if (isDeletion)
      Deletion
    else if (isStar)
      Star
    else if (ref.length == alt.length)
      MNP
    else
      Complex
  }

  def isStar: Boolean = alt == "*"

  def isSNP: Boolean = !isStar && ( (ref.length == 1 && alt.length == 1) ||
    (ref.length == alt.length && nMismatch == 1) )

  def isMNP: Boolean = ref.length > 1 &&
    ref.length == alt.length &&
    nMismatch > 1

  def isInsertion: Boolean = ref.length < alt.length && ref(0) == alt(0) && alt.endsWith(ref.substring(1))

  def isDeletion: Boolean = alt.length < ref.length && ref(0) == alt(0) && ref.endsWith(alt.substring(1))

  def isIndel: Boolean = isInsertion || isDeletion

  def isComplex: Boolean = ref.length != alt.length && !isInsertion && !isDeletion

  def isTransition: Boolean = isSNP && {
    val (refChar, altChar) = strippedSNP
    (refChar == 'A' && altChar == 'G') || (refChar == 'G' && altChar == 'A') ||
      (refChar == 'C' && altChar == 'T') || (refChar == 'T' && altChar == 'C')
  }

  def isTransversion: Boolean = isSNP && !isTransition

  def nMismatch: Int = {
    require(ref.length == alt.length, s"invalid nMismatch call on ref `${ ref }' and alt `${ alt }'")
    (ref, alt).zipped.map((a, b) => if (a == b) 0 else 1).sum
  }

  def strippedSNP: (Char, Char) = {
    require(isSNP, "called strippedSNP on non-SNP")
    (ref, alt).zipped.dropWhile { case (a, b) => a == b }.head
  }

  def toRow: Row = Row(ref, alt)

  def toJSON: JValue = JObject(
    ("ref", JString(ref)),
    ("alt", JString(alt)))

  override def toString: String = s"$ref/$alt"

  def compare(that: AltAllele): Int = {
    val c = ref.compare(that.ref)
    if (c != 0)
      return c

    alt.compare(that.alt)
  }
}

object Variant {
  def apply(contig: String,
    start: Int,
    ref: String,
    alt: String)(implicit gr: GenomeReference): Variant = {
    gr.contigIndex.get(contig) match {
      case Some(idx) => Variant(idx, start, ref, Array(AltAllele(ref, alt)))
      case None => fatal(s"Did not find contig `$contig' in genome reference `${ gr.name }'.")
    }
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String])(implicit gr: GenomeReference): Variant = {
    gr.contigIndex.get(contig) match {
      case Some(idx) => Variant(idx, start, ref, alts.map(alt => AltAllele(ref, alt)))
      case None => fatal(s"Did not find contig `$contig' in genome reference `${ gr.name }'.")
    }
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[AltAllele])(implicit gr: GenomeReference): Variant = {
    gr.contigIndex.get(contig) match {
      case Some(idx) => Variant(idx, start, ref, alts)
      case None => fatal(s"Did not find contig `$contig' in genome reference `${ gr.name }'.")
    }
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: java.util.ArrayList[String])(implicit gr: GenomeReference): Variant = Variant(contig, start, ref, alts.asScala.toArray)(gr)

  def apply(contig: Int,
    start: Int,
    ref: String,
    alt: String): Variant = Variant(contig, start, ref, Array(AltAllele(ref, alt)))

  def apply(contig: Int,
    start: Int,
    ref: String,
    alts: Array[String]): Variant =
    Variant(contig, start, ref, alts.map(alt => AltAllele(ref, alt)))


  def parse(str: String)(implicit gr: GenomeReference): Variant = {
    val colonSplit = str.split(":")
    if (colonSplit.length != 4)
      fatal(s"expected 4 colon-delimited fields, but found ${ colonSplit.length }")
    val Array(contig, start, ref, alts) = colonSplit
    Variant(contig, start.toInt, ref, alts.split(","))(gr)
  }

  def nGenotypes(nAlleles: Int): Int = {
    require(nAlleles > 0, s"called nGenotypes with invalid number of alternates: $nAlleles")
    nAlleles * (nAlleles + 1) / 2
  }

  def gen: Gen[Variant] = VariantSubgen.random.gen

  implicit def arbVariant: Arbitrary[Variant] = Arbitrary(gen)

  def sparkSchema: StructType =
    StructType(Array(
      StructField("contig", StringType, nullable = false),
      StructField("start", IntegerType, nullable = false),
      StructField("ref", StringType, nullable = false),
      StructField("altAlleles", ArrayType(AltAllele.sparkSchema, containsNull = false),
        nullable = false)))

  def expandedType: TStruct =
    TStruct("contig" -> TString,
      "start" -> TInt,
      "ref" -> TString,
      "altAlleles" -> TArray(AltAllele.expandedType))

  def fromRow(r: Row, gr: GenomeReference) = {
    val contig = r.getAs[String](0)
    gr.contigIndex.get(contig) match {
      case Some(idx) =>
        Variant(idx,
          r.getAs[Int](1),
          r.getAs[String](2),
          r.getSeq[Row](3)
            .map(s => AltAllele.fromRow(s))
            .toArray)
      case None => fatal(s"Did not find contig `$contig' in genome reference `${ gr.name }'.")
    }

  }

  implicit val orderedKey: OrderedKey[Locus, Variant] =
    new OrderedKey[Locus, Variant] {
      def project(key: Variant): Locus = key.locus

      def kOrd: Ordering[Variant] = implicitly[Ordering[Variant]]

      def pkOrd: Ordering[Locus] = implicitly[Ordering[Locus]]

      def kct: ClassTag[Variant] = implicitly[ClassTag[Variant]]

      def pkct: ClassTag[Locus] = implicitly[ClassTag[Locus]]
    }

  implicit def variantOrder: Ordering[Variant] = new Ordering[Variant] {
    def compare(x: Variant, y: Variant): Int = x.compare(y)
  }
}

object VariantSubgen {
  val random = VariantSubgen(
    genomeRefGen = Gen.const(GenomeReference.GRCh37),
    startGen = Gen.posInt,
    nAllelesGen = Gen.frequency((5, Gen.const(2)), (1, Gen.choose(2, 10))),
    refGen = genDNAString,
    altGen = Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))))

  val plinkCompatible = random.copy(
    genomeRefGen = Gen.const(GenomeReference.GRCh37)
  )

  val biallelic = random.copy(nAllelesGen = Gen.const(2))
}

case class VariantSubgen(
  genomeRefGen: Gen[GenomeReference],
  startGen: Gen[Int],
  nAllelesGen: Gen[Int],
  refGen: Gen[String],
  altGen: Gen[String]) {

  def gen: Gen[Variant] =
    for (gr <- genomeRefGen;
      contig <- Gen.oneOfSeq(gr.contigs.indices);
      start <- startGen;
      nAlleles <- nAllelesGen;
      ref <- refGen;
      altAlleles <- Gen.distinctBuildableOfN[Array, String](
        nAlleles,
        altGen)
        .filter(!_.contains(ref))) yield
      Variant(contig, start, ref, altAlleles.tail.map(alt => AltAllele(ref, alt)))
}

case class Variant(contig: Int,
  start: Int,
  ref: String,
  altAlleles: IndexedSeq[AltAllele]) {

  /* The position is 1-based. Telomeres are indicated by using positions 0 or N+1, where N is the length of the
       corresponding chromosome or contig. See the VCF spec, v4.2, section 1.4.1. */
  require(start >= 0, s"invalid variant: negative position: `${ this.toString }'")
  require(!ref.isEmpty, s"invalid variant: empty contig: `${ this.toString }'")

  def contigStr(gr: GenomeReference): String = gr.contigNames(contig)

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

  def isAutosomalOrPseudoAutosomal(gr: GenomeReference): Boolean =
    isAutosomal(gr) || inXPar(gr) || inYPar(gr)

  def isAutosomal(gr: GenomeReference) = !(inX(gr) || inY(gr) || isMitochondrial(gr))

  def isMitochondrial(gr: GenomeReference) = gr.isMitochondrial(contig)

  def inXPar(gr: GenomeReference): Boolean = gr.inXPar(locus)

  def inYPar(gr: GenomeReference): Boolean = gr.inYPar(locus)

  def inXNonPar(gr: GenomeReference): Boolean = inX(gr) && !inXPar(gr)

  def inYNonPar(gr: GenomeReference): Boolean = inY(gr) && !inYPar(gr)

  def inX(gr: GenomeReference): Boolean = gr.inX(contig)

  def inY(gr: GenomeReference): Boolean = gr.inY(contig)

  import CopyState._

  def copyState(sex: Sex.Sex, gr: GenomeReference): CopyState =
    if (sex == Sex.Male)
      if (inXNonPar(gr))
        HemiX
      else if (inYNonPar(gr))
        HemiY
      else
        Auto
    else
      Auto

  def compare(that: Variant): Int = {
    var c = contig.compare(that.contig)
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

  def minRep: Variant = {
    if (ref.length == 1)
      this
    else if (altAlleles.forall(a => a.isStar))
      Variant(contig, start, ref.substring(0, 1), altAlleles.map(_.alt).toArray)
    else {
      val alts = altAlleles.filter(!_.isStar).map(a => a.alt)
      require(alts.forall(ref != _))

      val min_length = math.min(ref.length, alts.map(x => x.length).min)
      var ne = 0

      while (ne < min_length - 1
        && alts.forall(x => ref(ref.length - ne - 1) == x(x.length - ne - 1))
      ) {
        ne += 1
      }

      var ns = 0
      while (ns < min_length - ne - 1
        && alts.forall(x => ref(ns) == x(ns))
      ) {
        ns += 1
      }

      if (ne + ns == 0)
        this
      else {
        assert(ns < ref.length - ne && alts.forall(x => ns < x.length - ne))
        Variant(contig, start + ns, ref.substring(ns, ref.length - ne),
          altAlleles.map(a => if (a.isStar) a.alt else a.alt.substring(ns, a.alt.length - ne)).toArray)
      }
    }
  }

  def toString(gr: GenomeReference): String = {
    s"${ contigStr(gr) }:$start:$ref:${ altAlleles.map(_.alt).mkString(",") }"
  }

  override def toString: String =
    s"$contig:$start:$ref:${ altAlleles.map(_.alt).mkString(",") }"

  def toRow(gr: GenomeReference) = {
    Row.fromSeq(Array(
      contigStr(gr),
      start,
      ref,
      altAlleles.map { a => Row.fromSeq(Array(a.ref, a.alt)) }))
  }

  def toJSON(gr: GenomeReference): JValue = {
    JObject(
      ("contig", JString(contigStr(gr))),
      ("start", JInt(start)),
      ("ref", JString(ref)),
      ("altAlleles", JArray(altAlleles.map(_.toJSON).toList))
    )
  }
}