package is.hail.variant

import is.hail.annotations.{Annotation, Region, RegionValue}
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

import scala.collection.JavaConverters._

object Contig {
  def gen(rg: ReferenceGenome): Gen[(String, Int)] = Gen.oneOfSeq(rg.lengths.toSeq)

  def gen(rg: ReferenceGenome, contig: String): Gen[(String, Int)] = {
    assert(rg.isValidContig(contig), s"Contig $contig not found in reference genome.")
    Gen.const((contig, rg.contigLength(contig)))
  }

  def gen(nameGen: Gen[String] = Gen.identifier, lengthGen: Gen[Int] = Gen.choose(1000000, 500000000)): Gen[(String, Int)] = for {
    name <- nameGen
    length <- lengthGen
  } yield (name, length)
}

object Variant {
  def apply(contig: String,
    start: Int,
    ref: String,
    alt: String): Variant = {
    Variant(contig, start, ref, Array(AltAllele(ref, alt)))
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alt: String,
    rg: RGBase): Variant = {
    rg.checkVariant(contig, start, ref, alt)
    Variant(contig, start, ref, alt)
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String]): Variant = Variant(contig, start, ref, alts.map(alt => AltAllele(ref, alt)))

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String],
    rg: RGBase): Variant = {
    rg.checkVariant(contig, start, ref, alts)
    Variant(contig, start, ref, alts)
  }

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: java.util.ArrayList[String],
    rg: RGBase): Variant = Variant(contig, start, ref, alts.asScala.toArray, rg)

  def parse(str: String, rg: RGBase): Variant = {
    val elts = str.split(":")
    val size = elts.length
    if (size < 4)
      fatal(s"Invalid string for Variant. Expecting contig:pos:ref:alt1,alt2 -- found `$str'.")

    val contig = elts.take(size - 3).mkString(":")
    Variant(contig, elts(size - 3).toInt, elts(size - 2), elts(size - 1).split(","), rg)
  }

  def variantID(contig: String, start: Int, alleles: IndexedSeq[String]): String = {
    require(alleles.length >= 2)
    s"$contig:$start:${alleles(0)}:${alleles.tail.mkString(",")}"
  }

  def nGenotypes(nAlleles: Int): Int = {
    require(nAlleles > 0, s"called nGenotypes with invalid number of alternates: $nAlleles")
    nAlleles * (nAlleles + 1) / 2
  }

  def gen: Gen[Variant] = VariantSubgen.random.gen

  implicit def arbVariant: Arbitrary[Variant] = Arbitrary(gen)

  def fromLocusAlleles(a: Annotation): Variant = {
    val r = a.asInstanceOf[Row]
    val l = r.getAs[Locus](0)
    val alleles = r.getAs[IndexedSeq[String]](1)
    if (l == null || alleles == null)
      null
    else
      Variant(l.contig, l.position, alleles(0), alleles.tail.map(x => AltAllele(alleles(0), x)))
  }

  def locusAllelesToString(locus: Locus, alleles: IndexedSeq[String]): String =
    s"$locus:${ alleles(0) }:${ alleles.tail.mkString(",") }"
}

object VariantSubgen {
  val random = VariantSubgen(
    contigGen = Contig.gen(),
    nAllelesGen = Gen.frequency((5, Gen.const(2)), (1, Gen.choose(2, 10))),
    refGen = genDNAString,
    altGen = Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))))

  val plinkCompatible = random.copy(contigGen = Contig.gen(nameGen = Gen.choose(1, 22).map(_.toString)))

  val biallelic = random.copy(nAllelesGen = Gen.const(2))

  def fromGenomeRef(rg: ReferenceGenome): VariantSubgen =
    random.copy(contigGen = Contig.gen(rg))
}

case class VariantSubgen(
  contigGen: Gen[(String, Int)],
  nAllelesGen: Gen[Int],
  refGen: Gen[String],
  altGen: Gen[String]) {

  def gen: Gen[Variant] =
    for ((contig, length) <- contigGen;
      start <- Gen.choose(1, length);
      nAlleles <- nAllelesGen;
      ref <- refGen;
      altAlleles <- Gen.distinctBuildableOfN[Array](
        nAlleles - 1,
        altGen)
        .filter(!_.contains(ref))) yield
      Variant(contig, start, ref, altAlleles.map(alt => AltAllele(ref, alt)))
}

trait IVariant { self =>
  def contig(): String

  def start(): Int

  def ref(): String

  def altAlleles(): IndexedSeq[AltAllele]

  def copy(contig: String = contig, start: Int = start, ref: String = ref, altAlleles: IndexedSeq[AltAllele] = altAlleles): Variant =
    Variant(contig, start, ref, altAlleles)

  def nAltAlleles: Int = altAlleles.length

  def isBiallelic: Boolean = nAltAlleles == 1

  // FIXME altAllele, alt to be deprecated
  def altAllele: AltAllele = {
    require(isBiallelic, "called altAllele on a non-biallelic variant")
    altAlleles()(0)
  }

  def alt: String = altAllele.alt

  def nAlleles: Int = 1 + nAltAlleles

  def allele(i: Int): String = if (i == 0)
    ref
  else
    altAlleles()(i - 1).alt

  def nGenotypes = Variant.nGenotypes(nAlleles)

  def locus: Locus = Locus(contig, start)

  def isAutosomalOrPseudoAutosomal(rg: RGBase): Boolean = isAutosomal(rg) || inXPar(rg) || inYPar(rg)

  def isAutosomal(rg: RGBase): Boolean = !(inX(rg) || inY(rg) || isMitochondrial(rg))

  def isMitochondrial(rg: RGBase): Boolean = rg.isMitochondrial(contig)

  def inXPar(rg: RGBase): Boolean = rg.inXPar(locus)

  def inYPar(rg: RGBase): Boolean = rg.inYPar(locus)

  def inXNonPar(rg: RGBase): Boolean = inX(rg) && !inXPar(rg)

  def inYNonPar(rg: RGBase): Boolean = inY(rg) && !inYPar(rg)

  private def inX(rg: RGBase): Boolean = rg.inX(contig)

  private def inY(rg: RGBase): Boolean = rg.inY(contig)

  import CopyState._

  def copyState(sex: Sex.Sex, rg: ReferenceGenome): CopyState =
    if (sex == Sex.Male)
      if (inXNonPar(rg))
        HemiX
      else if (inYNonPar(rg))
        HemiY
      else
        Auto
    else
      Auto

  def compare(that: Variant, rg: ReferenceGenome): Int = rg.compare(this, that)

  def minRep: IVariant = {
    if (ref.length == 1)
      self
    else if (altAlleles.forall(a => a.isStar))
      Variant(contig, start, ref.substring(0, 1), altAlleles.map(_.alt).toArray)
    else {
      val alts = altAlleles.filter(!_.isStar).map(a => a.alt)
      require(alts.forall(ref != _))

      val min_length = math.min(ref.length, alts.map(x => x.length).min)
      var ne = 0

      while (ne < min_length - 1
        && alts.forall(x => ref()(ref.length - ne - 1) == x(x.length - ne - 1))
      ) {
        ne += 1
      }

      var ns = 0
      while (ns < min_length - ne - 1
        && alts.forall(x => ref()(ns) == x(ns))
      ) {
        ns += 1
      }

      if (ne + ns == 0)
        self
      else {
        assert(ns < ref.length - ne && alts.forall(x => ns < x.length - ne))
        Variant(contig, start + ns, ref.substring(ns, ref.length - ne),
          altAlleles.map(a => if (a.isStar) a.alt else a.alt.substring(ns, a.alt.length - ne)).toArray)
      }
    }
  }

  override def toString: String =
    s"$contig:$start:$ref:${ altAlleles.map(_.alt).mkString(",") }"

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

case class Variant(contig: String,
  start: Int,
  ref: String,
  override val altAlleles: IndexedSeq[AltAllele]) extends IVariant {
  require(altAlleles.forall(_.ref == ref))

  /* The position is 1-based. Telomeres are indicated by using positions 0 or N+1, where N is the length of the
       corresponding chromosome or contig. See the VCF spec, v4.2, section 1.4.1. */
  require(start >= 0, s"invalid variant: negative position: `${ this.toString }'")
  require(!ref.isEmpty, s"invalid variant: empty contig: `${ this.toString }'")

  override def minRep: Variant = super.minRep.asInstanceOf[Variant]

  def toLocusAlleles: Row = Row(locus, IndexedSeq(ref) ++ altAlleles.map(_.alt))

  def alleles: IndexedSeq[String] = {
    val a = new Array[String](nAlleles)
    a(0) = ref
    var i = 1
    while (i < a.length) {
      a(i) = altAlleles(i - 1).alt
      i += 1
    }
    a
  }
}
