package is.hail.variant

import is.hail.annotations.{Region, RegionValue}
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr._
import is.hail.sparkextras.OrderedKey
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

import scala.collection.JavaConverters._
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

object Contig {
  def gen(gr: GenomeReference): Gen[(String, Int)] = Gen.oneOfSeq(gr.lengths.toSeq)

  def gen(gr: GenomeReference, contig: String): Gen[(String, Int)] = {
    assert(gr.isValidContig(contig), s"Contig $contig not found in genome reference.")
    Gen.const((contig, gr.contigLength(contig)))
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
    alt: String): Variant = Variant(contig, start, ref, Array(AltAllele(ref, alt)))

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: Array[String]): Variant =
    Variant(contig, start, ref, alts.map(alt => AltAllele(ref, alt)))

  def apply(contig: String,
    start: Int,
    ref: String,
    alts: java.util.ArrayList[String]): Variant = Variant(contig, start, ref, alts.asScala.toArray)

  def parse(str: String): Variant = {
    val colonSplit = str.split(":")
    if (colonSplit.length != 4)
      fatal(s"invalid variant, expected 4 colon-delimited fields, found ${ colonSplit.length }: $str")
    val Array(contig, start, ref, alts) = colonSplit
    Variant(contig, start.toInt, ref, alts.split(","))
  }

  def fromRegionValue(r: Region, offset: Long): Variant = {
    val t = TVariant.representation()
    val altsType = t.fieldType(3).asInstanceOf[TArray]

    val contig = TString.loadString(r, t.loadField(r, offset, 0))
    val pos = r.loadInt(t.loadField(r, offset, 1))
    val ref = TString.loadString(r, t.loadField(r, offset, 2))

    val altsOffset = t.loadField(r, offset, 3)
    val nAlts = altsType.loadLength(r, altsOffset)
    val altArray = new Array[AltAllele](nAlts)
    var i = 0
    while (i < nAlts) {
      val o = altsType.loadElement(r, altsOffset, nAlts, i)
      altArray(i) = AltAllele.fromRegionValue(r, o)
      i += 1
    }
    Variant(contig, pos, ref, altArray)
  }

  def fromRegionValue(rv: RegionValue): Variant =
    fromRegionValue(rv.region, rv.offset)

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

  def fromRow(r: Row) =
    Variant(r.getAs[String](0),
      r.getAs[Int](1),
      r.getAs[String](2),
      r.getSeq[Row](3)
        .map(s => AltAllele.fromRow(s))
        .toArray)

  def orderedKey(gr: GenomeReference): OrderedKey[Locus, Variant] =
    new OrderedKey[Locus, Variant] {
      def project(key: Variant): Locus = key.locus

      val kOrd: Ordering[Variant] = gr.variantOrdering

      val pkOrd: Ordering[Locus] = gr.locusOrdering

      val kct: ClassTag[Variant] = implicitly[ClassTag[Variant]]

      val pkct: ClassTag[Locus] = implicitly[ClassTag[Locus]]
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
    contigGen = Contig.gen(),
    nAllelesGen = Gen.frequency((5, Gen.const(2)), (1, Gen.choose(2, 10))),
    refGen = genDNAString,
    altGen = Gen.frequency((10, genDNAString),
      (1, Gen.const("*"))))

  val plinkCompatible = random.copy(contigGen = Contig.gen(nameGen = Gen.choose(1, 22).map(_.toString)))

  val biallelic = random.copy(nAllelesGen = Gen.const(2))

  def fromGenomeRef(gr: GenomeReference): VariantSubgen =
    random.copy(contigGen = Contig.gen(gr))
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
      altAlleles <- Gen.distinctBuildableOfN[Array, String](
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

  def isAutosomalOrPseudoAutosomal(gr: GRBase): Boolean = isAutosomal(gr) || inXPar(gr) || inYPar(gr)

  def isAutosomal(gr: GRBase): Boolean = !(inX(gr) || inY(gr) || isMitochondrial(gr))

  def isMitochondrial(gr: GRBase): Boolean = gr.isMitochondrial(contig)

  def inXPar(gr: GRBase): Boolean = gr.inXPar(locus)

  def inYPar(gr: GRBase): Boolean = gr.inYPar(locus)

  def inXNonPar(gr: GRBase): Boolean = inX(gr) && !inXPar(gr)

  def inYNonPar(gr: GRBase): Boolean = inY(gr) && !inYPar(gr)

  private def inX(gr: GRBase): Boolean = gr.inX(contig)

  private def inY(gr: GRBase): Boolean = gr.inY(contig)

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

  def compare(that: Variant, gr: GenomeReference): Int = gr.compare(this, that)

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
}
