package is.hail.io.vcf

import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.vcf.VCFConstants
import is.hail.annotations._
import is.hail.expr._
import is.hail.utils._
import is.hail.variant._

import scala.collection.JavaConverters._
import scala.collection.mutable

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() {
    throw new UnsupportedOperationException
  }
}

class HtsjdkRecordReader(val callFields: Set[String]) extends Serializable {

  import HtsjdkRecordReader._

  def readVariantInfo(vc: VariantContext, infoSignature: Option[TStruct]): (Variant, Annotation) = {
    val filters: Set[String] = {
      if (!vc.filtersWereApplied)
        null
      else if (vc.isNotFiltered)
        Set()
      else
        vc.getFilters.asScala.toSet
    }
    val vcID = vc.getID
    val rsid = if (vcID == ".")
      null
    else
      vcID

    val ref = vc.getReference.getBaseString
    val v = Variant(vc.getContig,
      vc.getStart,
      ref,
      vc.getAlternateAlleles.iterator.asScala.map(a => {
        val base = if (a.getBaseString.isEmpty) "." else a.getBaseString // TODO: handle structural variants
        AltAllele(ref, base)
      }).toArray)
    val nAlleles = v.nAlleles
    val nGeno = v.nGenotypes

    val info = infoSignature.map { sig =>
      val a = Annotation(
        sig.fields.map { f =>
          val a = vc.getAttribute(f.name)
          try {
            cast(a, f.typ)
          } catch {
            case e: Exception =>
              fatal(
                s"""variant $v: INFO field ${ f.name }:
                   |  unable to convert $a (of class ${ a.getClass.getCanonicalName }) to ${ f.typ }:
                   |  caught $e""".stripMargin)
          }
        }: _*)
      assert(sig.typeCheck(a))
      a
    }

    val va = info match {
      case Some(infoAnnotation) => Annotation(rsid, vc.getPhredScaledQual, filters, infoAnnotation)
      case None => Annotation(rsid, vc.getPhredScaledQual, filters)
    }

    (v, va)
  }

  def readRecord(vc: VariantContext,
    infoSignature: Option[TStruct],
    genotypeSignature: Type): (Variant, (Annotation, Iterable[Annotation])) = {

    val (v, va) = readVariantInfo(vc, infoSignature)
    val nAlleles = v.nAlleles

    val gs = vc.getGenotypes.iterator.asScala.map { g =>

      val alleles = g.getAlleles.asScala
      assert(alleles.length == 1 || alleles.length == 2,
        s"expected 1 or 2 alleles in genotype, but found ${ alleles.length }")
      val a0 = alleles(0)
      val a1 = if (alleles.length == 2)
        alleles(1)
      else
        a0

      assert(a0.isCalled || a0.isNoCall)
      assert(a1.isCalled || a1.isNoCall)
      assert(a0.isCalled == a1.isCalled)

      val gt = if (a0.isCalled) {
        val i = vc.getAlleleIndex(a0)
        val j = vc.getAlleleIndex(a1)
        Genotype.gtIndexWithSwap(i, j)
      } else null

      val a = Annotation(
        genotypeSignature.asInstanceOf[TStruct].fields.map { f =>
          val a =
            if (f.name == "GT")
              gt
            else {
              var x = g.getAnyAttribute(f.name)
              // getAnyAttribute returns empty list for missing AD, PL
              if (f.name == "AD" && !g.hasAD)
                x = null
              if (f.name == "PL" && !g.hasPL)
                x = null

              if (x == null || f.typ != TCall)
                x
              else {
                try {
                  GenericRecordReader.getCall(x.asInstanceOf[String], nAlleles)
                } catch {
                  case e: Exception =>
                    fatal(
                      s"""variant $v: Genotype field ${ f.name }:
                         |  unable to convert $x (of class ${ x.getClass.getCanonicalName }) to ${ f.typ }:
                         |  caught $e""".stripMargin)
                }
              }
            }

          try {
            var r = HtsjdkRecordReader.cast(a, f.typ)

            // handle diploid
            if (f.name == "PL" && r != null) {
              val pl = r.asInstanceOf[IndexedSeq[Int]]

              if (alleles.length == 1) {
                val expandedPL = Array.fill(v.nGenotypes)(HtsjdkRecordReader.haploidNonsensePL)
                var i = 0
                while (i < pl.length) {
                  expandedPL(triangle(i + 1) - 1) = pl(i)
                  i += 1
                }
                r = expandedPL: IndexedSeq[Int]
              }
            }

            r
          } catch {
            case e: Exception =>
              fatal(
                s"""variant $v: Genotype field ${ f.name }:
                   |  unable to convert $a (of class ${ a.getClass.getCanonicalName }) to ${ f.typ }:
                   |  caught $e""".stripMargin)
          }
        }: _*)
      assert(genotypeSignature.typeCheck(a))
      a
    }.toArray

    (v, (va, gs))
  }
}

object GenericRecordReader {
  val haploidRegex = """^([0-9]+)$""".r
  val diploidRegex = """^([0-9]+)([|/]([0-9]+))$""".r

  def getCall(gt: String, nAlleles: Int): Call = {
    val call: Call = gt match {
      case diploidRegex(a0, _, a1) => Call(Genotype.gtIndexWithSwap(a0.toInt, a1.toInt))
      case VCFConstants.EMPTY_GENOTYPE => null
      case haploidRegex(a0) => Call(Genotype.gtIndexWithSwap(a0.toInt, a0.toInt))
      case VCFConstants.EMPTY_ALLELE => null
      case _ => fatal(s"Invalid input format for Call type. Found `$gt'.")
    }
    Call.check(call, nAlleles)
    call
  }
}

object HtsjdkRecordReader {

  val haploidNonsensePL = 1000

  def cast(value: Any, t: Type): Any = {
    ((value, t): @unchecked) match {
      case (null, _) => null
      case (".", _) => null
      case (s: String, TArray(TInt32)) =>
        s.split(",").map(x => (if (x == ".") null else x.toInt): java.lang.Integer): IndexedSeq[java.lang.Integer]
      case (s: String, TArray(TFloat64)) =>
        s.split(",").map(x => (if (x == ".") null else x.toDouble): java.lang.Double): IndexedSeq[java.lang.Double]
      case (s: String, TArray(TString)) =>
        s.split(",").map(x => if (x == ".") null else x): IndexedSeq[String]
      case (s: String, TBoolean) => s.toBoolean
      case (b: Boolean, TBoolean) => b
      case (s: String, TString) => s
      case (s: String, TInt32) => s.toInt
      case (s: String, TFloat64) => if (s == "nan") Double.NaN else s.toDouble

      case (i: Int, TInt32) => i
      case (d: Double, TInt32) => d.toInt

      case (d: Double, TFloat64) => d
      case (f: Float, TFloat64) => f.toDouble

      case (f: Float, TFloat32) => f
      case (d: Double, TFloat32) => d.toFloat

      case (l: java.util.List[_], TArray(TInt32)) =>
        l.asScala.iterator.map[java.lang.Integer] {
          case "." => null
          case s: String => s.toInt
          case i: Int => i
        }.toArray[java.lang.Integer]: IndexedSeq[java.lang.Integer]
      case (l: java.util.List[_], TArray(TFloat64)) =>
        l.asScala.iterator.map[java.lang.Double] {
          case "." => null
          case s: String => s.toDouble
          case i: Int => i.toDouble
          case d: Double => d
        }.toArray[java.lang.Double]: IndexedSeq[java.lang.Double]
      case (l: java.util.List[_], TArray(TString)) =>
        l.asScala.iterator.map {
          case "." => null
          case s: String => s
          case i: Int => i.toString
          case d: Double => d.toString
        }.toArray[String]: IndexedSeq[String]

      case (i: Int, TCall) => i
      case (s: String, TCall) => s.toInt
    }
  }
}
