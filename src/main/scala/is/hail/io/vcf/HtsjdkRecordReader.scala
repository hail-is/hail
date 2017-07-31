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

abstract class HtsjdkRecordReader[T] extends Serializable {

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
    genotypeSignature: Type): (Variant, (Annotation, Iterable[T]))

  def genericGenotypes: Boolean
}

case class GenotypeRecordReader(vcfSettings: VCFSettings) extends HtsjdkRecordReader[Genotype] {
  def genericGenotypes = false

  def readRecord(vc: VariantContext,
    infoSignature: Option[TStruct],
    genotypeSignature: Type): (Variant, (Annotation, Iterable[Genotype])) = {

    val (v, va) = readVariantInfo(vc, infoSignature)

    val nAlleles = v.nAlleles

    if (vcfSettings.dropSamples)
      return (v, (va, Iterable.empty))

    val gb = new GenotypeBuilder(v.nAlleles, false) // FIXME: make dependent on fields in genotypes; for now, assumes PLs

    val noCall = Genotype()
    val gsb = new GenotypeStreamBuilder(v.nAlleles, isLinearScale = false)

    vc.getGenotypes.iterator.asScala.foreach { g =>

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

      var filter = false
      gb.clear()

      var pl = if (vcfSettings.ppAsPL) {
        val str = g.getAnyAttribute("PP")
        if (str != null)
          str.asInstanceOf[String].split(",").map(_.toInt)
        else null
      }
      else g.getPL

      // support haploid genotypes
      if (alleles.length == 1 && pl != null) {
        val expandedPL = Array.fill(v.nGenotypes)(HtsjdkRecordReader.haploidNonsensePL)
        var i = 0
        while (i < pl.length) {
          expandedPL(triangle(i + 1) - 1) = pl(i)
          i += 1
        }
        pl = expandedPL
      }

      if (g.hasPL) {
        val minPL = pl.min
        if (minPL != 0) {
          pl = pl.clone()
          var i = 0
          while (i < pl.length) {
            pl(i) -= minPL
            i += 1
          }
        }
      }

      var gt = -1 // notCalled

      if (a0.isCalled) {
        val i = vc.getAlleleIndex(a0)
        val j = vc.getAlleleIndex(a1)

        gt = if (i <= j)
          Genotype.gtIndex(i, j)
        else
          Genotype.gtIndex(j, i)

        if (g.hasPL && pl(gt) != 0)
          filter = true

        if (gt != -1)
          gb.setGT(gt)
      }

      val ad = g.getAD
      if (g.hasAD) {
        if (ad.length == nAlleles || !vcfSettings.skipBadAD)
          gb.setAD(ad)
      }

      if (g.hasDP) {
        var dp = g.getDP
        if (g.hasAD) {
          val adsum = ad.sum
          if (dp < adsum)
            filter = true
        }

        gb.setDP(dp)
      }

      if (pl != null)
        gb.setPX(pl)

      if (g.hasGQ) {
        val gq = g.getGQ
        gb.setGQ(gq)

        if (!vcfSettings.storeGQ) {
          if (pl != null) {
            val gqFromPL = Genotype.gqFromPL(pl)

            if (gq != gqFromPL)
              filter = true
          } else
            filter = true
        }
      }

      val odObj = g.getExtendedAttribute("OD")
      if (odObj != null) {
        val od = odObj.asInstanceOf[String].toInt

        if (g.hasAD) {
          val adsum = ad.sum
          if (!g.hasDP)
            gb.setDP(adsum + od)
          else if (adsum + od != g.getDP)
            filter = true
        } else
          filter = true
      }

      if (filter)
        gsb += noCall
      else
        gsb.write(gb)
    }

    (v, (va, gsb.result()))
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

case class GenericRecordReader(callFields: Set[String]) extends HtsjdkRecordReader[Annotation] {
  def genericGenotypes: Boolean = true

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
              val x = g.getAnyAttribute(f.name)
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
            HtsjdkRecordReader.cast(a, f.typ)
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

object HtsjdkRecordReader {

  val haploidNonsensePL = 1000

  def cast(value: Any, t: Type): Any = {
    ((value, t): @unchecked) match {
      case (null, _) => null
      case (".", _) => null
      case (s: String, TArray(TInt)) =>
        s.split(",").map(x => (if (x == ".") null else x.toInt): java.lang.Integer): IndexedSeq[java.lang.Integer]
      case (s: String, TArray(TDouble)) =>
        s.split(",").map(x => (if (x == ".") null else x.toDouble): java.lang.Double): IndexedSeq[java.lang.Double]
      case (s: String, TArray(TString)) =>
        s.split(",").map(x => if (x == ".") null else x): IndexedSeq[String]
      case (s: String, TBoolean) => s.toBoolean
      case (b: Boolean, TBoolean) => b
      case (s: String, TString) => s
      case (s: String, TInt) => s.toInt
      case (s: String, TDouble) => if (s == "nan") Double.NaN else s.toDouble

      case (i: Int, TInt) => i
      case (d: Double, TInt) => d.toInt

      case (d: Double, TDouble) => d
      case (f: Float, TDouble) => f.toDouble

      case (f: Float, TFloat) => f
      case (d: Double, TFloat) => d.toFloat

      case (l: java.util.List[_], TArray(TInt)) =>
        l.asScala.iterator.map[java.lang.Integer] {
          case "." => null
          case s: String => s.toInt
          case i: Int => i
        }.toArray[java.lang.Integer]: IndexedSeq[java.lang.Integer]
      case (l: java.util.List[_], TArray(TDouble)) =>
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
