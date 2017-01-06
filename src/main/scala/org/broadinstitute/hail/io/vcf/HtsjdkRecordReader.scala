package org.broadinstitute.hail.io.vcf

import htsjdk.variant.variantcontext.VariantContext
import org.apache.spark.Accumulable
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._

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

class HtsjdkRecordReader(codec: htsjdk.variant.vcf.VCFCodec) extends Serializable {

  import HtsjdkRecordReader._

  def readRecord(reportAcc: Accumulable[mutable.Map[Int, Int], Int],
    vc: VariantContext,
    infoSignature: Option[TStruct],
    vcfSettings: VCFSettings): (Variant, (Annotation, Iterable[Genotype])) = {

    val pass = vc.filtersWereApplied() && vc.getFilters.isEmpty
    val filters: Set[String] = {
      if (vc.filtersWereApplied && vc.isNotFiltered)
        Set("PASS")
      else
        vc.getFilters.asScala.toSet
    }
    val rsid = vc.getID

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
            cast(a, f.`type`)
          } catch {
            case e: Exception =>
              fatal(
                s"""variant $v: INFO field ${f.name}:
                    |  unable to convert $a (of class ${a.getClass.getCanonicalName}) to ${f.`type`}:
                    |  caught $e""".stripMargin)
          }
        }: _*)
      assert(sig.typeCheck(a))
      a
    }

    val va = info match {
      case Some(infoAnnotation) => Annotation(rsid, vc.getPhredScaledQual, filters, pass, infoAnnotation)
      case None => Annotation(rsid, vc.getPhredScaledQual, filters, pass)
    }

    if (vcfSettings.skipGenotypes)
      return (v, (va, Iterable.empty))

    val gb = new GenotypeBuilder(v.nAlleles, false) //FIXME: make dependent on fields in genotypes; for now, assumes PLs

    // FIXME compress
    val noCall = Genotype()
    val gsb = new GenotypeStreamBuilder(v.nAlleles, isDosage = false, compress = vcfSettings.compress)

    vc.getGenotypes.iterator.asScala.foreach { g =>

      val alleles = g.getAlleles.asScala
      assert(alleles.length == 2, s"expected 2 alleles in genotype, but found ${alleles.length}")
      val a0 = alleles(0)
      val a1 = alleles(1)

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

        if (g.hasPL && pl(gt) != 0) {
          reportAcc += VCFReport.GTPLMismatch
          filter = true
        }

        if (gt != -1)
          gb.setGT(gt)
      }

      val ad = g.getAD
      if (g.hasAD) {
        if (vcfSettings.skipBadAD && ad.length != nAlleles)
          reportAcc += VCFReport.ADInvalidNumber
        else
          gb.setAD(ad)
      }

      if (g.hasDP) {
        var dp = g.getDP
        if (g.hasAD) {
          val adsum = ad.sum
          if (!filter && dp < adsum) {
            reportAcc += VCFReport.ADDPMismatch
            filter = true
          }
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

            if (!filter && gq != gqFromPL) {
              reportAcc += VCFReport.GQPLMismatch
              filter = true
            }
          } else if (!filter) {
            reportAcc += VCFReport.GQMissingPL
            filter = true
          }
        }
      }

      val odObj = g.getExtendedAttribute("OD")
      if (odObj != null) {
        val od = odObj.asInstanceOf[String].toInt

        if (g.hasAD) {
          val adsum = ad.sum
          if (!g.hasDP)
            gb.setDP(adsum + od)
          else if (!filter && adsum + od != g.getDP) {
            reportAcc += VCFReport.ADODDPPMismatch
            filter = true
          }
        } else if (!filter) {
          reportAcc += VCFReport.ODMissingAD
          filter = true
        }
      }

      if (filter)
        gsb += noCall
      else
        gsb.write(gb)
    }

    (v, (va, gsb.result()))
  }
}

object HtsjdkRecordReader {
  def apply(headerLines: Array[String], codec: htsjdk.variant.vcf.VCFCodec): HtsjdkRecordReader = {
    codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
    new HtsjdkRecordReader(codec)
  }

  def cast(value: Any, t: Type): Any = {
    ((value, t): @unchecked) match {
      case (null, _) => null
      case (s: String, TArray(TInt)) =>
        s.split(",").map(_.toInt): IndexedSeq[Int]
      case (s: String, TArray(TDouble)) =>
        s.split(",").map(_.toDouble): IndexedSeq[Double]
      case (s: String, TArray(TString)) =>
        s.split(","): IndexedSeq[String]
      case (s: String, TArray(TChar)) =>
        s.split(","): IndexedSeq[String]
      case (s: String, TBoolean) => s.toBoolean
      case (b: Boolean, TBoolean) => b
      case (s: String, TString) => s
      case (s: String, TChar) => s
      case (s: String, TInt) => s.toInt
      case (s: String, TDouble) => if (s == "nan") Double.NaN else s.toDouble

      case (a: java.util.ArrayList[_], TArray(TInt)) =>
        a.asScala.iterator.map(_.asInstanceOf[String].toInt).toArray: IndexedSeq[Int]
      case (a: java.util.ArrayList[_], TArray(TDouble)) =>
        a.asScala.iterator.map(_.asInstanceOf[String].toDouble).toArray: IndexedSeq[Double]
      case (a: java.util.ArrayList[_], TArray(TString)) =>
        a.asScala.iterator.map(_.asInstanceOf[String]).toArray[String]: IndexedSeq[String]
      case (a: java.util.ArrayList[_], TArray(TChar)) =>
        a.asScala.iterator.map(_.asInstanceOf[String]).toArray[String]: IndexedSeq[String]
    }
  }
}
