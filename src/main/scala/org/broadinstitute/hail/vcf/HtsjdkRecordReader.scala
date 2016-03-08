package org.broadinstitute.hail.vcf

import org.apache.spark.Accumulable
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.VCFReport
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
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
    line: String,
    infoSignatures: TStruct,
    storeGQ: Boolean): (Variant, Annotations, Iterable[Genotype]) = {
    val vc = codec.decode(line)

    val pass = vc.filtersWereApplied() && vc.getFilters.isEmpty
    val filts = {
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
      vc.getAlternateAlleles.iterator.asScala.map(a => AltAllele(ref, a.getBaseString)).toArray)

    val va = Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
      .asScala
      .iterator
      .flatMap { case (k, v) =>
        infoSignatures.fields.get(k).map { f =>
          (k, cast(v, f.`type`))
        }
      }.toMap),
      "qual" -> vc.getPhredScaledQual,
      "filters" -> filts,
      "pass" -> pass,
      "rsid" -> rsid))

    val gb = new GenotypeBuilder(v)

    // FIXME compress
    val noCall = Genotype()
    val gsb = new GenotypeStreamBuilder(v, true)
    vc.getGenotypes.iterator.asScala.foreach { g =>

      val alleles = g.getAlleles.asScala
      assert(alleles.length == 2)
      val a0 = alleles(0)
      val a1 = alleles(1)

      assert(a0.isCalled || a0.isNoCall)
      assert(a1.isCalled || a1.isNoCall)
      assert(a0.isCalled == a1.isCalled)

      var filter = false
      gb.clear()

      var pl = g.getPL
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
      if (g.hasAD)
        gb.setAD(ad)

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
        gb.setPL(pl)

      if (g.hasGQ) {
        val gq = g.getGQ
        gb.setGQ(gq)

        if (!storeGQ) {
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

    (v, va, gsb.result())
  }
}

object HtsjdkRecordReader {
  def apply(headerLines: Array[String]): HtsjdkRecordReader = {
    val codec = new htsjdk.variant.vcf.VCFCodec()
    codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
    new HtsjdkRecordReader(codec)
  }

  // FIXME document types returned by htsjdk, support full set of types
  def cast(value: Any, t: Type): Any = t match {
    case TInt =>
      value match {
        case s: String => s.toInt
        case _ => value
      }

    case TDouble =>
      value match {
        case s: String => s.toDouble
        case _ => value
      }

    case TArray(TInt) =>
      value match {
        case s: String =>
          s.split(",").iterator.map(_.toInt).toArray: IndexedSeq[Int]
        case  al: java.util.ArrayList[_] =>
          al.asScala.iterator.map(v => cast(v, TInt).asInstanceOf[Int]).toArray: IndexedSeq[Int]
        case it: Iterable[_] =>
          it.iterator.map(v => cast(v, TInt).asInstanceOf[Int]).toArray: IndexedSeq[Int]
        case _ => value
      }

    case TArray(TDouble) =>
      value match {
        case s: String =>
          s.split(",").iterator.map(_.toDouble).toArray: IndexedSeq[Double]
        case  al: java.util.ArrayList[_] =>
          al.asScala.iterator.map(v => cast(v, TDouble).asInstanceOf[Double]).toArray: IndexedSeq[Double]
        case it: Iterable[_] =>
          it.iterator.map(v => cast(v, TDouble).asInstanceOf[Double]).toArray: IndexedSeq[Double]
        case _ => value
      }

    case TArray(TString) =>
      value match {
        case s: String =>
          s.split(",")
        case  al: java.util.ArrayList[_] =>
          al.asScala.iterator.map(v => cast(v, TString).asInstanceOf[String]).toArray
        case it: Iterable[_] =>
          it.iterator.map(v => cast(v, TString).asInstanceOf[String]).toArray
        case _ => value
      }

    case _ => value
  }
}
