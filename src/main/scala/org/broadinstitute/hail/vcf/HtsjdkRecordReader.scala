package org.broadinstitute.hail.vcf

import org.apache.spark.Accumulable
import org.apache.spark.sql.Row
import org.broadinstitute.hail.expr
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
  def readRecord(reportAcc: Accumulable[mutable.Map[Int, Int], Int],
    line: String,
    typeMap: Array[(String, VCFSignature)], storeGQ: Boolean): (Variant, Annotation, Iterable[Genotype]) = {
    val vc = codec.decode(line)

    val pass = vc.filtersWereApplied() && vc.getFilters.isEmpty
    val filts: mutable.WrappedArray[String] = {
      if (vc.filtersWereApplied && vc.isNotFiltered)
        Array("PASS")
      else {
        val arr = vc.getFilters.asScala.toArray
        arr
      }
    }
    val rsid = vc.getID

    val ref = vc.getReference.getBaseString
    val v = Variant(vc.getContig,
      vc.getStart,
      ref,
      vc.getAlternateAlleles.iterator.asScala.map(a => AltAllele(ref, a.getBaseString)).toArray)

    val infoAttrs = vc.getAttributes
      .asScala
      .toMap

    val infoRow = Row.fromSeq(typeMap.map { case (key, sig) =>
      infoAttrs.get(key)
        .map(elem => HtsjdkRecordReader.mapType(elem, sig))
        .orNull
    })
    val va = Row.fromSeq(Array(
      vc.getPhredScaledQual,
      filts,
      pass,
      rsid,
      infoRow))

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

  def mapType(value: Any, sig: VCFSignature): Any = {
    value match {
      case str: String =>
        sig.dType match {
          case expr.TInt => str.toInt
          case expr.TDouble => str.toDouble
          case expr.TArray(expr.TInt) => str.split(",").map(_.toInt): mutable.WrappedArray[Int]
          case expr.TArray(expr.TDouble) => str.split(",").map(_.toDouble): mutable.WrappedArray[Double]
          case expr.TArray(expr.TString) => str.split(","): mutable.WrappedArray[String]
          case _ => value
        }
      case i: Array[_] =>
        sig.dType match {
          case expr.TArray(expr.TInt) => i.map(_.asInstanceOf[String].toInt): mutable.WrappedArray[Int]
          case expr.TArray(expr.TDouble) => i.map(_.asInstanceOf[String].toDouble): mutable.WrappedArray[Double]
          case expr.TArray(expr.TString) => i.map(_.asInstanceOf[String]): mutable.WrappedArray[String]
        }
      case stupid: java.util.ArrayList[_] =>
        (sig.dType: @unchecked) match {
          case expr.TArray(expr.TInt) =>
            stupid.asScala.map(_.asInstanceOf[String].toInt).toArray: mutable.WrappedArray[Int]
          case expr.TArray(expr.TDouble) =>
            stupid.asScala.map(_.asInstanceOf[String].toDouble).toArray: mutable.WrappedArray[Double]
          case expr.TArray(expr.TString) =>
            stupid.asScala.map(_.asInstanceOf[String]).toArray[String]: mutable.WrappedArray[String]
        }
      case stupid: java.util.ArrayList[_] =>
        sig.typeOf match {
          case "Array[Int]" => stupid.asScala.iterator.map(_.asInstanceOf[String].toInt).toIndexedSeq
          case "Array[Double]" => stupid.asScala.iterator.map(_.asInstanceOf[String].toDouble).toIndexedSeq
          case "Array[String]" => stupid.asScala.iterator.map(_.asInstanceOf[String]).toIndexedSeq
        }
      case _ => value
    }
  }
}
