package org.broadinstitute.hail.vcf

import org.apache.spark.Accumulable
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
    typeMap: Map[String, Any], storeGQ: Boolean): (Variant, Annotations, Iterable[Genotype]) = {
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
      vc.getAlternateAlleles.iterator.asScala.map(a => {
        if (a.getBaseString.isEmpty)
          println(line.take(200))
        AltAllele(ref, a.getBaseString)}
      ).toArray)

    val va = Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
      .asScala
      .mapValues(HtsjdkRecordReader.purgeJavaArrayLists)
      .toMap
      .flatMap { case (k, v) =>
        typeMap.get(k).map { t =>
          (k, HtsjdkRecordReader.mapType(v, t.asInstanceOf[VCFSignature]))
        }
      }),
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

  def purgeJavaArrayLists(ar: AnyRef): Any = {
    ar match {
      case arr: java.util.ArrayList[_] => arr.asScala
      case _ => ar
    }
  }

  def mapType(value: Any, sig: VCFSignature): Any = {
    value match {
      case str: String =>
        sig.typeOf match {
          case "Int" => str.toInt
          case "Double" => str.toDouble
          case "Array[Int]" => str.split(",").map(_.toInt): IndexedSeq[Int]
          case "Array[Double]" => str.split(",").map(_.toDouble): IndexedSeq[Double]
          case _ => value
        }
      case i: IndexedSeq[_] =>
        sig.number match {
          case "Array[Int]" => i.map(_.asInstanceOf[String].toInt)
          case "Array[Double]" => i.map(_.asInstanceOf[String].toDouble)

        }
      case _ => value
    }
  }
}
