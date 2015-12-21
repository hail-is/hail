package org.broadinstitute.hail.vcf

import htsjdk.variant.variantcontext.Allele
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() {
    throw new UnsupportedOperationException
  }
}

class HtsjdkRecordReader(codec: htsjdk.variant.vcf.VCFCodec) extends Serializable {

  def infoToString(ar: AnyRef): String = {
    ar match {
      case iter: Iterable[_] => if (iter.isEmpty) "" else iter.map(_.toString).mkString(", ")
      case _ => ar.toString
    }
  }

  def readRecord(line: String): Iterator[(Variant, AnnotationData, Iterator[Genotype])] = {
    val vc = codec.decode(line)
    //maybe count tabs to get filter field
    val pass = (vc.filtersWereApplied() && vc.getFilters.size() == 0).toString
    val filts = {
      if (vc.filtersWereApplied && vc.isNotFiltered)
        "PASS"
      else {
        if (vc.getFilters.isEmpty)
          ""
        else
          vc.getFilters.toArray.map(_.toString).mkString(",")
      }
    }
    val rsid = vc.getID
    if (vc.isBiallelic) {
      val variant = Variant(vc.getContig, vc.getStart, vc.getReference.getBaseString,
        vc.getAlternateAllele(0).getBaseString)
      Iterator.single((variant, Annotations[String](Map[String, Map[String, String]]("info" -> vc.getAttributes
        .asScala
        .mapValues(infoToString)
        .toMap),
        Map[String, String](
          "qual" -> vc.getPhredScaledQual.toString,
          "filters" -> filts,
          "pass" -> pass,
          "rsid" -> rsid)),
        for (g <- vc.getGenotypes.iterator.asScala) yield {

          val gt = if (g.isNoCall)
            -1
          else if (g.isHomRef)
            0
          else if (g.isHet)
            1
          else {
            assert(g.isHomVar)
            2
          }

          val ad = if (g.hasAD) {
            val gad = g.getAD
            (gad(0), gad(1))
          } else
            (0, 0)

          val dp = if (g.hasDP)
            g.getDP
          else
            0

          val pl = if (g.isCalled) {
            if (g.hasPL) {
              val gpl = g.getPL
              (gpl(0), gpl(1), gpl(2))
            } else
              (0, 0, 0)
          } else
            null

          Genotype(gt, ad, dp, pl)
        }))
    } else {

//      build one biallelic variant and set of genotypes for each alternate allele (except spanning deletions)
      val ref = vc.getReference
      val alts = vc.getAlternateAlleles.asScala.filter(_ != Allele.SPAN_DEL)
      val altIndices = alts.map(vc.getAlleleIndex) // index in the VCF, used to access AD and PL fields
      val biVs = alts.map { alt =>
          (Variant(vc.getContig, vc.getStart, ref.getBaseString, alt.getBaseString, wasSplit = true),
          Annotations[String](Map[String, Map[String, String]]("info" -> vc.getAttributes
            .asScala
            .mapValues(infoToString)
            .toMap),
            Map[String, String](
              "qual" -> vc.getPhredScaledQual.toString,
              "filters" -> filts,
              "pass" -> pass,
              "rsid" -> rsid,
              "multiallelic" -> "true")))
        } //FixMe: need to normalize strings
      val n = vc.getNAlleles
      val biGBs = alts.map { _ => new ArrayBuffer[Genotype] }

      for (g <- vc.getGenotypes.iterator.asScala) {

        val gadSum = if (g.hasAD) g.getAD.sum else 0
        val dp = if (g.hasDP) g.getDP else 0

        for (((alt, j), i) <- alts.zip(altIndices).zipWithIndex) {

          val gt = if (g.isCalled)
//            downcode other alts to ref, preserving the count of this alt:
            g.getAlleles.asScala.count(_ == alt)
          else
            -1

          val ad = if (g.hasAD) {
            val gad = g.getAD
//            consistent with downcoding other alts to the ref:
            (gadSum - gad(j), gad(j))
//            what bcftools does:
//            (gad(0), gad(j))
          } else
            (0, 0)

          val pl = if (g.isCalled) {
            if (g.hasPL) {
//              for each downcoded genotype, minimum PL among original genotypes that downcode to it:
              def pl(gt: Int) = (for (k <- 0 until n; l <- k until n; if List(k, l).count(_ == j) == gt)
                yield l * (l + 1) / 2 + k).map(g.getPL.apply).min
              (pl(0), pl(1), pl(2))
//            what bcftools does; ignores all het-non-ref PLs
//            val gpl = g.getPL
//            (gpl(0), gpl(j * (j + 1) / 2), gpl(j * (j + 1) / 2 + j))
            } else
              (0, 0, 0)
          } else
            null


          val fakeRef: Boolean =
            g.isCalled && g.getAlleles.asScala.count(_ == ref) != 2 - gt

          biGBs(i) += Genotype(gt, ad, dp, pl, fakeRef)
        }
      }
      biVs.iterator.zip(biGBs.iterator.map(_.iterator)).map {
        case ((v, vi), gs) => (v, vi, gs)
      }
    }
  }
}

object HtsjdkRecordReader {
  def apply(headerLines: Array[String]): HtsjdkRecordReader = {
    val codec = new htsjdk.variant.vcf.VCFCodec()
    codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
    new HtsjdkRecordReader(codec)
  }
}
