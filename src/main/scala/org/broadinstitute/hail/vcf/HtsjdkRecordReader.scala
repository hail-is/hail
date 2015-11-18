package org.broadinstitute.hail.vcf

import htsjdk.variant.variantcontext.Allele
import org.broadinstitute.hail.variant._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() { throw new UnsupportedOperationException }
}

class HtsjdkRecordReader(codec: htsjdk.variant.vcf.VCFCodec)
  extends AbstractRecordReader {
  override def readRecord(line: String): Iterator[(Variant, Iterator[Genotype])] = {

    val vc = codec.decode(line)
    if (vc.isBiallelic) {
      val variant = Variant(vc.getContig, vc.getStart, vc.getReference.getBaseString,
        vc.getAlternateAllele(0).getBaseString)
      Iterator.single((variant,
        for (g <- vc.getGenotypes.iterator.asScala) yield {

          val gt = if (g.isNoCall)
            None
          else if (g.isHomRef)
            Some(0)
          else if (g.isHet)
            Some(1)
          else {
            assert(g.isHomVar)
            Some(2)
          }

          val ad: Option[IndexedSeq[Int]] = if (g.hasAD) {
            val gad = g.getAD
            Some(Array(gad(0), gad(1)))
          } else
            None

          val dp = if (g.hasDP)
            Some(g.getDP)
          else
            None

          val gq = if (g.hasGQ)
            Some(g.getGQ)
          else
            None

          val pl: Option[IndexedSeq[Int]] =
            if (g.hasPL) {
              val gpl = g.getPL
              Some(Array(gpl(0), gpl(1), gpl(2)))
            } else
              None

          Genotype(gt, ad, dp, gq, pl)
        }))
    } else {

      // build one biallelic variant and set of genotypes for each alternate allele (except spanning deletions)
      val ref = vc.getReference
      val alts = vc.getAlternateAlleles.asScala.filter(_ != Allele.SPAN_DEL)
      val altIndices = alts.map(vc.getAlleleIndex) // index in the VCF, used to access AD and PL fields
      val biVs = alts.map { alt => Variant(vc.getContig, vc.getStart, ref.getBaseString, alt.getBaseString) } //FixMe: need to normalize strings
      val biGBs = alts.map { _ => new ArrayBuffer[Genotype] }

      for (g <- vc.getGenotypes.iterator.asScala) {
        for (((alt, j), i) <- alts.zip(altIndices).zipWithIndex) {

          val gt = if (g.isNoCall)
            None
          else
            Some(g.getAlleles.asScala.count(_ == alt)) // downcode other alts to ref, preserving the count of this alt

          val ad: Option[IndexedSeq[Int]] = if (g.hasAD) {
            val gad = g.getAD
            Some(Array(gad.sum - gad(j), gad(j))) // consistent with downcoding other alts to the ref
            //            (gad(0), gad(j))  // what bcftools does

          } else
            None

          val dp = if (g.hasDP)
            Some(g.getDP)
          else
            None

          /// FIXME based on PL
          val gq = if (g.hasGQ)
            Some(g.getGQ)
          else
            None

          val pl: Option[IndexedSeq[Int]] = if (g.hasPL) {
              val n = vc.getNAlleles
              val gpl = g.getPL
              def pl(gt: Int) = (for (k <- 0 until n; l <- k until n; if List(k, l).count(_ == j) == gt)
                yield l * (l + 1) / 2 + k).map(gpl).min
              Some(Array(pl(0), pl(1), pl(2))) // for each downcoded genotype, minimum PL among original genotypes that downcode to it
              //            (gpl(0), gpl(j * (j + 1) / 2), gpl(j * (j + 1) / 2 + j))  // what bcftools does; ignores all het-non-ref PLs
            } else
              None

          biGBs(i) += Genotype(gt, ad, dp, gq, pl)
        }
      }
      biVs.iterator.zip(biGBs.iterator.map(_.iterator))
    }
  }
}


object HtsjdkRecordReaderBuilder extends AbstractRecordReaderBuilder {
  def result(headerLines: Array[String]): HtsjdkRecordReader = {
    val codec = new htsjdk.variant.vcf.VCFCodec()
    codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
    new HtsjdkRecordReader(codec)
  }
}
