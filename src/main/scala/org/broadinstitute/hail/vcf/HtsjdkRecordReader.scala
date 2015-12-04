package org.broadinstitute.hail.vcf

import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import scala.collection.JavaConverters._

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() { throw new UnsupportedOperationException }
}

class HtsjdkRecordReader(codec: htsjdk.variant.vcf.VCFCodec) extends Serializable {
  def readRecord(line: String): (Variant, Iterable[Genotype]) = {
    val vc = codec.decode(line)

    val ref = vc.getReference.getBaseString
    val variant = Variant(vc.getContig,
      vc.getStart,
      ref,
      vc.getAlternateAlleles.iterator.asScala.map(a => AltAllele(ref, a.getBaseString)).toArray)

    val gb = new GenotypeBuilder(variant)

    // FIXME compress
    val gsb = new GenotypeStreamBuilder(variant, true)
    vc.getGenotypes.iterator.asScala.foreach { g =>

      val alleles = g.getAlleles.asScala
      assert(alleles.length == 2)
      val a0 = alleles(0)
      val a1 = alleles(1)

      assert(a0.isCalled || a0.isNoCall)
      assert(a1.isCalled || a1.isNoCall)

      gb.clear()

      var pl = g.getPL

      if (a0.isCalled) {
        val i = vc.getAlleleIndex(a0)
        val j = vc.getAlleleIndex(a1)
        val gt = if (i <= j)
          Genotype.gtIndex(i, j)
        else
          Genotype.gtIndex(j, i)
        gb.setGT(gt)

        if (pl != null && pl(gt) != 0) {
          // FIXME warn
          pl = pl.clone()
          pl(gt) = 0
        }
      }

      if (g.hasAD)
        gb.setAD(g.getAD)
      if (g.hasDP)
        gb.setDP(g.getDP)

      assert((pl != null) == g.hasPL)
      if (pl != null)
        gb.setPL(pl)

      // FIXME write htsjdk Genotype?
      gsb.write(gb)
    }

    (variant, gsb.result())
  }
}

object HtsjdkRecordReader {
  def apply(headerLines: Array[String]): HtsjdkRecordReader = {
    val codec = new htsjdk.variant.vcf.VCFCodec()
    codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
    new HtsjdkRecordReader(codec)
  }
}
