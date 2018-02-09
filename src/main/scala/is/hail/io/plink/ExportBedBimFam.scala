package is.hail.io.plink

import is.hail.utils._
import is.hail.expr.types._
import is.hail.annotations._
import is.hail.variant._

object ExportBedBimFam {

  val gtMap = Array(3, 2, 0)

  def bedRowTransformer(nSamples: Int, rowType: TStruct): Iterator[RegionValue] => Iterator[Array[Byte]] = { it =>
    val hcv = HardCallView(rowType)
    val rv2 = RegionValue()
    val gtMap = Array(3, 2, 0)
    val nBytes = (nSamples + 3) / 4
    val a = new Array[Byte](nBytes)

    it.map { rv =>
      hcv.setRegion(rv)

      var b = 0
      var k = 0
      while (k < nSamples) {
        hcv.setGenotype(k)
        val gt = if (hcv.hasGT) gtMap(hcv.getGT) else 1
        b |= gt << ((k & 3) * 2)
        if ((k & 3) == 3) {
          a(k >> 2) = b.toByte
          b = 0
        }
        k += 1
      }
      if ((k & 3) > 0)
        a(nBytes - 1) = b.toByte

      // FIXME: NO BYTE ARRAYS, go directly through writePartitions
      a
    }
  }

  def bimRowTransformer(rowType: TStruct): Iterator[RegionValue] => Iterator[String] = { it =>
    val v = new RegionValueVariant(rowType)

    it.map { rv =>
      v.setRegion(rv)
      val contig = v.contig()
      val start = v.position()
      val alleles = v.alleles()
      if (alleles.length != 2)
        fatal(s"expected 2 alleles, found ${alleles.length} at $contig:$start:${alleles.tail.mkString(",")}")
      // FIXME: NO STRINGS, go directly through writePartitions
      val id = s"${contig}:${start}:${alleles(0)}:${alleles(1)}"
      s"""${contig}\t$id\t0\t${start}\t${alleles(1)}\t${alleles(0)}"""
    }
  }
}
