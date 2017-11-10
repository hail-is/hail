package is.hail.io.plink

import is.hail.sparkextras._
import is.hail.expr._
import is.hail.annotations._
import is.hail.variant._

object ExportBedBimFam {

  val gtMap = Array(1, 3, 2, 0)

  def bedRowTransformer(nSamples: Int, rowType: TStruct): Iterator[RegionValue] => Iterator[Array[Byte]] = { it =>
    val hcv = HardCallView(rowType)
    val rvb = new RegionValueBuilder()
    val rv2 = RegionValue()
    val gtMap = Array(3, 2, 0)

    it.map { rv =>
      hcv.setRegion(rv)
      rvb.set(rv.region)

      val nBytes = (nSamples + 3) / 4
      val a = new Array[Byte](nBytes)
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
    val vIdx = rowType.fieldIdx("v")
    val t = rowType.fieldType(vIdx).asInstanceOf[TVariant].representation
    val tArrayAltAllele = TArray(TAltAllele.representation)
    val contigIdx = t.fieldIdx("contig")
    val startIdx = t.fieldIdx("start")
    val refIdx = t.fieldIdx("ref")
    val altAllelesIdx = t.fieldIdx("altAlleles")
    val altIdx = TAltAllele.representation.fieldIdx("alt")
    val vrv = RegionValue()

    it.map { rv =>
      val region = rv.region
      vrv.set(region, rowType.loadField(rv, vIdx))
      val contig = TString.loadString(region, t.loadField(vrv, contigIdx))
      val start = region.loadInt(t.loadField(vrv, startIdx))
      val ref = TString.loadString(region, t.loadField(vrv, refIdx))
      val alt = TString.loadString(region,
        TAltAllele.representation.loadField(
          region,
          tArrayAltAllele.loadElement(region, t.loadField(vrv, altAllelesIdx), 1, 0),
          altIdx))
      // FIXME: NO STRINGS, go directly through writePartitions
      val id = s"${contig}:${start}:${ref}:${alt}"
      s"""${contig}\t$id\t0\t${start}\t${alt}\t${ref}"""
    }
  }
}
