package is.hail.io.plink

import java.io.OutputStreamWriter
import is.hail.annotations.Region
import is.hail.expr.types._
import is.hail.variant._
import is.hail.utils._

object ExportPlinkUtils {
  val bedHeader = Array[Byte](108, 27, 1)
  val gtMap = Array(3, 2, 0)
  val spaceRegex = """\s+""".r

  def writeBimRow(v: RegionValueVariant, a: BimAnnotationView, osw: OutputStreamWriter): Unit = {
    val contig = v.contig()
    val position = v.position()
    val alleles = v.alleles()
    val a0 = alleles(0)
    val a1 = alleles(1)
    val varid = a.varid()

    if (spaceRegex.findFirstIn(contig).isDefined)
      fatal(s"Invalid 'contig' found -- no white space allowed: '$contig'")
    if (spaceRegex.findFirstIn(a0).isDefined)
      fatal(s"Invalid allele found at locus '$contig:$position' -- no white space allowed: '$a0'")
    if (spaceRegex.findFirstIn(a1).isDefined)
      fatal(s"Invalid allele found at locus '$contig:$position' -- no white space allowed: '$a1'")
    if (spaceRegex.findFirstIn(varid).isDefined)
      fatal(s"Invalid 'varid' found at locus '$contig:$position' -- no white space allowed: '$varid'")

    osw.write(contig)
    osw.write('\t')
    osw.write(varid)
    osw.write('\t')
    osw.write(a.positionMorgan().toString)
    osw.write('\t')
    osw.write(position.toString)
    osw.write('\t')
    osw.write(a1)
    osw.write('\t')
    osw.write(a0)
    osw.write('\n')
  }

  def writeBedRow(hcv: HardCallView, bp: BitPacker, nSamples: Int): Unit = {
    var k = 0
    while (k < nSamples) {
      hcv.setGenotype(k)
      val gt = if (hcv.hasGT) gtMap(Call.unphasedDiploidGtIndex(hcv.getGT)) else 1
      bp += gt
      k += 1
    }
    bp.flush()
  }
}

class BimAnnotationView(rowType: TStruct) extends View {
  private val varidField = rowType.fieldByName("varid")
  private val posMorganField = rowType.fieldByName("pos_morgan")
  private val varidIdx = varidField.index
  private val posMorganIdx = posMorganField.index
  private val tvarid = varidField.typ.asInstanceOf[TString]
  private val tposm = posMorganField.typ.asInstanceOf[TInt32]
  private var region: Region = _
  private var varidOffset: Long = _
  private var posMorganOffset: Long = _

  private var cachedVarid: String = _

  def setRegion(region: Region, offset: Long) {
    this.region = region

    assert(rowType.isFieldDefined(region, offset, varidIdx))
    assert(rowType.isFieldDefined(region, offset, posMorganIdx))
    this.varidOffset = rowType.loadField(region, offset, varidIdx)
    this.posMorganOffset = rowType.loadField(region, offset, posMorganIdx)

    cachedVarid = null
  }

  def positionMorgan(): Int =
    region.loadInt(posMorganOffset)

  def varid(): String = {
    if (cachedVarid == null)
      cachedVarid = TString.loadString(region, varidOffset)
    cachedVarid
  }
}

class BitPacker(nBitsPerItem: Int, consume: (Int) => Unit) extends Serializable {
  require(nBitsPerItem > 0)

  private val bitMask = (1L << nBitsPerItem) - 1
  private var data = 0L
  private var nBitsStaged = 0

  def +=(i: Int) {
    data |= ((i.toUIntFromRep.toLong & bitMask) << nBitsStaged)
    nBitsStaged += nBitsPerItem
    write()
  }

  private def write() {
    while (nBitsStaged >= 8) {
      consume(data.toByte)
      data = data >>> 8
      nBitsStaged -= 8
    }
  }

  def flush() {
    if (nBitsStaged > 0)
      consume(data.toByte)
    data = 0L
    nBitsStaged = 0
  }
}
