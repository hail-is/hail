package is.hail.io.plink

import is.hail.utils._
import is.hail.variant._

import java.io.OutputStream

object ExportPlink {
  val bedHeader = Array[Byte](108, 27, 1)
  val spaceRegex = """\s+""".r

  def checkVariant(contig: String, varid: String, position: Int, a0: String, a1: String): Unit = {
    def locus: Locus = Locus(contig, position)
    def alleles: Array[String] = Array(a0, a1)

    if (spaceRegex.findFirstIn(contig).isDefined)
      fatal(
        s"Invalid contig found at '${VariantMethods.locusAllelesToString(locus, alleles)}' -- no white space allowed: '$contig'"
      )
    if (spaceRegex.findFirstIn(a0).isDefined)
      fatal(
        s"Invalid allele found at '${VariantMethods.locusAllelesToString(locus, alleles)}' -- no white space allowed: '$a0'"
      )
    if (spaceRegex.findFirstIn(a1).isDefined)
      fatal(
        s"Invalid allele found at '${VariantMethods.locusAllelesToString(locus, alleles)}' -- no white space allowed: '$a1'"
      )
    if (spaceRegex.findFirstIn(varid).isDefined)
      fatal(
        s"Invalid 'varid' found at '${VariantMethods.locusAllelesToString(locus, alleles)}' -- no white space allowed: '$varid'"
      )
  }
}

class BitPacker(nBitsPerItem: Int, os: OutputStream) extends Serializable {
  require(nBitsPerItem > 0)

  private val bitMask = (1L << nBitsPerItem) - 1
  private var data = 0L
  private var nBitsStaged = 0

  def add(i: Int) = {
    data |= ((i & 0xffffffffL & bitMask) << nBitsStaged)
    nBitsStaged += nBitsPerItem
    write()
  }

  def +=(i: Int) = add(i)

  private def write(): Unit =
    while (nBitsStaged >= 8) {
      os.write(data.toByte)
      data = data >>> 8
      nBitsStaged -= 8
    }

  def flush(): Unit = {
    if (nBitsStaged > 0)
      os.write(data.toByte)
    data = 0L
    nBitsStaged = 0
  }
}
