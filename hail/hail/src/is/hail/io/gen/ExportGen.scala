package is.hail.io.gen

import is.hail.types.physical.{PString, PStruct}
import is.hail.utils._
import is.hail.variant.{Locus, VariantMethods, View}

object ExportGen {
  val spaceRegex = """\s+""".r

  def checkSample(id1: String, id2: String, missing: Double): Unit = {
    if (spaceRegex.findFirstIn(id1).isDefined)
      fatal(s"Invalid 'id1' found -- no white space allowed: '$id1'")
    if (spaceRegex.findFirstIn(id2).isDefined)
      fatal(s"Invalid 'id2' found -- no white space allowed: '$id2'")
    if (missing < 0 || missing > 1)
      fatal(s"'missing' values must be in the range [0, 1]. Found $missing for ($id1, $id2).")
  }

  def checkVariant(
    contig: String,
    position: Int,
    a0: String,
    a1: String,
    varid: String,
    rsid: String,
  ): Unit = {
    if (spaceRegex.findFirstIn(contig).isDefined)
      fatal(
        s"Invalid contig found at '${VariantMethods.locusAllelesToString(Locus(contig, position), Array(a0, a1))}' -- no white space allowed: '$contig'"
      )
    if (spaceRegex.findFirstIn(a0).isDefined)
      fatal(
        s"Invalid allele found at '${VariantMethods.locusAllelesToString(Locus(contig, position), Array(a0, a1))}' -- no white space allowed: '$a0'"
      )
    if (spaceRegex.findFirstIn(a1).isDefined)
      fatal(
        s"Invalid allele found at '${VariantMethods.locusAllelesToString(Locus(contig, position), Array(a0, a1))}' -- no white space allowed: '$a1'"
      )
    if (spaceRegex.findFirstIn(varid).isDefined)
      fatal(
        s"Invalid 'varid' found at '${VariantMethods.locusAllelesToString(Locus(contig, position), Array(a0, a1))}' -- no white space allowed: '$varid'"
      )
    if (spaceRegex.findFirstIn(rsid).isDefined)
      fatal(
        s"Invalid 'rsid' found at '${VariantMethods.locusAllelesToString(Locus(contig, position), Array(a0, a1))}' -- no white space allowed: '$rsid'"
      )
  }
}

class GenAnnotationView(rowType: PStruct) extends View {
  private val rsidField = rowType.fieldByName("rsid")
  private val varidField = rowType.fieldByName("varid")

  private val rsidIdx = rsidField.index
  private val varidIdx = varidField.index

  private var rsidOffset: Long = _
  private var varidOffset: Long = _

  private var cachedVarid: String = _
  private var cachedRsid: String = _

  def set(offset: Long): Unit = {
    assert(rowType.isFieldDefined(offset, varidIdx))
    assert(rowType.isFieldDefined(offset, rsidIdx))
    this.rsidOffset = rowType.loadField(offset, rsidIdx)
    this.varidOffset = rowType.loadField(offset, varidIdx)

    cachedVarid = null
    cachedRsid = null
  }

  def varid(): String = {
    if (cachedVarid == null)
      cachedVarid = varidField.typ.asInstanceOf[PString].loadString(varidOffset)
    cachedVarid
  }

  def rsid(): String = {
    if (cachedRsid == null)
      cachedRsid = rsidField.typ.asInstanceOf[PString].loadString(rsidOffset)
    cachedRsid
  }
}
