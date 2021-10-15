package is.hail.io.gen

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.MatrixValue
import is.hail.types.physical.{PString, PStruct}
import is.hail.variant.{ArrayGenotypeView, RegionValueVariant, VariantMethods, View}
import is.hail.utils._
import org.apache.spark.sql.Row

object ExportGen {
  val spaceRegex = """\s+""".r

  def apply(ctx: ExecuteContext, mv: MatrixValue, path: String, precision: Int = 4) {
    val fs = ctx.fs

    fs.writeTable(path + ".sample",
      "ID_1 ID_2 missing\n0 0 0" +: mv.colValues.javaValue.map { a =>
        val r = a.asInstanceOf[Row]
        assert(r.length == 3)

        val id1 = r.get(0).asInstanceOf[String]
        val id2 = r.get(1).asInstanceOf[String]
        val missing = r.get(2).asInstanceOf[Double]

        if (spaceRegex.findFirstIn(id1).isDefined)
          fatal(s"Invalid 'id1' found -- no white space allowed: '$id1'")
        if (spaceRegex.findFirstIn(id2).isDefined)
          fatal(s"Invalid 'id2' found -- no white space allowed: '$id2'")
        if (missing < 0 || missing > 1)
          fatal(s"'missing' values must be in the range [0, 1]. Found $missing for ($id1, $id2).")

        s"$id1 $id2 $missing"
      }.toArray)

    val localNSamples = mv.nCols
    val fullRowType = mv.rvRowPType

    mv.rvd.mapPartitions { (ctx, it) =>
      val sb = new StringBuilder
      val gpView = new ArrayGenotypeView(fullRowType)
      val v = new RegionValueVariant(fullRowType)
      val va = new GenAnnotationView(fullRowType)

      it.map { ptr =>
        gpView.set(ptr)
        v.set(ptr)
        va.set(ptr)

        val contig = v.contig()
        val alleles = v.alleles()
        val a0 = alleles(0)
        val a1 = alleles(1)

        val varid = va.varid()
        val rsid = va.rsid()
        
        if (spaceRegex.findFirstIn(contig).isDefined)
          fatal(s"Invalid contig found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$contig'")
        if (spaceRegex.findFirstIn(a0).isDefined)
          fatal(s"Invalid allele found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$a0'")
        if (spaceRegex.findFirstIn(a1).isDefined)
          fatal(s"Invalid allele found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$a1'")
        if (spaceRegex.findFirstIn(varid).isDefined)
          fatal(s"Invalid 'varid' found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$varid'")
        if (spaceRegex.findFirstIn(rsid).isDefined)
          fatal(s"Invalid 'rsid' found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$rsid'")


        sb.clear()
        sb.append(contig)
        sb += ' '
        sb.append(varid)
        sb += ' '
        sb.append(rsid)
        sb += ' '
        sb.append(v.position())
        sb += ' '
        sb.append(a0)
        sb += ' '
        sb.append(a1)

        var i = 0
        while (i < localNSamples) {
          gpView.setGenotype(i)
          if (gpView.hasGP) {
            if (gpView.getGPLength() != 3)
              fatal(s"Invalid 'gp' at variant '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' and sample index $i. The array must have length equal to 3.")
            sb += ' '
            sb.append(formatDouble(gpView.getGP(0), precision))
            sb += ' '
            sb.append(formatDouble(gpView.getGP(1), precision))
            sb += ' '
            sb.append(formatDouble(gpView.getGP(2), precision))
          } else
            sb.append(" 0 0 0")
          i += 1
        }
        sb.result()
      }
    }.writeTable(ctx, path + ".gen", None)
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

  def set(offset: Long) {
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
