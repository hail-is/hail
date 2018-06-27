package is.hail.io.gen

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.expr.ir.MatrixValue
import is.hail.expr.types.{TString, TStruct}
import is.hail.io.plink.BimAnnotationView
import is.hail.variant.{ArrayGenotypeView, RegionValueVariant, VariantMethods, View}
import is.hail.utils._
import org.apache.spark.sql.Row

object ExportGen {
  val spaceRegex = """\s+""".r

  def apply(mv: MatrixValue, path: String, precision: Int = 4) {
    val hc = HailContext.get
    val hConf = hc.hadoopConf

    hConf.writeTable(path + ".sample",
      "ID_1 ID_2 missing\n0 0 0" +: mv.colValues.value.map { a =>
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
    val fullRowType = mv.typ.rvRowType

    mv.rvd.mapPartitions { it =>
      val sb = new StringBuilder
      val gpView = new ArrayGenotypeView(fullRowType)
      val v = new RegionValueVariant(fullRowType)
      val va = new GenAnnotationView(fullRowType)

      it.map { rv =>
        gpView.setRegion(rv)
        v.setRegion(rv)
        va.setRegion(rv)

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
    }.writeTable(path + ".gen", hc.tmpDir, None)
  }
}

class GenAnnotationView(rowType: TStruct) extends View {
  private val rsidField = rowType.fieldByName("rsid")
  private val varidField = rowType.fieldByName("varid")

  private val rsidIdx = rsidField.index
  private val varidIdx = varidField.index

  private var region: Region = _
  private var rsidOffset: Long = _
  private var varidOffset: Long = _

  private var cachedVarid: String = _
  private var cachedRsid: String = _

  def setRegion(region: Region, offset: Long) {
    this.region = region

    assert(rowType.isFieldDefined(region, offset, varidIdx))
    assert(rowType.isFieldDefined(region, offset, rsidIdx))
    this.rsidOffset = rowType.loadField(region, offset, rsidIdx)
    this.varidOffset = rowType.loadField(region, offset, varidIdx)

    cachedVarid = null
    cachedRsid = null
  }

  def varid(): String = {
    if (cachedVarid == null)
      cachedVarid = TString.loadString(region, varidOffset)
    cachedVarid
  }

  def rsid(): String = {
    if (cachedRsid == null)
      cachedRsid = TString.loadString(region, rsidOffset)
    cachedRsid
  }
}
