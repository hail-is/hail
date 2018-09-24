package is.hail.io.plink

import java.io.{OutputStream, OutputStreamWriter}

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.expr.ir.MatrixValue
import is.hail.expr.types._
import is.hail.variant._
import is.hail.utils._
import org.apache.spark.TaskContext

object ExportPlink {
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
      fatal(s"Invalid contig found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$contig'")
    if (spaceRegex.findFirstIn(a0).isDefined)
      fatal(s"Invalid allele found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$a0'")
    if (spaceRegex.findFirstIn(a1).isDefined)
      fatal(s"Invalid allele found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$a1'")
    if (spaceRegex.findFirstIn(varid).isDefined)
      fatal(s"Invalid 'varid' found at '${ VariantMethods.locusAllelesToString(v.locus(), v.alleles()) }' -- no white space allowed: '$varid'")

    osw.write(contig)
    osw.write('\t')
    osw.write(varid)
    osw.write('\t')
    osw.write(a.cmPosition().toString)
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

  def apply(mv: MatrixValue, path: String): Unit = {
    val hc = HailContext.get
    val sc = hc.sc
    val hConf = hc.hadoopConf

    val tmpBedDir = hConf.getTemporaryFile(hc.tmpDir)
    val tmpBimDir = hConf.getTemporaryFile(hc.tmpDir)

    hConf.mkDir(tmpBedDir)
    hConf.mkDir(tmpBimDir)

    val sHConfBc = sc.broadcast(new SerializableHadoopConfiguration(hConf))

    val nPartitions = mv.rvd.getNumPartitions
    val d = digitsNeeded(nPartitions)

    val nSamples = mv.colValues.value.length
    val fullRowType = mv.typ.rvRowType

    val nRecordsWritten = mv.rvd.mapPartitionsWithIndex { (i, ctx, it) =>
      val hConf = sHConfBc.value.value
      val f = partFile(d, i, TaskContext.get)
      val bedPartPath = tmpBedDir + "/" + f
      val bimPartPath = tmpBimDir + "/" + f
      var rowCount = 0L

      hConf.writeTextFile(bimPartPath) { bimOS =>
        hConf.writeFile(bedPartPath) { bedOS =>
          val v = new RegionValueVariant(fullRowType)
          val a = new BimAnnotationView(fullRowType)
          val hcv = HardCallView(fullRowType)
          val bp = new BitPacker(2, bedOS)

          it.foreach { rv =>
            v.setRegion(rv)
            a.setRegion(rv)
            ExportPlink.writeBimRow(v, a, bimOS)

            hcv.setRegion(rv)
            ExportPlink.writeBedRow(hcv, bp, nSamples)
            ctx.region.clear()
            rowCount += 1
          }
        }
      }

      Iterator.single(rowCount)
    }.collect().sum

    hConf.writeFile(tmpBedDir + "/_SUCCESS")(out => ())
    hConf.writeFile(tmpBedDir + "/header")(out => out.write(ExportPlink.bedHeader))
    hConf.copyMerge(tmpBedDir, path + ".bed", nPartitions, header = true)

    hConf.writeTextFile(tmpBimDir + "/_SUCCESS")(out => ())
    hConf.copyMerge(tmpBimDir, path + ".bim", nPartitions, header = false)

    mv.colsTableValue.export(path + ".fam", header = false)

    info(s"wrote $nRecordsWritten variants and $nSamples samples to '$path'")
  }
}

class BimAnnotationView(rowType: TStruct) extends View {
  private val varidField = rowType.fieldByName("varid")
  private val cmPosField = rowType.fieldByName("cm_position")

  private val varidIdx = varidField.index
  private val cmPosIdx = cmPosField.index

  private var region: Region = _
  private var varidOffset: Long = _
  private var cmPosOffset: Long = _

  private var cachedVarid: String = _

  def setRegion(region: Region, offset: Long) {
    this.region = region

    assert(rowType.isFieldDefined(region, offset, varidIdx))
    assert(rowType.isFieldDefined(region, offset, cmPosIdx))

    this.varidOffset = rowType.loadField(region, offset, varidIdx)
    this.cmPosOffset = rowType.loadField(region, offset, cmPosIdx)

    cachedVarid = null
  }

  def cmPosition(): Double =
    region.loadDouble(cmPosOffset)

  def varid(): String = {
    if (cachedVarid == null)
      cachedVarid = TString.loadString(region, varidOffset)
    cachedVarid
  }
}

class BitPacker(nBitsPerItem: Int, os: OutputStream) extends Serializable {
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
      os.write(data.toByte)
      data = data >>> 8
      nBitsStaged -= 8
    }
  }

  def flush() {
    if (nBitsStaged > 0)
      os.write(data.toByte)
    data = 0L
    nBitsStaged = 0
  }
}
