package is.hail.io.bgen

import java.util.zip.Inflater

import is.hail.annotations._
import is.hail.io._
import is.hail.io.gen.GenReport._
import is.hail.variant.{Genotype, GenotypeBuilder, GenotypeStreamBuilder, Variant}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

object BgenRecord {
  val dosageDivisor: Double = 32768.0
}

class BgenRecord(compressed: Boolean,
  gb: GenotypeBuilder,
  gsb: GenotypeStreamBuilder,
  nSamples: Int,
  tolerance: Double) extends KeySerializedValueRecord[Variant, Iterable[Genotype]] {
  var ann: Annotation = _

  def setAnnotation(ann: Annotation) {
    this.ann = ann
  }

  def getAnnotation: Annotation = ann

  override def getValue: Iterable[Genotype] = {
    require(input != null, "called getValue before serialized value was set")

    val bytes = {
      if (compressed) {
        val expansion = Array.ofDim[Byte](nSamples * 6)
        val inflater = new Inflater
        inflater.setInput(input)
        while (!inflater.finished()) {
          inflater.inflate(expansion)
        }
        expansion
      } else
        input
    }

    assert(bytes.length == nSamples * 6)

    resetWarnings()

    gsb.clear()
    val bar = new ByteArrayReader(bytes)

    for (_ <- 0 until nSamples) {
      gb.clear()

      val d0 = bar.readShort()
      val d1 = bar.readShort()
      val d2 = bar.readShort()
      val dosageSum = (d0 + d1 + d2) / BgenRecord.dosageDivisor
      if (dosageSum == 0.0)
        setWarning(dosageNoCall)
      else if (1d - dosageSum > tolerance)
        setWarning(dosageLessThanTolerance)
      else if (dosageSum - 1d > tolerance)
        setWarning(dosageGreaterThanTolerance)
      else {
        val px = Genotype.weightsToLinear(d0, d1, d2)
        val gt = Genotype.gtFromLinear(px)
        gt.foreach(gb.setGT)
        gb.setPX(px)
      }

      gsb.write(gb)
    }
    gsb.result()
  }
}

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[BgenRecord](job, split) {
  val file = split.getPath
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val compressGS = job.getBoolean("compressGS", false)
  val tolerance = job.get("tolerance").toDouble

  val gb = new GenotypeBuilder(2, isDosage = true)
  val gsb = new GenotypeStreamBuilder(2, isDosage = true, compress = compressGS)

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): BgenRecord = new BgenRecord(bState.compressed, gb, gsb, bState.nSamples, tolerance)

  def seekToFirstBlockInSplit(start: Long) {
    pos = btree.queryIndex(start) match {
      case Some(x) => x
      case None => end
    }

    btree.close()
    bfis.seek(pos)
  }

  def next(key: LongWritable, value: BgenRecord): Boolean = {
    if (pos >= end)
      false
    else {
      val nRow = bfis.readInt()

      // we silently assumed this in previous iterations of the code.  Now explicitly assume.
      assert(nRow == bState.nSamples, "row nSamples is not equal to header nSamples")

      val lid = bfis.readLengthAndString(2)
      val rsid = bfis.readLengthAndString(2)
      val chr = bfis.readLengthAndString(2)
      val position = bfis.readInt()

      val ref = bfis.readLengthAndString(4)
      val alt = bfis.readLengthAndString(4)

      val recodedChr = chr match {
        case "23" => "X"
        case "24" => "Y"
        case "25" => "X"
        case "26" => "MT"
        case x => x
      }

      val variant = Variant(recodedChr, position, ref, alt)

      val bytesInput = if (bState.compressed) {
        val compressedBytes = bfis.readInt()
        bfis.readBytes(compressedBytes)
      } else
        bfis.readBytes(nRow * 6)

      value.setKey(variant)
      value.setAnnotation(Annotation(rsid, lid))
      value.setSerializedValue(bytesInput)

      pos = bfis.getPosition
      true
    }
  }
}
