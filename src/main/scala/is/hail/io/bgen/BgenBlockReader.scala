package is.hail.io.bgen

import java.util.zip.Inflater

import is.hail.HailContext
import is.hail.annotations._
import is.hail.io._
import is.hail.io.gen.GenReport._
import is.hail.utils.{SharedIterable, SharedIterator}
import is.hail.variant.{DosageGenotype, Genotype, Variant}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred.FileSplit

class BgenRecord(compressed: Boolean,
  nSamples: Int,
  tolerance: Double) extends KeySerializedValueRecord[Variant, SharedIterable[Genotype]] {
  var ann: Annotation = _

  def setAnnotation(ann: Annotation) {
    this.ann = ann
  }

  def getAnnotation: Annotation = ann

  override def getValue: SharedIterable[Genotype] = {
    require(input != null, "called getValue before serialized value was set")

    val bytes =
      if (compressed) {
        val expansion = Array.ofDim[Byte](nSamples * 6)
        val inflater = new Inflater
        inflater.setInput(input)
        var off = 0
        while (off < expansion.length) {
          off += inflater.inflate(expansion, off, expansion.length - off)
        }
        expansion
      } else
        input

    assert(bytes.length == nSamples * 6)

    resetWarnings()

    val lowerTol = (32768 * (1.0 - tolerance) + 0.5).toInt
    val upperTol = (32768 * (1.0 + tolerance) + 0.5).toInt
    assert(lowerTol > 0)

    val noCall: Genotype = new DosageGenotype(-1, null)

    new SharedIterable[Genotype] {
      def iterator = new SharedIterator[Genotype] {
        var i = 0

        def hasNext: Boolean = i < bytes.length

        def next(): Genotype = {
          val d0 = (bytes(i) & 0xff) | ((bytes(i + 1) & 0xff) << 8)
          val d1 = (bytes(i + 2) & 0xff) | ((bytes(i + 3) & 0xff) << 8)
          val d2 = (bytes(i + 4) & 0xff) | ((bytes(i + 5) & 0xff) << 8)

          i += 6

          val dsum = d0 + d1 + d2
          if (dsum >= lowerTol) {
            if (dsum <= upperTol) {
              val px =
                if (dsum == 32768)
                  Array(d0, d1, d2)
                else
                  Genotype.weightsToLinear(d0, d1, d2)
              val gt = Genotype.unboxedGTFromLinear(px)
              new DosageGenotype(gt, px)
            } else {
              setWarning(dosageGreaterThanTolerance)
              noCall
            }
          } else {
            if (dsum == 0)
              setWarning(dosageNoCall)
            else
              setWarning(dosageLessThanTolerance)

            noCall
          }
        }
      }
    }
  }
}

class BgenBlockReader(job: Configuration, split: FileSplit) extends IndexedBinaryBlockReader[BgenRecord](job, split) {
  val file = split.getPath
  val bState = BgenLoader.readState(bfis)
  val indexPath = file + ".idx"
  val btree = new IndexBTree(indexPath, job)

  val tolerance = job.get("tolerance").toDouble

  seekToFirstBlockInSplit(split.getStart)

  override def createValue(): BgenRecord = new BgenRecord(bState.compressed, bState.nSamples, tolerance)

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
