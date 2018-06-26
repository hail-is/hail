package is.hail.io.bgen

import is.hail.utils._
import is.hail.io._
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

class BgenInputFormatV12 extends IndexedBinaryInputFormat[BgenRecordV12] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecordV12] = {
    reporter.setStatus(split.toString)
    new BgenBlockReaderV12(job, split.asInstanceOf[BgenV12InputSplit])
  }

  override def getSplits(job: JobConf, numSplits: Int): Array[InputSplit] = {
    val splits = super.getSplits(job, numSplits)
    splits.flatMap { x =>
      val split = x.asInstanceOf[FileSplit]
      val path = split.getPath
      val positions = job.get(LoadBgen.includedVariantsPositionsHadoopPrefix + path)
      val indices = job.get(LoadBgen.includedVariantsIndicesHadoopPrefix + path)
      assert(positions == null && indices == null ||
        positions != null && indices != null)
      if (positions != null) {
        val decodedPositions = LoadBgen.decodeLongs(positions)
        val (keptIndices, keptPositions) = LoadBgen.decodeInts(indices).zip(decodedPositions)
          .filter { case (_, x) =>
            split.getStart <= x && x < split.getStart + split.getLength
        } .unzip

        log.info(s"kept ${keptPositions.length} for split $x")

        if (keptPositions.isEmpty)
          None
        else
          Some(new BgenV12InputSplit(split, keptIndices, keptPositions))
      } else {
        log.info(s"no variant filters found for $path")
        Some(new BgenV12InputSplit(split, null, null))
      }
    }
  }
}

class BgenV12InputSplit(
  var fileSplit: FileSplit,
  var keptIndices: Array[Int],
  var keptPositions: Array[Long]
) extends InputSplit {
  require(keptIndices == null && keptPositions == null ||
    keptIndices.length == keptPositions.length)
  def this() = this(null, null, null)
  def hasFilter: Boolean = keptIndices != null
  def getPath(): Path = fileSplit.getPath()
  def getStart(): Long = fileSplit.getStart()
  def getLength(): Long = fileSplit.getLength()
  def getLocations(): Array[String] = fileSplit.getLocations()
  def readFields(in: java.io.DataInput): Unit = {
    fileSplit = new FileSplit(new org.apache.hadoop.mapreduce.lib.input.FileSplit())
    fileSplit.readFields(in)
    val indicesLen = in.readInt()
    if (indicesLen != -1) {
      var i = 0
      keptIndices = new Array[Int](indicesLen)
      while (i < indicesLen) {
        keptIndices(i) = in.readInt()
        i += 1
      }
    }
    val positionsLen = in.readInt()
    if (positionsLen != -1) {
      var i = 0
      keptPositions = new Array[Long](positionsLen)
      while (i < positionsLen) {
        keptPositions(i) = in.readLong()
        i += 1
      }
    }
  }
  def write(out: java.io.DataOutput): Unit = {
    fileSplit.write(out)
    if (keptIndices == null)
      out.writeInt(-1)
    else {
      out.writeInt(keptIndices.length)
      var i = 0
      while (i < keptIndices.length) {
        out.writeInt(keptIndices(i))
        i += 1
      }
    }
    if (keptPositions == null)
      out.writeInt(-1)
    else {
      out.writeInt(keptPositions.length)
      var i = 0
      while (i < keptPositions.length) {
        out.writeLong(keptPositions(i))
        i += 1
      }
    }
  }
  override def toString(): String =
    s"BgenV12InputSplit($fileSplit, $keptPositions)"
}
