package is.hail.io.bgen

import is.hail.utils._
import is.hail.io._
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

class BgenInputFormatV12 extends IndexedBinaryInputFormat[BgenRecordV12] {
  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[LongWritable,
    BgenRecordV12] = {
    reporter.setStatus(split.toString)
    new BgenBlockReaderV12(job, split.asInstanceOf[BgenV12InputSplit])
  }

  private[this] val indices = new java.util.concurrent.ConcurrentHashMap[String, IndexBTree2]()
  private[this] def indexFor(indexPath: String, job: JobConf, nVariants: Int): IndexBTree2 =
    indices.computeIfAbsent(
      indexPath,
      { (path: String) => new IndexBTree2(path, job, nVariants) })

  override def getSplits(job: JobConf, numSplits: Int): Array[InputSplit] = {
    val nVariants = job.get("nVariants").toInt
    val splits = super.getSplits(job, numSplits)
    splits.flatMap { x =>
      val split = x.asInstanceOf[FileSplit]
      val path = split.getPath
      val indexPath = path + ".idx"
      val index = indexFor(indexPath, job, nVariants)
      val s = job.get("__"+path.toString.replaceAllLiterally("file:",""))
      if (s != null) {
        val keptPositions = LoadBgen.decodeInts(s)
          .sorted
          .map(index.positionOfVariant _)
          .filter(x =>
          split.getStart <= x && x < split.getStart + split.getLength)

        if (keptPositions.isEmpty)
          None
        else
          Some(new BgenV12InputSplit(split, keptPositions))
      }
      else {
        Some(new BgenV12InputSplit(split, null))
      }
    }
  }
}

class BgenV12InputSplit(
  var fileSplit: FileSplit,
  var keptPositions: Array[Long]
) extends InputSplit {
  def this() = this(null, null)
  def getLength(): Long = fileSplit.getLength()
  def getLocations(): Array[String] = fileSplit.getLocations()
  def readFields(in: java.io.DataInput): Unit = {
    fileSplit = new FileSplit(new org.apache.hadoop.mapreduce.lib.input.FileSplit())
    fileSplit.readFields(in)
    val len = in.readInt()
    if (len != -1) {
      var i = 0
      keptPositions = new Array[Long](len)
      while (i < len) {
        keptPositions(i) = in.readLong()
        i += 1
      }
    }
  }
  def write(out: java.io.DataOutput): Unit = {
    fileSplit.write(out)
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
