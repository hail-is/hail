package is.hail.driver

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.io.compress.SplittableCompressionCodec
import org.apache.hadoop.mapreduce.{InputSplit => NewInputSplit, RecordReader => NewRecordReader, TaskAttemptContext => NewTaskAttemptContext}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat => NewFileInputFormat, FileSplit => NewFileSplit}
import is.hail.io.compress.BGzipInputStream
import is.hail.utils.richUtils.RichHadoopConfiguration
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

class CountBytesRecordReader extends NewRecordReader[LongWritable, LongWritable] {
  var first = true
  var start: Long = _
  var end: Long = _
  var in: FSDataInputStream = _
  var key: LongWritable = _
  var value: LongWritable = _
  var bgzCompressed: Boolean = _

  def initialize(genericSplit: NewInputSplit, context: NewTaskAttemptContext) {
    val split = genericSplit.asInstanceOf[NewFileSplit]
    println("split", split)
    val job = context.getConfiguration
    start = split.getStart
    end = start + split.getLength
    val file: Path = split.getPath

    val forceBGz = job.getBoolean("forceBGz", false)

    val path = file.toUri.getPath
    println("path", path)
    bgzCompressed = (path.endsWith(".gz") && forceBGz) || path.endsWith(".bgz")

    // open the file and seek to the start of the split
    val fs: FileSystem = file.getFileSystem(job)
    in = fs.open(file)
    in.seek(start)
  }

  def getPos: Long = in.getPos

  def nextKeyValue(): Boolean = {
    if (first) {
      first = false

      if (key == null)
        key = new LongWritable
      key.set(getPos)
      if (value == null)
        value = new LongWritable

      val buf = new Array[Byte](64 * 1024)
      var bytes = 0L

      if (bgzCompressed) {
        val cin = new BGzipInputStream(in, start, end, SplittableCompressionCodec.READ_MODE.BYBLOCK)
        def f() {
          if (cin.blockPos() < end) {
            val read = cin.readBlock(buf)
            if (read > 0) {
              bytes += read
              f()
            } else
              assert(read == -1)
          }
        }
        f()
      } else {
        def f() {
          val needed = end - (start + bytes)
          if (needed > 0) {
            val read = in.read(buf, 0, buf.length.toLong.min(needed).toInt)
            if (read > 0) {
              bytes += read
              f()
            } else
              assert(read == -1)
          }
        }
        f()
      }

      value.set(bytes)
      true
    } else {
      key = null
      value = null
      false
    }
  }

  def getCurrentKey = key

  def getCurrentValue = value

  def getProgress: Float =
    if (start == end)
      0.0f
    else
      Math.min(1.0f, (getPos - start).toFloat / (end - start))

  def close() {
    if (in != null)
      in.close()
  }
}

class CountBytesInputFormat extends NewFileInputFormat[LongWritable, LongWritable] {
  def isSplitable(fs: FileSystem, file: Path): Boolean = true

  def createRecordReader(genericSplit: NewInputSplit, context: NewTaskAttemptContext): NewRecordReader[LongWritable, LongWritable] = {
    new CountBytesRecordReader
  }
}

object CountBytes extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "--force-bgz", usage = "Force load .gz file using BGzip codec")
    var forceBGz: Boolean = false

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def name = "countbytes"

  def description = "Count number of bytes in file"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val jobConf = new Configuration(state.hadoopConf)

    jobConf.set("forceBGz", options.forceBGz.toString)
    val bytes =
      state.sc.newAPIHadoopFile[LongWritable, LongWritable, CountBytesInputFormat](options.arguments.asScala.mkString(","),
        classOf[CountBytesInputFormat],
        classOf[LongWritable],
        classOf[LongWritable],
        jobConf)
        .map { case (k, v) => v.get }
        .aggregate(0L)(_ + _, _ + _)

    println(s"bytes = $bytes")

    state
  }
}