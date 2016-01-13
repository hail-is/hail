package org.broadinstitute.hail.io.hadoop

import java.io.{DataOutputStream, IOException, UnsupportedEncodingException}

import org.apache.hadoop.fs._
import org.apache.hadoop.io._
import org.apache.hadoop.io.compress.{CompressionCodec, GzipCodec}
import org.apache.hadoop.mapred._
import org.apache.hadoop.util.{Progressable, ReflectionUtils}

object LineRecordWriterNoNewlines {
  val utf8: String = "UTF-8"
}

class ByteArrayOutputFormat[K, V] extends FileOutputFormat[K, V] {

  class ByteArrayRecordWriter[K, V](out: DataOutputStream) extends RecordWriter[K, V] {

    @throws(classOf[IOException])
    private def writeObject(o: Any) {
      o match {
        case arr: BytesWritableUnseparated => arr.write(out)
        case _ => throw new IOException("passed improper input to ByteArrayRecordWriter")
      }
    }

    @throws(classOf[IOException])
    def write(key: K, value: V) {
      val nullKey: Boolean = key == null || key.isInstanceOf[NullWritable]
      val nullValue: Boolean = value == null || value.isInstanceOf[NullWritable]
      if (nullKey && nullValue) {
        return
      }
      if (!nullKey) {
        writeObject(key)
      }
      if (!nullValue) {
        writeObject(value)
      }
    }

    @throws(classOf[IOException])
    def close(reporter: Reporter) {
      out.close()
    }
  }

  @throws(classOf[IOException])
  override def getRecordWriter(ignored: FileSystem, job: JobConf,
    name: String, progress: Progressable): RecordWriter[K, V] = {
    val file: Path = FileOutputFormat.getTaskOutputPath(job, name)
    val fs: FileSystem = file.getFileSystem(job)
    val fileOut: FSDataOutputStream = fs.create(file, progress)
    new ByteArrayRecordWriter[K, V](fileOut)
  }
}

class TextOutputFormatNoNewlines[K, V] extends TextOutputFormat[K, V] {

  class LineRecordWriterNoNewlines[K, V](out: DataOutputStream, sep: String) extends RecordWriter[K, V] {
    private val keyValueSeparator: Array[Byte] = {
      try {
        sep.getBytes(LineRecordWriterNoNewlines.utf8)
      }
      catch {
        case uee: UnsupportedEncodingException =>
          throw new IllegalArgumentException("can't find " + LineRecordWriterNoNewlines.utf8 + " encoding")
      }
    }

    def this(out: DataOutputStream) {
      this(out, "\t")
    }

    @throws(classOf[IOException])
    private def writeObject(o: Any) {
      o match {
        case t: Text => out.write(t.getBytes, 0, t.getLength)
        case _ => out.write(o.toString.getBytes(LineRecordWriterNoNewlines.utf8))
      }
    }

    @throws(classOf[IOException])
    def write(key: K, value: V) {
      val nullKey: Boolean = key == null || key.isInstanceOf[NullWritable]
      val nullValue: Boolean = value == null || value.isInstanceOf[NullWritable]
      if (nullKey && nullValue) {
        return
      }
      if (!nullKey) {
        writeObject(key)
      }
      if (!(nullKey || nullValue)) {
        out.write(keyValueSeparator)
      }
      if (!nullValue) {
        writeObject(value)
      }
    }

    @throws(classOf[IOException])
    def close(reporter: Reporter) {
      out.close()
    }
  }

  @throws(classOf[IOException])
  override def getRecordWriter(ignored: FileSystem, job: JobConf,
    name: String, progress: Progressable): RecordWriter[K, V] = {

    val isCompressed: Boolean = FileOutputFormat.getCompressOutput(job)
    val keyValueSeparator: String = job.get("mapreduce.output.textoutputformat.separator", "\t")
    if (!isCompressed) {
      val file: Path = FileOutputFormat.getTaskOutputPath(job, name)
      val fs: FileSystem = file.getFileSystem(job)
      val fileOut: FSDataOutputStream = fs.create(file, progress)
      new LineRecordWriterNoNewlines[K, V](fileOut, keyValueSeparator)
    }
    else {
      val codecClass: Class[_ <: CompressionCodec] = FileOutputFormat.getOutputCompressorClass(job, classOf[GzipCodec])
      val codec: CompressionCodec = ReflectionUtils.newInstance(codecClass, job)
      val file: Path = FileOutputFormat.getTaskOutputPath(job, name + codec.getDefaultExtension)
      val fs: FileSystem = file.getFileSystem(job)
      val fileOut: FSDataOutputStream = fs.create(file, progress)
      new LineRecordWriterNoNewlines[K, V](new DataOutputStream(codec.createOutputStream(fileOut)), keyValueSeparator)
    }
  }
}
