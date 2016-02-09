package org.broadinstitute.hail.io.hadoop

import java.io.{DataOutputStream, IOException}
import org.apache.hadoop.fs._
import org.apache.hadoop.io._
import org.apache.hadoop.mapred._
import org.apache.hadoop.util.Progressable

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

