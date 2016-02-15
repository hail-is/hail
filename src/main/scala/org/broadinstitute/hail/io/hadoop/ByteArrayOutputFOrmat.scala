package org.broadinstitute.hail.io.hadoop

import java.io.{DataOutputStream, IOException}
import org.apache.hadoop.fs._
import org.apache.hadoop.io._
import org.apache.hadoop.mapred._
import org.apache.hadoop.util.Progressable

class ByteArrayOutputFormat extends FileOutputFormat[NullWritable, BytesWritableUnseparated] {

  class ByteArrayRecordWriter(out: DataOutputStream) extends RecordWriter[NullWritable, BytesWritableUnseparated] {

    def write(key: NullWritable, value: BytesWritableUnseparated) {
      if (value != null)
        value.write(out)
    }

    def close(reporter: Reporter) {
      out.close()
    }
  }

  override def getRecordWriter(ignored: FileSystem, job: JobConf,
    name: String, progress: Progressable): RecordWriter[NullWritable, BytesWritableUnseparated] = {
    val file: Path = FileOutputFormat.getTaskOutputPath(job, name)
    val fs: FileSystem = file.getFileSystem(job)
    val fileOut: FSDataOutputStream = fs.create(file, progress)
    new ByteArrayRecordWriter(fileOut)
  }
}

