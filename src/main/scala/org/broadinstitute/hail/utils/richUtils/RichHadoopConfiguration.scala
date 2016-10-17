package org.broadinstitute.hail.utils.richUtils

import java.io._

import org.apache.hadoop
import org.apache.hadoop.fs.{FileStatus, PathIOException}
import org.apache.hadoop.io.IOUtils._
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.broadinstitute.hail.io.compress.BGzipCodec
import org.broadinstitute.hail.utils.{TextContext, WithContext, _}

import scala.io.Source

class RichHadoopConfiguration(val hConf: hadoop.conf.Configuration) extends AnyVal {

  def fileSystem(filename: String): hadoop.fs.FileSystem =
    new hadoop.fs.Path(filename).getFileSystem(hConf)

  private def create(filename: String): OutputStream = {
    val fs = fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    val os = fs.create(hPath)
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = codecFactory.getCodec(hPath)

    if (codec != null)
      codec.createOutputStream(os)
    else
      os
  }

  private def open(filename: String): InputStream = {
    val fs = fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    val is = fs.open(hPath)
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = codecFactory.getCodec(hPath)
    if (codec != null)
      codec.createInputStream(is)
    else
      is
  }

  def getFileSize(filename: String): Long = {
    val fs = fileSystem(filename)
    val hPath = new hadoop.fs.Path(filename)
    fs.getFileStatus(hPath).getLen
  }

  def exists(files: String*): Boolean = {
    files.forall(filename => fileSystem(filename).exists(new hadoop.fs.Path(filename)))
  }

  def mkDir(dirname: String) {
    fileSystem(dirname).mkdirs(new hadoop.fs.Path(dirname))
  }

  def delete(filename: String, recursive: Boolean) {
    fileSystem(filename).delete(new hadoop.fs.Path(filename), recursive)
  }

  def getTemporaryFile(tmpdir: String, nChar: Int = 10,
    prefix: Option[String] = None, suffix: Option[String] = None): String = {

    val destFS = fileSystem(tmpdir)
    val prefixString = if (prefix.isDefined) prefix + "-" else ""
    val suffixString = if (suffix.isDefined) "." + suffix else ""

    def getRandomName: String = {
      val randomName = tmpdir + "/" + prefixString + scala.util.Random.alphanumeric.take(nChar).mkString + suffixString
      val fileExists = destFS.exists(new hadoop.fs.Path(randomName))

      if (!fileExists)
        randomName
      else
        getRandomName
    }
    getRandomName
  }

  def globAll(filenames: Iterable[String]): Array[String] = {
    filenames.iterator
      .flatMap { arg =>
        val fss = globAndSort(arg)
        val files = fss.map(_.getPath.toString)
        if (files.isEmpty)
          warn(s"`$arg' refers to no files")
        files
      }.toArray
  }

  def globAndSort(filename: String): Array[FileStatus] = {
    val fs = fileSystem(filename)
    val path = new hadoop.fs.Path(filename)

    val files = fs.globStatus(path)
    if (files == null)
      return Array.empty[FileStatus]

    files.sortWith(_.compareTo(_) < 0)
  }

  def copy(src: String, dst: String) {
    hadoop.fs.FileUtil.copy(
      fileSystem(src), new hadoop.fs.Path(src),
      fileSystem(dst), new hadoop.fs.Path(dst),
      false, hConf)
  }

  def copyMerge(srcFilenames: Array[String], destFilename: String, deleteSource: Boolean = true) {
    val destPath = new hadoop.fs.Path(destFilename)
    val destFS = fileSystem(destFilename)

    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(destFilename)))
    val isBGzip = codec.exists(_.isInstanceOf[BGzipCodec])

    val srcFileStatuses = srcFilenames.flatMap(f => globAndSort(f))
    require(srcFileStatuses.forall {
      fileStatus => fileStatus.getPath != destPath && fileStatus.isFile
    })

    val outputStream = destFS.create(destPath)

    try {
      var i = 0
      while (i < srcFileStatuses.length) {
        val fileStatus = srcFileStatuses(i)
        val lenAdjust: Long = if (isBGzip && i < srcFileStatuses.length - 1)
          -28
        else
          0
        val srcFS = fileSystem(fileStatus.getPath.toString)
        val inputStream = srcFS.open(fileStatus.getPath)
        try {
          copyBytes(inputStream, outputStream,
            fileStatus.getLen + lenAdjust,
            false)
        } finally {
          inputStream.close()
        }
        i += 1
      }
    } finally {
      outputStream.close()
    }

    if (deleteSource) {
      srcFileStatuses.foreach {
        fileStatus => delete(fileStatus.getPath.toString, recursive = true)
      }
    }
  }

  def stripCodec(s: String): String = {
    val path = new org.apache.hadoop.fs.Path(s)

    Option(new CompressionCodecFactory(hConf)
      .getCodec(path))
      .map { codec =>
        val ext = codec.getDefaultExtension
        assert(s.endsWith(ext))
        s.dropRight(ext.length)
      }.getOrElse(s)
  }

  def writeObjectFile[T](filename: String)(f: (ObjectOutputStream) => T): T = {
    val oos = new ObjectOutputStream(create(filename))
    try {
      f(oos)
    } finally {
      oos.close()
    }
  }

  def fileStatus(filename: String): FileStatus = fileSystem(filename).getFileStatus(new hadoop.fs.Path(filename))

  def readObjectFile[T](filename: String)(f: (ObjectInputStream) => T): T = {
    val ois = new ObjectInputStream(open(filename))
    try {
      f(ois)
    } finally {
      ois.close()
    }
  }

  def readDataFile[T](filename: String)(f: (DataInputStream) => T): T = {
    val dis = new DataInputStream(open(filename))
    try {
      f(dis)
    } finally {
      dis.close()
    }
  }

  def writeTextFile[T](filename: String)(writer: (OutputStreamWriter) => T): T = {
    val oos = create(filename)
    val fw = new OutputStreamWriter(oos)
    try {
      writer(fw)
    } finally {
      fw.close()
    }
  }

  def writeDataFile[T](filename: String)(writer: (DataOutputStream) => T): T = {
    val oos = create(filename)
    val dos = new DataOutputStream(oos)
    try {
      writer(dos)
    } finally {
      dos.close()
    }
  }

  def readFile[T](filename: String)(reader: (InputStream) => T): T = {
    val is = open(filename)
    try {
      reader(is)
    } finally {
      is.close()
    }
  }

  def readLines[T](filename: String)(reader: (Iterator[WithContext[String]] => T)): T = {
    readFile[T](filename) {
      is =>
        val lines = Source.fromInputStream(is)
          .getLines()
          .zipWithIndex
          .map {
            case (value, position) =>
              val source = TextContext(value, filename, Some(position))
              WithContext(value, source)
          }
        reader(lines)
    }
  }

  def writeTable(filename: String, lines: Traversable[String], header: Option[String] = None) {
    writeTextFile(filename) {
      fw =>
        header.foreach { h =>
          fw.write(h)
          fw.write('\n')
        }
        lines.foreach { line =>
          fw.write(line)
          fw.write('\n')
        }
    }
  }
}
