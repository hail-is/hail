package is.hail.io

import is.hail.io.fs.{FS, FileStatus, PositionedDataOutputStream, SeekableDataInputStream}

abstract class FakeFS extends FS {
  override def openNoCompression(filename: String, _debug: Boolean): SeekableDataInputStream = ???

  override def createNoCompression(filename: String): PositionedDataOutputStream = ???

  override def delete(filename: String, recursive: Boolean): Unit = ???

  override def listStatus(filename: String): Array[FileStatus] = ???

  override def glob(filename: String): Array[FileStatus] = ???

  override def fileStatus(filename: String): FileStatus = ???

  override def makeQualified(path: String): String = ???

  override def getConfiguration(): Any = ???

  override def setConfiguration(config: Any): Unit = ???
}
