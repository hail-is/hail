package is.hail.io.fs

abstract class FakeFS extends FS {
  override type URL = LocalFSURL
  override def validUrl(filename: String): Boolean = ???
  override def openNoCompression(filename: String): SeekableDataInputStream = ???
  override def createNoCompression(filename: String): PositionedDataOutputStream = ???
  override def delete(filename: String, recursive: Boolean): Unit = ???
  override def listStatus(filename: String): Array[FileStatus] = ???
  override def glob(filename: String): Array[FileStatus] = ???
  override def fileStatus(filename: String): FileStatus = ???
  override def eTag(filename: String): Option[String] = ???
  override def makeQualified(path: String): String = ???
  override def getConfiguration(): Any = ???
  override def setConfiguration(config: Any): Unit = ???
}