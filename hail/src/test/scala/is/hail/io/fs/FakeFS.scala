package is.hail.io.fs

case class FakeURL(path: String) extends FSURL {
  def getPath: String = path
  def getActualUrl: String = path
}

abstract class FakeFS extends FS {
  override type URL = FakeURL
  override def validUrl(filename: String): Boolean = ???
  override def parseUrl(filename: String): FakeURL = FakeURL(filename)
  override def urlAddPathComponent(url: FakeURL, component: String): FakeURL = ???
  override def openNoCompression(url: FakeURL): SeekableDataInputStream = ???
  override def createNoCompression(url: FakeURL): PositionedDataOutputStream = ???
  override def delete(url: FakeURL, recursive: Boolean): Unit = ???
  override def eTag(url: FakeURL): Option[String] = ???
  override def fileStatus(url: FakeURL): FileStatus = ???
  override def fileListEntry(url: FakeURL): FileListEntry = ???
  override def glob(url: FakeURL): Array[FileListEntry] = ???
  override def listDirectory(url: FakeURL): Array[FileListEntry] = ???
  override def makeQualified(path: String): String = ???
  override def getConfiguration(): Any = ???
  override def setConfiguration(config: Any): Unit = ???
}
