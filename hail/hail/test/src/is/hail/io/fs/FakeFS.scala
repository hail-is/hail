package is.hail.io.fs

class FakeURL(override val path: String) extends FSURL[FakeURL] {
  override def /(component: String): FakeURL =
    new FakeURL(f"$path/$component")
}

abstract class FakeFS extends FS {
  override type URL = FakeURL
  override def validUrl(filename: String): Boolean = ???
  override def parseUrl(filename: String): FakeURL = new FakeURL(filename)
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
