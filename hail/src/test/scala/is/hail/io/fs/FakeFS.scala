package is.hail.io.fs

abstract class FakeFS extends FS {
  override type URL = HadoopFSURL
  override def validUrl(filename: String): Boolean = ???
  override def parseUrl(filename: String): HadoopFSURL = ???
  override def urlAddPathComponent(url: HadoopFSURL,component: String): HadoopFSURL = ???
  override def openNoCompression(url: HadoopFSURL): SeekableDataInputStream = ???
  override def createNoCompression(url: HadoopFSURL): PositionedDataOutputStream = ???
  override def delete(url: HadoopFSURL,recursive: Boolean): Unit = ???
  override def eTag(url: HadoopFSURL): Option[String] = ???
  override def fileListEntry(url: HadoopFSURL): FileListEntry = ???
  override def glob(url: HadoopFSURL): Array[FileListEntry] = ???
  override def listStatus(url: HadoopFSURL): Array[FileListEntry] = ???
  override def makeQualified(path: String): String = ???
  override def getConfiguration(): Any = ???
  override def setConfiguration(config: Any): Unit = ???
}
