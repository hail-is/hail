package is.hail.io.fs

object RouterFSURL {
  def apply(fs: FS)(_url: fs.URL): RouterFSURL = RouterFSURL(_url, fs)
}

case class RouterFSURL private (_url: FSURL, val fs: FS) extends FSURL {
  val url = _url.asInstanceOf[fs.URL]

  def getPath: String = url.getPath
  def addPathComponent(component: String): RouterFSURL = {
    RouterFSURL(fs)(fs.urlAddPathComponent(url, component))
  }
  override def toString(): String = url.toString
}

class RouterFS(fss: IndexedSeq[FS]) extends FS {
  type URL = RouterFSURL

  def lookupFS(path: String): FS = {
    fss.find(_.validUrl(path)) match {
      case Some(fs) => fs
      case None => throw new IllegalArgumentException(s"Unsupported URI: $path")
    }
  }

  override def parseUrl(filename: String): URL = {
    val fs = lookupFS(filename)

    RouterFSURL(fs)(fs.parseUrl(filename))
  }

  override def validUrl(filename: String): Boolean =
    fss.exists(_.validUrl(filename))

  def urlAddPathComponent(url: URL, component: String): URL = url.addPathComponent(component)

  override def openCachedNoCompression(url: URL): SeekableDataInputStream = url.fs.openCachedNoCompression(url.url)

  override def createCachedNoCompression(url: URL): PositionedDataOutputStream = url.fs.createCachedNoCompression(url.url)

  def openNoCompression(url: URL): SeekableDataInputStream = url.fs.openNoCompression(url.url)

  def createNoCompression(url: URL): PositionedDataOutputStream = url.fs.createNoCompression(url.url)

  override def readNoCompression(url: URL): Array[Byte] = url.fs.readNoCompression(url.url)

  override def mkDir(url: URL): Unit = url.fs.mkDir(url.url)

  def delete(url: URL, recursive: Boolean) = url.fs.delete(url.url, recursive)

  def listDirectory(url: URL): Array[FileListEntry] = url.fs.listDirectory(url.url)

  def glob(url: URL): Array[FileListEntry] = url.fs.glob(url.url)

  def fileStatus(url: URL): FileStatus = url.fs.fileStatus(url.url)

  def fileListEntry(url: URL): FileListEntry = url.fs.fileListEntry(url.url)

  override def eTag(url: URL): Option[String] = url.fs.eTag(url.url)

  def makeQualified(path: String): String = lookupFS(path).makeQualified(path)

  def getConfiguration(): Any = fss.map(_.getConfiguration())

  def setConfiguration(config: Any): Unit = {
    fss.zip(config.asInstanceOf[IndexedSeq[_]]).foreach { case (fs: FS, config: Any) => fs.setConfiguration(config) }
  }
}
