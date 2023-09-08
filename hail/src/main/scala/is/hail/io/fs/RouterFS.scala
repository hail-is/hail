package is.hail.io.fs

import scala.language.existentials

case class RouterFSURL(
  routerFs: RouterFS,
  url: FSURL
) extends FSURL {
  def getPath: String = url.getPath
  def addPathComponent(component: String): RouterFSURL = {
    RouterFSURL(routerFs, url.addPathComponent(component))
  }
  def fromString(s: String): RouterFSURL = {
    RouterFSURL(routerFs, url.fromString(s))
  }

  override def toString(): String = url.toString
}

class RouterFS(fss: IndexedSeq[FS]) extends FS {
  type URL = RouterFSURL

  def parseUrl(filename: String): URL = {
    val fs = lookupFS(filename)
    val newUrl = fs.parseUrl(filename)
    RouterFSURL(this, newUrl)
  }

  def lookupFS(path: String): FS = {
    fss.find(_.validUrl(path)) match {
      case Some(fs) => fs
      case None => throw new IllegalArgumentException(s"Unsupported URI: $path")
    }
  }

  def validUrl(filename: String): Boolean = {
    fss.exists(_.validUrl(filename))
  }

  override def openCachedNoCompression(filename: String): SeekableDataInputStream = lookupFS(filename).openCachedNoCompression(filename)

  override def createCachedNoCompression(filename: String): PositionedDataOutputStream = lookupFS(filename).createCachedNoCompression(filename)

  def openNoCompression(filename: String): SeekableDataInputStream = lookupFS(filename).openNoCompression(filename)

  def createNoCompression(filename: String): PositionedDataOutputStream = lookupFS(filename).createNoCompression(filename)

  override def readNoCompression(filename: String): Array[Byte] = lookupFS(filename).readNoCompression(filename)

  override def mkDir(dirname: String): Unit = lookupFS(dirname).mkDir(dirname)

  def delete(filename: String, recursive: Boolean) = lookupFS(filename).delete(filename, recursive)

  def listDirectory(filename: String): Array[FileListEntry] = lookupFS(filename).listDirectory(filename)

  def glob(filename: String): Array[FileListEntry] = lookupFS(filename).glob(filename)

  override def fileStatus(filename: String): FileStatus = lookupFS(filename).fileStatus(filename)

  override def fileStatus(url: URL): FileStatus = lookupFS(url.toString).fileStatus(url.toString)

  override def getFileListEntry(filename: String): FileListEntry = lookupFS(filename).getFileListEntry(filename)

  override def getFileListEntry(url: URL): FileListEntry = lookupFS(url.toString).getFileListEntry(url.toString)

  def makeQualified(path: String): String = lookupFS(path).makeQualified(path)

  def getConfiguration(): Any = fss.map(_.getConfiguration())

  def setConfiguration(config: Any): Unit = {
    fss.zip(config.asInstanceOf[IndexedSeq[_]]).foreach { case (fs: FS, config: Any) => fs.setConfiguration(config) }
  }
}
