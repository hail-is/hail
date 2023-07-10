package is.hail.io.fs

class RouterFS(fss: IndexedSeq[FS]) extends FS {
  // This is never actually used
  type URL = LocalFSURL

  def lookupFS(path: String): FS = {
    fss.find(_.validUrl(path)) match {
      case Some(fs) => fs
      case None => throw new IllegalArgumentException(s"Unsupported URI: $path")
    }
  }

  override def validUrl(filename: String): Boolean =
    fss.exists(_.validUrl(filename))

  override def openCachedNoCompression(filename: String): SeekableDataInputStream = lookupFS(filename).openCachedNoCompression(filename)

  override def createCachedNoCompression(filename: String): PositionedDataOutputStream = lookupFS(filename).createCachedNoCompression(filename)

  def openNoCompression(filename: String, _debug: Boolean = false): SeekableDataInputStream = lookupFS(filename).openNoCompression(filename, _debug)

  def createNoCompression(filename: String): PositionedDataOutputStream = lookupFS(filename).createNoCompression(filename)

  override def readNoCompression(filename: String): Array[Byte] = lookupFS(filename).readNoCompression(filename)

  override def mkDir(dirname: String): Unit = lookupFS(dirname).mkDir(dirname)

  def delete(filename: String, recursive: Boolean) = lookupFS(filename).delete(filename, recursive)

  def listStatus(filename: String): Array[FileStatus] = lookupFS(filename).listStatus(filename)

  def glob(filename: String): Array[FileStatus] = lookupFS(filename).glob(filename)

  def fileStatus(filename: String): FileStatus = lookupFS(filename).fileStatus(filename)

  override def fileChecksum(filename: String): Array[Byte] = lookupFS(filename).fileChecksum(filename)

  def makeQualified(path: String): String = lookupFS(path).makeQualified(path)

  def getConfiguration(): Any = fss.map(_.getConfiguration())

  def setConfiguration(config: Any): Unit = {
    fss.zip(config.asInstanceOf[IndexedSeq[_]]).foreach { case (fs: FS, config: Any) => fs.setConfiguration(config) }
  }
}
