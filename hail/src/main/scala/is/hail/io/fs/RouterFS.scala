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

  def openNoCompression(filename: String): SeekableDataInputStream = lookupFS(filename).openNoCompression(filename)

  def createNoCompression(filename: String): PositionedDataOutputStream = lookupFS(filename).createNoCompression(filename)

  override def readNoCompression(filename: String): Array[Byte] = lookupFS(filename).readNoCompression(filename)

  override def mkDir(dirname: String): Unit = lookupFS(dirname).mkDir(dirname)

  def delete(filename: String, recursive: Boolean) = lookupFS(filename).delete(filename, recursive)

  def listDirectory(filename: String): Array[FileListEntry] = lookupFS(filename).listDirectory(filename)

  def glob(filename: String): Array[FileListEntry] = lookupFS(filename).glob(filename)

  def fileListEntry(filename: String): FileListEntry = lookupFS(filename).fileListEntry(filename)

  override def eTag(filename: String): Option[String] = lookupFS(filename).eTag(filename)

  def makeQualified(path: String): String = lookupFS(path).makeQualified(path)

  def getConfiguration(): Any = fss.map(_.getConfiguration())

  def setConfiguration(config: Any): Unit = {
    fss.zip(config.asInstanceOf[IndexedSeq[_]]).foreach { case (fs: FS, config: Any) => fs.setConfiguration(config) }
  }
}
