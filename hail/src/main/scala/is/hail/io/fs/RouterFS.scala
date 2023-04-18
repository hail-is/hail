package is.hail.io.fs

class RouterFS(schemes: Map[String, FS], default: String) extends FS {

  def lookupFS(path: String): FS = {
    val uri = new java.net.URI(path)
    schemes.getOrElse(uri.getScheme, schemes(default))
  }

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

  def makeQualified(path: String): String = lookupFS(path).makeQualified(path)

  def getConfiguration(): Any = schemes.map { case (k, v) => (k, v.getConfiguration()) }

  def setConfiguration(config: Any): Unit = {
    config.asInstanceOf[Map[_, _]].foreach { case (k: String, v: Any) => schemes(k).setConfiguration(v) }
  }
}
