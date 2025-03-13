package is.hail.io

case class FileWriteMetadata(path: String, rowsWritten: Long, bytesWritten: Long) {
  def render(): String = s"$path\t$rowsWritten\t$bytesWritten"
}
