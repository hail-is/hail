package is.hail.types.physical.mtypes

trait MPointer {
  def byteSize: Long = 8

  def alignment: Long = 8

}
