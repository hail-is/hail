package is.hail.utils

case class Muple[T, U](var _1: T, var _2: U) {
  def set(newLeft: T, newRight: U): Muple[T, U] = {
    _1 = newLeft
    _2 = newRight
    this
  }
}
