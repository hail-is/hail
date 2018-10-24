package is.hail

package object cxx {

  var symCounter: Long = 0

  def genSym(name: String): String = {
    symCounter += 1
    s"${name}_$symCounter"
  }

  type Code = String
  type Type = String

}
