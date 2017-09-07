package is.hail.expr

import is.hail.asm4s.TypeInfo

package object ir {
  def Out1(x: IR) = new Out(Array(x))
}
