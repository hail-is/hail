package is.hail.types.physical.stypes

import is.hail.asm4s.Code
import is.hail.types.physical.stypes.primitives.{SFloat32Code, SFloat64Code, SInt32Code, SInt64Code}

package object interfaces {

  def primitive(x: Code[Long]): SInt64Code = new SInt64Code(true, x)
  def primitive(x: Code[Int]): SInt32Code = new SInt32Code(true, x)
  def primitive(x: Code[Double]): SFloat64Code = new SFloat64Code(true, x)
  def primitive(x: Code[Float]): SFloat32Code = new SFloat32Code(true, x)
}
