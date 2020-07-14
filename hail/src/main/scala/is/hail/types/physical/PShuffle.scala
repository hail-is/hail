package is.hail.types.physical

import is.hail.asm4s._
import is.hail.types.virtual._
import is.hail.expr.ir._

abstract class PShuffle extends ComplexPType {
  def tShuffle: TShuffle

  def virtualType: TShuffle = tShuffle
}

abstract class PShuffleValue extends PValue {
  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]
}

abstract class PShuffleCode extends PCode {
  def pt: PShuffle

  def memoize(cb: EmitCodeBuilder, name: String): PShuffleValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PShuffleValue
}
