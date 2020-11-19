package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.Code
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.PShuffle
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SShuffle extends SType

trait SShuffleValue extends SValue {
  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]
}

trait SShuffleCode extends SCode {
  def pt: PShuffle

  def memoize(cb: EmitCodeBuilder, name: String): SShuffleValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SShuffleValue
}
