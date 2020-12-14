package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SShuffle extends SType

trait SShuffleValue extends SValue {
  def loadLength()(implicit line: LineNumber): Code[Int]

  def loadBytes()(implicit line: LineNumber): Code[Array[Byte]]
}

trait SShuffleCode extends SCode {
  def memoize(cb: EmitCodeBuilder, name: String): SShuffleValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SShuffleValue
}
