package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes.interfaces.{SShuffleCode, SShuffleValue}
import is.hail.types.virtual._

abstract class PShuffle extends PType {
  def tShuffle: TShuffle

  def virtualType: TShuffle = tShuffle

  def loadLength(bAddress: Long): Int

  def loadLength(bAddress: Code[Long]): Code[Int]

  def bytesAddress(boff: Long): Long

  def bytesAddress(boff: Code[Long]): Code[Long]

  def storeLength(boff: Long, len: Int): Unit

  def storeLength(boff: Code[Long], len: Code[Int]): Code[Unit]

  def allocate(region: Region, length: Int): Long

  def allocate(region: Code[Region], length: Code[Int]): Code[Long]
}

abstract class PShuffleValue extends PValue with SShuffleValue {
  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]
}

abstract class PShuffleCode extends PCode with SShuffleCode {
  def pt: PShuffle

  def memoize(cb: EmitCodeBuilder, name: String): PShuffleValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PShuffleValue
}
