package is.hail.asm4s

import org.objectweb.asm.tree._
import scala.collection.generic.Growable

abstract class StagedBitSet {

  private var used = 0
  private var bits: Settable[Long] = null
  private var count = 0

  def getNewVar: Settable[Long]

  def getCount: Int = count

  def newBit(mb: MethodBuilder): SettableBit = {
    if (used >= 64 || bits == null) {
      bits = getNewVar
      count += 1
      mb.emit(bits.store(0L))
      used = 0
    }

    used += 1
    new SettableBit(bits, used - 1)
  }
}

class LocalBitSet(mb: MethodBuilder) extends StagedBitSet {
  def this(fb: FunctionBuilder[_]) =
    this(fb.apply_method)

  def getNewVar: Settable[Long] = mb.newLocal[Long](s"settable$getCount")

  def newBit(): SettableBit =
    newBit(mb)
}

class ClassBitSet(fb: FunctionBuilder[_]) extends StagedBitSet {
  def getNewVar: Settable[Long] = fb.newField[Long](s"settable$getCount")
}

class SettableBit(bits: Settable[Long], i: Int) extends Settable[Boolean] {
  assert(i >= 0)
  assert(i < 64)

  def store(b: Code[Boolean]): Code[Unit] = {
    bits := bits & ~(1L << i) | (b.toI.toL << i)
  }

  def load(): Code[Boolean] = (bits >> i & 1L).toI.toZ
}
