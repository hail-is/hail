package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, Threefry}
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.virtual.{TRNGState, Type}
import is.hail.utils.{Bitstring, toRichIterable}

import scala.collection.mutable

import scala.collection.mutable

object SRNGState {
  val staticTweak = -1L
  val finalBlockNoPadTweak = -2L
  val finalBlockPaddedTweak = -3L
}

final case class SRNGState(
  key: IndexedSeq[Long],
  numWordsInLastDynBlock: Int
) extends SType {
  assert(key.length == 4)
  assert(numWordsInLastDynBlock <= 4 && numWordsInLastDynBlock >= 0)

  def virtualType: Type = TRNGState

  override protected[stypes] def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = ???

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] =
    Array.fill(4 + numWordsInLastDynBlock)(typeInfo[Long])

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = ???

  override def fromValues(values: IndexedSeq[Value[_]]): SValue = ???

  override def storageType(): PType = ???

  override def copiedType: SType = ???

  override def castRename(t: Type): SType = ???

  override protected[stypes] def _typeWithRequiredness: TypeWithRequiredness = ???

  override def containsPointers: Boolean = false
}

object SRNGStateValue {
  def apply(cb: EmitCodeBuilder, key: IndexedSeq[Long]): SRNGStateValue = {
    val typ = SRNGState(key, 0)
    new SRNGStateValue(
      typ,
      Array.fill[Value[Long]](4)(0),
      Array[Value[Long]](),
      false,
      0)
  }
}

final case class SRNGStateValue(
  st: SRNGState,
  runningSum: IndexedSeq[Value[Long]],
  lastDynBlock: IndexedSeq[Value[Long]],
  hasStaticSplit: Boolean,
  numDynBlocks: Int,
) extends SValue {
  assert(runningSum.length == 4)
  assert(lastDynBlock.length == st.numWordsInLastDynBlock)

  override def valueTuple: IndexedSeq[Value[_]] =
    runningSum ++ lastDynBlock

  override def sizeToStoreInBytes(cb: EmitCodeBuilder) = ???

  def splitStatic(cb: EmitCodeBuilder, idx: Long): SRNGStateValue = {
    assert(!hasStaticSplit)
    val x = Array.ofDim[Long](4)
    x(0) = idx
    Threefry.encrypt(st.key, SRNGState.staticTweak, x)
    val newDynBlocksSum = Array.tabulate[Value[Long]](4)(i => cb.memoize(runningSum(i) ^ x(i)))
    copy(
      runningSum = newDynBlocksSum,
      hasStaticSplit = true)
  }

  def splitDyn(cb: EmitCodeBuilder, idx: Value[Long]): SRNGStateValue = {
    if (st.numWordsInLastDynBlock < 4) {
      return copy(
        st = st.copy(numWordsInLastDynBlock = st.numWordsInLastDynBlock + 1),
        lastDynBlock = lastDynBlock :+ idx)
    }
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"splitDyn_x$i", lastDynBlock(i)))
    Threefry.encrypt(cb, st.key, numDynBlocks.toLong, x)
    for (i <- 0 until 4) cb.assign(x(i), x(i) ^ runningSum(i))
    copy(
      st = st.copy(numWordsInLastDynBlock = 1),
      runningSum = x,
      lastDynBlock = Array(idx),
      numDynBlocks = numDynBlocks + 1)
  }

  def rand(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"rand_x$i", runningSum(i)))
    if (st.numWordsInLastDynBlock == 4) {
      for (i <- lastDynBlock.indices) cb.assign(x(i), x(i) ^ lastDynBlock(i))
      Threefry.encrypt(cb, st.key, SRNGState.finalBlockNoPadTweak, x)
    } else {
      for (i <- lastDynBlock.indices) cb.assign(x(i), x(i) ^ lastDynBlock(i))
      cb.assign(x(lastDynBlock.size), x(lastDynBlock.size) ^ (1L << 63))
      Threefry.encrypt(cb, st.key, SRNGState.finalBlockPaddedTweak, x)
    }
    x
  }
}
