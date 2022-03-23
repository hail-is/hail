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

object SRNGState {
  val staticTweakMask = 0L
  val dynTweakMask = 1L << 63
  val finalBlockNoPadTweak = -1L >>> 1
  val finalBlockPaddedTweak = -1L
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
      Array.fill[Long](4)(0),
      Bitstring(""),
      0,
      0)
  }
}

final case class SRNGStateValue(
  st: SRNGState,
  dynBlocksSum: IndexedSeq[Value[Long]],
  lastDynBlock: IndexedSeq[Value[Long]],
  staticBlocksSum: IndexedSeq[Long],
  lastStaticBlock: Bitstring,
  numStaticBlocks: Int,
  numDynBlocks: Int,
) extends SValue {
  assert(staticBlocksSum.length == 4)
  assert(lastStaticBlock.numWords <= 4)
  assert(dynBlocksSum.length == 4)
  assert(lastDynBlock.length == st.numWordsInLastDynBlock)

  override def valueTuple: IndexedSeq[Value[_]] =
    dynBlocksSum ++ lastDynBlock

  override def sizeToStoreInBytes(cb: EmitCodeBuilder) = ???

  def splitStatic(bitstring: Bitstring): SRNGStateValue = {
    val appendedBlock = lastStaticBlock ++ bitstring
    if (appendedBlock.length < 256) {
      return copy(lastStaticBlock = appendedBlock)
    }
    val (fullBlock, newLastStaticBlock) = appendedBlock.popWords(4)
    Threefry.encrypt(st.key, numStaticBlocks.toLong & SRNGState.staticTweakMask, fullBlock)
    for (i <- fullBlock.indices) fullBlock(i) ^= staticBlocksSum(i)
    copy(
      staticBlocksSum = fullBlock,
      lastStaticBlock = newLastStaticBlock,
      numStaticBlocks = numStaticBlocks + 1)
  }

  def splitDyn(cb: EmitCodeBuilder, idx: Value[Long]): SRNGStateValue = {
    if (st.numWordsInLastDynBlock < 4) {
      return copy(
        st = st.copy(numWordsInLastDynBlock = st.numWordsInLastDynBlock + 1),
        lastDynBlock = lastDynBlock :+ idx)
    }
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"splitDyn_x$i", lastDynBlock(i)))
    Threefry.encrypt(cb, st.key, numDynBlocks.toLong & SRNGState.dynTweakMask, x)
    for (i <- 0 until 4) cb.assign(x(i), x(i) ^ dynBlocksSum(i))
    copy(
      st = st.copy(numWordsInLastDynBlock = 1),
      dynBlocksSum = x,
      lastDynBlock = Array(idx),
      numDynBlocks = numDynBlocks + 1)
  }

  def rand(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    val finalStaticBlocksSum = if (lastStaticBlock.length == 0) {
      staticBlocksSum
    } else {
      val padded = lastStaticBlock.padTo(4)
      Threefry.encrypt(st.key, numStaticBlocks.toLong & SRNGState.staticTweakMask, padded)
      Array.tabulate(4)(i => staticBlocksSum(i) ^ padded(i)).toFastIndexedSeq
    }
    if (st.numWordsInLastDynBlock == 0) {
      val x = Array(finalStaticBlocksSum: _*)
      Threefry.encrypt(st.key, SRNGState.finalBlockNoPadTweak, x)
      return x.map(const)
    }
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"rand_x$i", finalStaticBlocksSum(i)))
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
