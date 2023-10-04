package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, Threefry, ThreefryRandomEngine}
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PType, StoredSTypePType}
import is.hail.types.virtual.{TRNGState, Type}
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.utils.FastSeq

final case class SRNGStateStaticInfo(numWordsInLastBlock: Int, hasStaticSplit: Boolean, numDynBlocks: Int) {
  assert(numWordsInLastBlock <= 4 && numWordsInLastBlock >= 0)
}

final case class SRNGState(staticInfo: Option[SRNGStateStaticInfo]) extends SType {
  def virtualType: Type = TRNGState

  override protected[stypes] def _typeWithRequiredness: TypeWithRequiredness =
    RPrimitive()

  override def containsPointers: Boolean = false

  override protected[stypes] def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue =
    value.st match {
      case SRNGState(_) => value
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = staticInfo match {
    case None =>
      Array.fill(8)(typeInfo[Long]) ++ Array(typeInfo[Int], typeInfo[Boolean], typeInfo[Int])
    case Some(info) =>
      Array.fill(4 + info.numWordsInLastBlock)(typeInfo[Long])
  }

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SRNGStateSettable = staticInfo match {
    case None =>
      new SCanonicalRNGStateSettable(
        this,
        settables.slice(0, 4).asInstanceOf[IndexedSeq[Settable[Long]]],
        settables.slice(4, 8).asInstanceOf[IndexedSeq[Settable[Long]]],
        coerce[Int](settables(8)),
        coerce[Boolean](settables(9)),
        coerce[Int](settables(10)))
    case Some(_) =>
      new SRNGStateStaticSizeSettable(
        this,
        settables.slice(0, 4).asInstanceOf[IndexedSeq[Settable[Long]]],
        settables.drop(4).asInstanceOf[IndexedSeq[Settable[Long]]])
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SRNGStateValue = staticInfo match {
    case None =>
      new SCanonicalRNGStateValue(
        this,
        values.slice(0, 4).asInstanceOf[IndexedSeq[Settable[Long]]],
        values.slice(4, 8).asInstanceOf[IndexedSeq[Settable[Long]]],
        coerce[Int](values(8)),
        coerce[Boolean](values(9)),
        coerce[Int](values(10)))
    case Some(info) =>
      new SRNGStateStaticSizeValue(
        this,
        values.slice(0, 4).asInstanceOf[IndexedSeq[Settable[Long]]],
        values.drop(4).asInstanceOf[IndexedSeq[Settable[Long]]])
  }

  override def storageType(): PType = StoredSTypePType(this, false)

  override def copiedType: SType = ???

  override def castRename(t: Type): SType = ???
}

trait SRNGStateValue extends SValue {
  override def st: SRNGState
  def splitStatic(cb: EmitCodeBuilder, idx: Long): SRNGStateValue
  def splitDyn(cb: EmitCodeBuilder, idx: Value[Long]): SRNGStateValue
  def rand(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]
  def copyIntoEngine(cb: EmitCodeBuilder, tf: Value[ThreefryRandomEngine]): Unit
}

trait SRNGStateSettable extends SRNGStateValue with SSettable

object SCanonicalRNGStateValue {
  def apply(cb: EmitCodeBuilder): SCanonicalRNGStateValue = {
    val typ = SRNGState(None)
    new SCanonicalRNGStateValue(
      typ,
      Array.fill[Value[Long]](4)(0),
      Array.fill[Value[Long]](4)(0),
      0, false, 0)
  }
}

class SCanonicalRNGStateValue(
  override val st: SRNGState,
  val runningSum: IndexedSeq[Value[Long]],
  val lastDynBlock: IndexedSeq[Value[Long]],
  val numWordsInLastBlock: Value[Int],
  val hasStaticSplit: Value[Boolean],
  val numDynBlocks: Value[Int]
) extends SRNGStateValue {

  override def valueTuple: IndexedSeq[Value[_]] =
    runningSum ++
      lastDynBlock ++
      FastSeq(numWordsInLastBlock, hasStaticSplit, numDynBlocks)

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value =
    new SInt64Value(4*8 + 4*8 + 4 + 4 + 4)

  def splitStatic(cb: EmitCodeBuilder, idx: Long): SCanonicalRNGStateValue = {
    cb.ifx(hasStaticSplit, cb._fatal("RNGState received two static splits"))
    val x = Array.ofDim[Long](4)
    x(0) = cb.emb.ctx.rngNonce
    x(1) = idx
    val key = Threefry.defaultKey
    Threefry.encrypt(key, Array(Threefry.staticTweak, 0L), x)
    val newDynBlocksSum = Array.tabulate[Value[Long]](4)(i => cb.memoize(runningSum(i) ^ x(i)))

    new SCanonicalRNGStateValue(st, newDynBlocksSum, lastDynBlock, numWordsInLastBlock, const(true), numDynBlocks)
  }

  def splitDyn(cb: EmitCodeBuilder, idx: Value[Long]): SCanonicalRNGStateValue = {
    val newRunningSum = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"splitDyn_x$i", runningSum(i)))
    val newLastDynBlock = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"splitDyn_m$i", lastDynBlock(i)))
    val newNumWordsInLastBlock = cb.newLocal[Int](s"splitDyn_numWords", numWordsInLastBlock)
    val newNumDynBlocks = cb.newLocal[Int](s"splitDyn_numBlocks", numDynBlocks)

    cb.ifx(numWordsInLastBlock < 4, {
      cb.switch(numWordsInLastBlock,
        cb._fatal("invalid numWordsInLastBlock"),
        for {i <- 0 until 4}
          yield () => cb.assign(newLastDynBlock(i), idx)
      )
      cb.assign(newNumWordsInLastBlock, newNumWordsInLastBlock + 1)
    }, {
      val key = Threefry.defaultKey
      Threefry.encrypt(cb, key, Array(cb.memoize(numDynBlocks.toL), const(0L)), newLastDynBlock)
      for (i <- 0 until 4) cb.assign(newRunningSum(i), newRunningSum(i) ^ newLastDynBlock(i))
      cb.assign(newLastDynBlock(0), idx)
      for (i <- 1 until 4) cb.assign(newLastDynBlock(i), 0L)
      cb.assign(newNumWordsInLastBlock, 1)
      cb.assign(newNumDynBlocks, newNumDynBlocks + 1)
    })

    new SCanonicalRNGStateValue(st, newRunningSum, newLastDynBlock, newNumWordsInLastBlock, hasStaticSplit, newNumDynBlocks)
  }

  def rand(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    cb.ifx(!hasStaticSplit, cb._fatal("RNGState never received static split"))
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"rand_x$i", runningSum(i)))
    val key = Threefry.defaultKey
    val finalTweak = cb.memoize((numWordsInLastBlock ceq 4).mux(Threefry.finalBlockNoPadTweak, Threefry.finalBlockPaddedTweak))
    for (i <- 0 until 4) cb.assign(x(i), x(i) ^ lastDynBlock(i))
    cb.switch(numWordsInLastBlock,
      cb._fatal("invalid numWordsInLastBlock"),
      for {i <- 0 until 4}
        yield () => cb.assign(x(i), x(i) ^ 1L)
    )
    Threefry.encrypt(cb, key, Array(finalTweak, const(0L)), x)
    x
  }

  def copyIntoEngine(cb: EmitCodeBuilder, tf: Value[ThreefryRandomEngine]): Unit = {
    cb.ifx(!hasStaticSplit, cb._fatal("RNGState never received static split"))
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"cie_x$i", runningSum(i)))
    val finalTweak = (numWordsInLastBlock ceq 4).mux(Threefry.finalBlockNoPadTweak, Threefry.finalBlockPaddedTweak)
    for (i <- 0 until 4) cb.assign(x(i), x(i) ^ lastDynBlock(i))
    cb.switch(numWordsInLastBlock,
      cb._fatal("invalid numWordsInLastBlock"),
      for {i <- 0 until 4}
        yield () => cb.assign(x(i), x(i) ^ 1L)
    )
    cb += tf.invoke[Long, Long, Long, Long, Long, Unit]("resetState", x(0), x(1), x(2), x(3), finalTweak)
  }
}

class SCanonicalRNGStateSettable(
  st: SRNGState,
  override val runningSum: IndexedSeq[Settable[Long]],
  override val lastDynBlock: IndexedSeq[Settable[Long]],
  override val numWordsInLastBlock: Settable[Int],
  override val hasStaticSplit: Settable[Boolean],
  override val numDynBlocks: Settable[Int]
) extends SCanonicalRNGStateValue(st, runningSum, lastDynBlock, numWordsInLastBlock, hasStaticSplit, numDynBlocks)
  with SRNGStateSettable {
  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SCanonicalRNGStateValue =>
      (runningSum, v.runningSum).zipped.foreach { (x, s) => cb.assign(x, s) }
      (lastDynBlock, v.lastDynBlock).zipped.foreach { (x, s) => cb.assign(x, s) }
      cb.assign(numWordsInLastBlock, v.numWordsInLastBlock)
      cb.assign(hasStaticSplit, v.hasStaticSplit)
      cb.assign(numDynBlocks, v.numDynBlocks)
  }

  override def settableTuple(): IndexedSeq[Settable[_]] =
    runningSum ++
      lastDynBlock ++
      FastSeq(numWordsInLastBlock, hasStaticSplit, numDynBlocks)
}

object SRNGStateStaticSizeValue {
  def apply(cb: EmitCodeBuilder): SRNGStateStaticSizeValue = {
    val typ = SRNGState(Some(SRNGStateStaticInfo(0, false, 0)))
    new SRNGStateStaticSizeValue(
      typ,
      Array.fill[Value[Long]](4)(0),
      Array[Value[Long]]())
  }
}

class SRNGStateStaticSizeValue(
  override val st: SRNGState,
  val runningSum: IndexedSeq[Value[Long]],
  val lastDynBlock: IndexedSeq[Value[Long]]
) extends SRNGStateValue {
  val staticInfo = st.staticInfo.get
  assert(runningSum.length == 4)
  assert(lastDynBlock.length == staticInfo.numWordsInLastBlock)

  override def valueTuple: IndexedSeq[Value[_]] =
    runningSum ++ lastDynBlock

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value =
    new SInt64Value(4*8 + staticInfo.numWordsInLastBlock*8)

  def splitStatic(cb: EmitCodeBuilder, idx: Long): SRNGStateStaticSizeValue = {
    assert(!staticInfo.hasStaticSplit)
    val x = Array.ofDim[Long](4)
    x(0) = cb.emb.ctx.rngNonce
    x(1) = idx
    val key = Threefry.defaultKey
    Threefry.encrypt(key, Array(Threefry.staticTweak, 0L), x)
    val newDynBlocksSum = Array.tabulate[Value[Long]](4)(i => cb.memoize(runningSum(i) ^ x(i)))

    new SRNGStateStaticSizeValue(
      st = SRNGState(Some(SRNGStateStaticInfo(staticInfo.numWordsInLastBlock, true, staticInfo.numDynBlocks))),
      runningSum = newDynBlocksSum,
      lastDynBlock = lastDynBlock)
  }

  def splitDyn(cb: EmitCodeBuilder, idx: Value[Long]): SRNGStateStaticSizeValue = {
    if (staticInfo.numWordsInLastBlock < 4) {
      return new SRNGStateStaticSizeValue(
        st = SRNGState(Some(SRNGStateStaticInfo(staticInfo.numWordsInLastBlock + 1, staticInfo.hasStaticSplit, staticInfo.numDynBlocks))),
        runningSum = runningSum,
        lastDynBlock = lastDynBlock :+ idx
      )
    }
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"splitDyn_x$i", lastDynBlock(i)))
    val key = Threefry.defaultKey
    Threefry.encrypt(cb, key, Array(const(staticInfo.numDynBlocks.toLong), const(0L)), x)
    for (i <- 0 until 4) cb.assign(x(i), x(i) ^ runningSum(i))

    new SRNGStateStaticSizeValue(
      st = SRNGState(Some(SRNGStateStaticInfo(1, staticInfo.hasStaticSplit, staticInfo.numDynBlocks + 1))),
      runningSum = x,
      lastDynBlock = Array(idx)
    )
  }

  def rand(cb: EmitCodeBuilder): IndexedSeq[Value[Long]] = {
    assert(staticInfo.hasStaticSplit)
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"rand_x$i", runningSum(i)))
    val key = Threefry.defaultKey
    if (staticInfo.numWordsInLastBlock == 4) {
      for (i <- lastDynBlock.indices) cb.assign(x(i), x(i) ^ lastDynBlock(i))
      Threefry.encrypt(cb, key, Array(const(Threefry.finalBlockNoPadTweak), const(0L)), x)
    } else {
      for (i <- lastDynBlock.indices) cb.assign(x(i), x(i) ^ lastDynBlock(i))
      cb.assign(x(lastDynBlock.size), x(lastDynBlock.size) ^ 1L)
      Threefry.encrypt(cb, key, Array(const(Threefry.finalBlockPaddedTweak), const(0L)), x)
    }
    x
  }

  def copyIntoEngine(cb: EmitCodeBuilder, tf: Value[ThreefryRandomEngine]): Unit = {
    val x = Array.tabulate[Settable[Long]](4)(i => cb.newLocal[Long](s"cie_x$i", runningSum(i)))
    for (i <- lastDynBlock.indices) cb.assign(x(i), x(i) ^ lastDynBlock(i))
    val finalTweak = if (staticInfo.numWordsInLastBlock < 4) {
      cb.assign(x(lastDynBlock.size), x(lastDynBlock.size) ^ 1L)
      Threefry.finalBlockPaddedTweak
    } else {
      Threefry.finalBlockNoPadTweak
    }
    cb += tf.invoke[Long, Long, Long, Long, Long, Unit]("resetState", x(0), x(1), x(2), x(3), finalTweak)
  }
}

class SRNGStateStaticSizeSettable(
  st: SRNGState,
  override val runningSum: IndexedSeq[Settable[Long]],
  override val lastDynBlock: IndexedSeq[Settable[Long]]
) extends SRNGStateStaticSizeValue(st, runningSum, lastDynBlock) with SRNGStateSettable {
  override def store(cb: EmitCodeBuilder, v: SValue): Unit = v match {
    case v: SRNGStateStaticSizeValue =>
      (runningSum, v.runningSum).zipped.foreach { (x, s) => cb.assign(x, s) }
      (lastDynBlock, v.lastDynBlock).zipped.foreach { (x, s) => cb.assign(x, s) }
  }

  override def settableTuple(): IndexedSeq[Settable[_]] =
    runningSum ++ lastDynBlock
}