package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitRegion}
import is.hail.expr.types.physical._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

abstract class RVAState {
  def mb: EmitMethodBuilder
  protected def r: ClassFieldRef[Region]
  def off: ClassFieldRef[Long]
  def typ: PType
  def region: Code[Region] = r.load()

  def assign(other: Code[Region]): Code[Unit] =
    Code(region.isNull.mux(Code._empty, region.close()), r := other)

  def loadStateFrom(src: Code[Long]): Code[Unit]
//  def loadStateFrom(region: Code[Region], src: Code[Long]): Code[Unit]
//
//  def storeStateInto(topRegion: Code[Region], rOffset: Code[Int]): Code[Long]

  def copyFrom(src: Code[Long]): Code[Unit]

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit]

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit]

  def close: Code[Unit] = region.isNull.mux(Code._empty, Code(region.close(), r := Code._null))

  def er: EmitRegion = EmitRegion(mb, region)
  def using(definition: Code[Region])(f: Code[Unit]): Code[Unit] =
    Code(assign(definition), f, close)
}

case class TypedRVAState(typ: PType, mb: EmitMethodBuilder, r: ClassFieldRef[Region], off: ClassFieldRef[Long]) extends RVAState {
  def loadStateFrom(src: Code[Long]): Code[Unit] = off := src

  def copyFrom(src: Code[Long]): Code[Unit] = off := StagedRegionValueBuilder.deepCopy(er, typ, src)

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
    val enc = codec.buildEmitEncoderF[Long](typ, typ, mb.fb)
    ob: Code[OutputBuffer] => enc(region, off, ob)
  }

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    val dec = codec.buildEmitDecoderF[Long](typ, typ, mb.fb)
    ib: Code[InputBuffer] => off := dec(region, ib)
  }
}

object StateContainer {
  def typ(n: Int): PTuple = PTuple(Array.fill(n)(PInt64()), required = true)
}

case class StateContainer(states: Array[RVAState], topRegion: Code[Region]) {
  val nStates: Int = states.length
  val typ: PTuple = StateContainer.typ(nStates)

  def apply(i: Int): RVAState = states(i)
  def getRegion(rOffset: Code[Int], i: Int): Code[Region] = topRegion.getParentReference(rOffset + i)
  def getStateOffset(off: Code[Long], i: Int): Code[Long] = typ.loadField(topRegion, off, i)
  def loadStateAddress(off: Code[Long], i: Int): Code[Long] = topRegion.loadAddress(getStateOffset(off, i))

  def setAllMissing(off: Code[Long]): Code[Unit] = toCode((i, _) => typ.setFieldMissing(topRegion, off, i))
  def setPresent(off: Code[Long], i: Int): Code[Unit] = typ.setFieldPresent(topRegion, off, i)

  def toCode(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    coerce[Unit](Code(Array.tabulate(nStates)(i => f(i, states(i))): _*))

  def loadRegions(rOffset: Code[Int]): Code[Unit] =
    toCode((i, s) => s.assign(topRegion.getParentReference(rOffset + i)))

  def loadStateOffsets(stateOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => typ.isFieldMissing(topRegion, stateOffset, i).mux(
      Code._empty,
      s.loadStateFrom(loadStateAddress(stateOffset, i))))

  def storeRegions(rOffset: Code[Int]): Code[Unit] =
    toCode((i, s) =>
      s.region.isNull.mux(Code._empty,
        Code(
          topRegion.setParentReference(s.region, rOffset + i),
          s.close)))

  def storeStateOffsets(statesOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => topRegion.storeAddress(getStateOffset(statesOffset, i), s.off))

  def scoped(rOffset: Code[Int], f: Code[Unit]): Code[Unit] =
    Array.range(0, nStates).foldLeft[Code[Unit]](f)( (cont, i) =>
        states(i).using(getRegion(rOffset, i))(cont)
      )

  def scoped(rOffset: Code[Int])(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    scoped(rOffset, toCode(f))

  def scoped(rOffset: Code[Int], statesOffset: Code[Long], f: Code[Unit]): Code[Unit] =
    scoped(rOffset, Code(loadStateOffsets(statesOffset), f))

  def scoped(rOffset: Code[Int], statesOffset: Code[Long])(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    scoped(rOffset, statesOffset, toCode(f))

  def update(rOffset: Code[Int], statesOffset: Code[Long], f: Code[Unit]): Code[Unit] =
    scoped(rOffset, statesOffset, Code(f, storeRegions(rOffset), storeStateOffsets(statesOffset)))

  def update(rOffset: Code[Int], statesOffset: Code[Long])(f: (Int, RVAState) => Code[Unit]): Code[Unit] = {
    update(rOffset, statesOffset, toCode(f))
  }

  def addState(srvb: StagedRegionValueBuilder): Code[Unit] = {
    srvb.addBaseStruct(typ, ssb =>
      Code(ssb.start(),
        toCode((_, s) => Code(ssb.addAddress(s.off), ssb.advance()))))
  }

  def closeNested: Code[Unit] = toCode((_, s) => s.close)
}