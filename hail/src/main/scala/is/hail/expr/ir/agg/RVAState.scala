package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitRegion}
import is.hail.expr.types.physical._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

abstract class RVAState {
  def mb: EmitMethodBuilder
  def r: ClassFieldRef[Region]
  def off: ClassFieldRef[Long]
  def typ: PType
  def region: Code[Region] = r.load()

  def copyFrom(src: Code[Long]): Code[Unit]

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit]

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit]

  def er: EmitRegion = EmitRegion(mb, region)
  def using(definition: Code[Region])(f: => Code[Unit]): Code[Unit] =
    Code(r := definition, f, region.close())
}

case class TypedRVAState(typ: PType, mb: EmitMethodBuilder, r: ClassFieldRef[Region], off: ClassFieldRef[Long]) extends RVAState {
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
  def typ(n: Int): PTuple = PTuple(Array.fill(n)(PInt64(true)), required = true)
}

case class StateContainer(states: Array[RVAState], topRegion: Code[Region]) {
  val nStates: Int = states.length
  val typ: PTuple = StateContainer.typ(nStates)
  def getRegion(rOffset: Code[Int], i: Int): Code[Region] = topRegion.getParentReference(rOffset + i)
  def getStateOffset(off: Code[Long], i: Int): Code[Long] = typ.loadField(topRegion, off, i)
  def loadStateAddress(off: Code[Long], i: Int): Code[Long] = topRegion.loadAddress(getStateOffset(off, i))

  def toCode(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    coerce[Unit](Code(Array.tabulate(nStates)(i => f(i, states(i))): _*))

  def loadStateOffsets(stateOffset: Code[Long]): Code[Unit] =
    toCode((i, s) => s.off := loadStateAddress(stateOffset, i))

  def scoped(rOffset: Code[Int], f: Code[Unit]): Code[Unit] =
    Array.range(0, nStates).foldLeft[Code[Unit]](f)( (cont, i) =>
      states(i).using(getRegion(rOffset, i))(cont))

  def scoped(rOffset: Code[Int])(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    scoped(rOffset, toCode(f))

  def scoped(rOffset: Code[Int], statesOffset: Code[Long], f: Code[Unit]): Code[Unit] =
    scoped(rOffset, Code(loadStateOffsets(statesOffset), f))

  def scoped(rOffset: Code[Int], statesOffset: Code[Long])(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    scoped(rOffset, statesOffset, toCode(f))

  def update(rOffset: Code[Int], statesOffset: Code[Long], f: Code[Unit]): Code[Unit] = {
    val updaters = toCode((i, s) => Code(
        topRegion.setParentReference(s.region, rOffset + i),
        topRegion.storeAddress(getStateOffset(statesOffset, i), s.off)))

    scoped(rOffset, statesOffset, Code(f, updaters))
  }

  def update(rOffset: Code[Int], statesOffset: Code[Long])(f: (Int, RVAState) => Code[Unit]): Code[Unit] = {
    update(rOffset, statesOffset, toCode(f))
  }

  def refreshOne(rOffset: Code[Int], i: Int): Code[Unit] = {
    states(i).using(Code.newInstance[Region]()) {topRegion.setParentReference(states(i).region, rOffset + i)}
  }

  def updateOne(rOffset: Code[Int], statesOffset: Code[Long], i: Int, f: Code[Unit]): Code[Unit] = {
    val s = states(i)
    s.using(getRegion(rOffset, i)) {Code(
      s.off := loadStateAddress(statesOffset, i),
      f,
      topRegion.setParentReference(s.region, rOffset + i),
      topRegion.storeAddress(getStateOffset(statesOffset, i), s.off))
    }
  }

  def addState(srvb: StagedRegionValueBuilder): Code[Unit] = {
    srvb.addBaseStruct(typ, ssb =>
      Code(ssb.start(),
        toCode((_, s) => Code(ssb.addAddress(s.off), ssb.advance()))))
  }
}