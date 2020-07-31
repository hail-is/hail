package is.hail.expr.ir

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, Settable, Value}
import is.hail.types.physical.{PCode, PType}
import is.hail.utils._

abstract class StagedRegion {
  def code: Value[Region]

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion

  def createDummyChildRegion: StagedOwnedRegion =
    new DummyStagedOwnedRegion(code, this)
}

trait StagedOwnedRegion extends StagedRegion {
  val parent: StagedRegion

  def allocateRegion(size: Int): Code[Unit]

  def free(): Code[Unit]

  def clear(): Code[Unit]

  def giveToParent(): Code[Unit]

  def shareWithParent(): Code[Unit]

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit]

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion, destType: PType): PCode

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion): PCode =
    copyToSibling(mb, value, dest, value.pt)
}

object StagedRegion {
  def apply(r: Value[Region], allowSubregions: Boolean = false): StagedRegion =
    if (allowSubregions) new RealStagedRegion(r) else new DummyStagedRegion(r)

  def swap(mb: EmitMethodBuilder[_], x: StagedOwnedRegion, y: StagedOwnedRegion): Code[Unit] = {
    assert(x.parent == y.parent)
    (x, y) match {
      case (x: RealStagedOwnedRegion, y: RealStagedOwnedRegion) =>
        val temp = mb.newLocal[Region]("sr_swap")
        Code(temp := x.r, x.r := y.r, y.r := temp)
      case (x: DummyStagedOwnedRegion, y: DummyStagedOwnedRegion) =>
        Code._empty
    }
  }
}

class RealStagedRegion(r: Value[Region]) extends StagedRegion {
  def code: Value[Region] = r

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion = {
    val newR = mb.newLocal[Region]("staged_region_child")
    new RealStagedOwnedRegion(newR, this)
  }
}

class RealStagedOwnedRegion(val r: Settable[Region], val parent: StagedRegion) extends RealStagedRegion(r) with StagedOwnedRegion {
  def allocateRegion(size: Int): Code[Unit] = r := Region.stagedCreate(size)

  def free(): Code[Unit] = Code(r.invalidate(), r := Code._null)

  def clear(): Code[Unit] = (r: Value[Region]).clear()

  def giveToParent(): Code[Unit] = r.invoke[Region, Unit]("move", parent.code)

  def shareWithParent(): Code[Unit] = parent.code.invoke[Region, Unit]("addReferenceTo", r)

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode =
    value.copyToRegion(mb, parent.code)

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = true)

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion, destType: PType): PCode = {
    assert(dest.asInstanceOf[RealStagedOwnedRegion].parent == parent)
    value.copyToRegion(mb, dest.code, destType)
  }
}

class DummyStagedRegion(r: Value[Region]) extends StagedRegion {
  def code: Value[Region] = r

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion =
    new DummyStagedOwnedRegion(r, this)
}

class DummyStagedOwnedRegion(val r: Value[Region], val parent: StagedRegion) extends DummyStagedRegion(r) with StagedOwnedRegion {
  def allocateRegion(size: Int): Code[Unit] = Code._empty

  def free(): Code[Unit] = Code._empty

  def clear(): Code[Unit] = Code._empty

  def giveToParent(): Code[Unit] = Code._empty

  def shareWithParent(): Code[Unit] = Code._empty

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode = value

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = false)

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion, destType: PType): PCode = {
    assert(dest.asInstanceOf[DummyStagedOwnedRegion].parent == parent)
    if (destType == value.pt)
      value
    else
      value.castTo(mb, dest.code, destType)
  }
}
