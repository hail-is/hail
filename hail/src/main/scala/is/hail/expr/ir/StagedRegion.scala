package is.hail.expr.ir

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, Settable, Value}
import is.hail.types.physical.PCode
import is.hail.utils._

abstract class StagedRegion {
  def code: Value[Region]

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion
}

trait StagedOwnedRegion extends StagedRegion {
  def allocateRegion(size: Int): Code[Unit]

  def free(): Code[Unit]

  def clear(): Code[Unit]

  def giveToParent(): Code[Unit]

  def shareWithParent(): Code[Unit]

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit]
}

object StagedRegion {
  def apply(r: Value[Region], allowSubregions: Boolean = false): StagedRegion =
    if (allowSubregions) new RealStagedRegion(r) else new DummyStagedRegion(r)
}

class RealStagedRegion(r: Value[Region]) extends StagedRegion {
  def code: Value[Region] = r

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion = {
    val newR = mb.newLocal[Region]("staged_region_child")
    new RealStagedOwnedRegion(newR, r)
  }
}

class RealStagedOwnedRegion(r: Settable[Region], parent: Value[Region]) extends RealStagedRegion(r) with StagedOwnedRegion {
  def allocateRegion(size: Int): Code[Unit] = r := Region.stagedCreate(size)

  def free(): Code[Unit] = Code(r.invalidate(), r := Code._null)

  def clear(): Code[Unit] = (r: Value[Region]).clear()

  def giveToParent(): Code[Unit] = parent.invoke("takeOwnershipOf", r)

  def shareWithParent(): Code[Unit] = parent.invoke("addReferenceTo", r)

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode =
    value.copyToRegion(mb, parent)

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = true)
}

class DummyStagedRegion(r: Value[Region]) extends StagedOwnedRegion {
  def code: Value[Region] = r

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion =
    this

  def allocateRegion(size: Int): Code[Unit] = Code._empty

  def free(): Code[Unit] = Code._empty

  def clear(): Code[Unit] = Code._empty

  def giveToParent(): Code[Unit] = Code._empty

  def shareWithParent(): Code[Unit] = Code._empty

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode = value

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = false)
}
