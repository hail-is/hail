package is.hail.expr.ir

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.types.physical.{PCode, PType}
import is.hail.utils._

abstract class StagedRegion {
  def code: Value[Region]

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion

  def createDummyChildRegion: StagedOwnedRegion =
    new DummyStagedOwnedRegion(code, this)

  def createChildRegionArray(mb: EmitMethodBuilder[_], length: Int): StagedOwnedRegionArray
}

trait StagedOwnedRegion extends StagedRegion {
  val parent: StagedRegion

  def allocateRegion(size: Int): Code[Unit]

  def free(): Code[Unit]

  def clear(): Code[Unit]

  def giveToParent(): Code[Unit]

  def shareWithParent(): Code[Unit]

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode, destType: PType): PCode =
    StagedRegion.copy(mb, value, this, parent, destType)

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode =
    StagedRegion.copy(mb, value, this, parent)

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit]

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: StagedOwnedRegion, destType: PType): PCode =
    StagedRegion.copy(mb, value, this, dest, destType)

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: StagedOwnedRegion): PCode =
    StagedRegion.copy(mb, value, this, dest)
}

abstract class StagedOwnedRegionArray {
  def apply(i: Value[Int]): StagedOwnedRegion

  def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit]

  def freeAll(mb: EmitMethodBuilder[_]): Code[Unit]
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

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: StagedOwnedRegion, dest: StagedOwnedRegion): PCode =
    copy(mb, value, source, dest, value.pt)

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: StagedOwnedRegion, dest: StagedOwnedRegion, destType: PType): PCode = {
    assert(source.parent eq dest.parent)
    dest match {
      case _: RealStagedOwnedRegion =>
        value.copyToRegion(mb, dest.code, destType)
      case _: DummyStagedOwnedRegion =>
        value
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: StagedRegion, dest: StagedOwnedRegion): PCode =
    copy(mb, value, source, dest, value.pt)

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: StagedRegion, dest: StagedOwnedRegion, destType: PType): PCode = {
    assert(dest.parent eq source)
    dest match {
      case _: RealStagedOwnedRegion =>
        value.copyToRegion(mb, dest.code, destType)
      case _: DummyStagedOwnedRegion =>
        value
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: StagedOwnedRegion, dest: StagedRegion): PCode = {
    assert(source.parent eq dest)
    source match {
      case _: RealStagedOwnedRegion =>
        value.copyToRegion(mb, dest.code)
      case _: DummyStagedOwnedRegion =>
        value
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: StagedOwnedRegion, dest: StagedRegion, destType: PType): PCode = {
    assert(source.parent eq dest)
    source match {
      case _: RealStagedOwnedRegion =>
        value.copyToRegion(mb, dest.code, destType)
      case _: DummyStagedOwnedRegion =>
        value.castTo(mb, dest.code, destType)
    }
  }
}

class RealStagedRegion(r: Value[Region]) extends StagedRegion { self =>
  def code: Value[Region] = r

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion = {
    val newR = mb.genFieldThisRef[Region]("staged_region_child")
    new RealStagedOwnedRegion(newR, this)
  }

  def createChildRegionArray(mb: EmitMethodBuilder[_], length: Int): StagedOwnedRegionArray = {
    val regionArray = mb.genFieldThisRef[Array[Region]]("staged_region_child_array")

    def get(i: Value[Int]): Settable[Region] = new Settable[Region] {
      def get: Code[Region] = regionArray(i)

      def store(rhs: Code[Region]): Code[Unit] = regionArray.update(i, rhs)
    }

    new StagedOwnedRegionArray {
      def apply(i: Value[Int]): StagedOwnedRegion = new RealStagedOwnedRegion(get(i), self)

      def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit] = {
        val i = mb.newLocal[Int]("sora_alloc_i")
        Code(
          regionArray := Code.newArray(length),
          Code.forLoop(i := 0, i < length, i := i + 1, apply(i).allocateRegion(size)))
      }

      def freeAll(mb: EmitMethodBuilder[_]): Code[Unit] = {
        val i = mb.newLocal[Int]("sora_free_i")
        Code(
          Code.forLoop(i := 0, i < length, i := i + 1, apply(i).free()),
          regionArray := Code._null)
      }
    }
  }
}

class RealStagedOwnedRegion(val r: Settable[Region], val parent: StagedRegion) extends RealStagedRegion(r) with StagedOwnedRegion {
  def allocateRegion(size: Int): Code[Unit] = r := Region.stagedCreate(size)

  def free(): Code[Unit] = Code(r.invalidate(), r := Code._null)

  def clear(): Code[Unit] = (r: Value[Region]).clear()

  def giveToParent(): Code[Unit] = r.invoke[Region, Unit]("move", parent.code)

  def shareWithParent(): Code[Unit] = parent.code.invoke[Region, Unit]("addReferenceTo", r)

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = true)
}

class DummyStagedRegion(r: Value[Region]) extends StagedRegion { self =>
  def code: Value[Region] = r

  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion =
    new DummyStagedOwnedRegion(r, this)

  def createChildRegionArray(mb: EmitMethodBuilder[_], length: Int): StagedOwnedRegionArray =
    new StagedOwnedRegionArray {
      def apply(i: Value[Int]): StagedOwnedRegion = new DummyStagedOwnedRegion(r, self)

      def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit] =
        Code._empty

      def freeAll(mb: EmitMethodBuilder[_]): Code[Unit] =
        Code._empty
    }
}

class DummyStagedOwnedRegion(val r: Value[Region], val parent: StagedRegion) extends DummyStagedRegion(r) with StagedOwnedRegion {
  def allocateRegion(size: Int): Code[Unit] = Code._empty

  def free(): Code[Unit] = Code._empty

  def clear(): Code[Unit] = Code._empty

  def giveToParent(): Code[Unit] = Code._empty

  def shareWithParent(): Code[Unit] = Code._empty

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = false)
}
