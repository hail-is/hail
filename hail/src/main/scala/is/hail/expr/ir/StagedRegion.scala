package is.hail.expr.ir

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.types.physical.{PCode, PType}
import is.hail.utils._

abstract class StagedRegion {
  def code: Value[Region]

  final def asRoot(allowAllocations: Boolean): RootStagedRegion =
    if (allowAllocations)
      new RealRootStagedRegion(code)
    else
      new DummyRootStagedRegion(code)
}

abstract class RootStagedRegion extends StagedRegion {
  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion

  def createChildRegion(
    mb: EmitMethodBuilder[_],
    allowAllocations: Boolean
  ): StagedOwnedRegion =
    if (allowAllocations) createRealChildRegion(mb) else createDummyChildRegion()

  final def createDummyChildRegion(): StagedOwnedRegion =
    new DummyStagedOwnedRegion(code, this)

  final def createRealChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion = {
    val newR = mb.genFieldThisRef[Region]("staged_region_child")
    new RealStagedOwnedRegion(newR, this)
  }

  def createChildRegionArray(mb: EmitMethodBuilder[_], length: Int): StagedOwnedRegionArray
}

abstract class ChildStagedRegion extends StagedRegion {
  def parent: RootStagedRegion

  final def createSiblingRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion =
    parent.createChildRegion(mb)

  final def createSiblingRegionArray(mb: EmitMethodBuilder[_], length: Int): StagedOwnedRegionArray =
    parent.createChildRegionArray(mb, length)

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: ChildStagedRegion, destType: PType): PCode =
    StagedRegion.copy(mb, value, this, dest, destType)
}

trait StagedOwnedRegion extends ChildStagedRegion {
  def allocateRegion(size: Int): Code[Unit]

  def free(): Code[Unit]

  def clear(): Code[Unit]

  def giveToParent(): Code[Unit]

  def giveToSibling(dest: ChildStagedRegion): Code[Unit]

  def shareWithSibling(dest: ChildStagedRegion): Code[Unit]

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode, destType: PType): PCode =
    StagedRegion.copy(mb, value, this, parent, destType)

  def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode =
    StagedRegion.copy(mb, value, this, parent)

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit]

  def copyToSibling(mb: EmitMethodBuilder[_], value: PCode, dest: ChildStagedRegion): PCode =
    StagedRegion.copy(mb, value, this, dest)
}

abstract class StagedOwnedRegionArray {
  def apply(i: Value[Int]): StagedOwnedRegion

  def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit]

  def freeAll(mb: EmitMethodBuilder[_]): Code[Unit]
}

object StagedRegion {
  def apply(r: Value[Region], allowSubregions: Boolean = false): RootStagedRegion =
    if (allowSubregions) new RealRootStagedRegion(r) else new DummyRootStagedRegion(r)

  def swap(mb: EmitMethodBuilder[_], x: StagedOwnedRegion, y: StagedOwnedRegion): Code[Unit] = {
    assert(x.parent eq y.parent)
    (x, y) match {
      case (x: RealStagedOwnedRegion, y: RealStagedOwnedRegion) =>
        val temp = mb.newLocal[Region]("sr_swap")
        Code(temp := x.r, x.r := y.r, y.r := temp)
      case (x: DummyStagedOwnedRegion, y: DummyStagedOwnedRegion) =>
        Code._empty
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: ChildStagedRegion, dest: ChildStagedRegion): PCode = {
    assert(source.parent eq dest.parent)
    source.parent match {
      case _: RealRootStagedRegion =>
        value.copyToRegion(mb, dest.code)
      case _: DummyRootStagedRegion =>
        value
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: ChildStagedRegion, dest: ChildStagedRegion, destType: PType): PCode = {
    assert(source.parent eq dest.parent)
    source.parent match {
      case _: RealRootStagedRegion =>
        value.copyToRegion(mb, dest.code, destType)
      case _: DummyRootStagedRegion =>
        value.castTo(mb, dest.code, destType)
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: RootStagedRegion, dest: ChildStagedRegion): PCode = {
    assert(dest.parent eq source)
    source match {
      case _: RealRootStagedRegion =>
        value.copyToRegion(mb, dest.code)
      case _: DummyRootStagedRegion =>
        value
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: RootStagedRegion, dest: ChildStagedRegion, destType: PType): PCode = {
    assert(dest.parent eq source)
    source match {
      case _: RealRootStagedRegion =>
        value.copyToRegion(mb, dest.code, destType)
      case _: DummyRootStagedRegion =>
        value.castTo(mb, dest.code, destType)
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: ChildStagedRegion, dest: RootStagedRegion): PCode = {
    assert(source.parent eq dest)
    dest match {
      case _: RealRootStagedRegion =>
        value.copyToRegion(mb, dest.code)
      case _: DummyRootStagedRegion =>
        value
    }
  }

  def copy(mb: EmitMethodBuilder[_], value: PCode, source: ChildStagedRegion, dest: RootStagedRegion, destType: PType): PCode = {
    assert(source.parent eq dest)
    dest match {
      case _: RealRootStagedRegion =>
        value.copyToRegion(mb, dest.code, destType)
      case _: DummyRootStagedRegion =>
        value.castTo(mb, dest.code, destType)
    }
  }
}

class RealRootStagedRegion(val code: Value[Region]) extends RootStagedRegion { self =>
  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion =
    createRealChildRegion(mb)

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

class RealStagedOwnedRegion(val r: Settable[Region], val parent: RootStagedRegion) extends StagedOwnedRegion {
  def code: Value[Region] = r

  def allocateRegion(size: Int): Code[Unit] = r := Region.stagedCreate(size)

  def free(): Code[Unit] = Code(r.invalidate(), r := Code._null)

  def clear(): Code[Unit] = (r: Value[Region]).clear()

  def giveToParent(): Code[Unit] = r.invoke[Region, Unit]("move", parent.code)

  def giveToSibling(dest: ChildStagedRegion): Code[Unit] = {
    assert(dest.parent eq parent)
    r.invoke[Region, Unit]("move", dest.code)
  }

  def shareWithSibling(dest: ChildStagedRegion): Code[Unit] = {
    assert(dest.parent eq parent)
    dest.code.invoke[Region, Unit]("addReferenceTo", r)
  }

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = true)
}

class DummyRootStagedRegion(val code: Value[Region]) extends RootStagedRegion { self =>
  def createChildRegion(mb: EmitMethodBuilder[_]): StagedOwnedRegion =
    createDummyChildRegion()

  def createChildRegionArray(mb: EmitMethodBuilder[_], length: Int): StagedOwnedRegionArray =
    new StagedOwnedRegionArray {
      def apply(i: Value[Int]): StagedOwnedRegion = new DummyStagedOwnedRegion(code, self)

      def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit] =
        Code._empty

      def freeAll(mb: EmitMethodBuilder[_]): Code[Unit] =
        Code._empty
    }
}

class DummyStagedOwnedRegion(val code: Value[Region], val parent: RootStagedRegion) extends StagedOwnedRegion {
  def allocateRegion(size: Int): Code[Unit] = Code._empty

  def free(): Code[Unit] = Code._empty

  def clear(): Code[Unit] = Code._empty

  def giveToParent(): Code[Unit] = Code._empty

  def giveToSibling(dest: ChildStagedRegion): Code[Unit] = {
    assert(dest.parent eq parent)
    Code._empty
  }

  def shareWithSibling(dest: ChildStagedRegion): Code[Unit] = {
    assert(dest.parent eq parent)
    Code._empty
  }

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = false)
}
