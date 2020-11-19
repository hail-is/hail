package is.hail.expr.ir

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.types.physical.{PCode, PType}
import is.hail.utils._

object StagedRegion {
  def apply(
    r: Value[Region],
    allowSubregions: Boolean = false,
    parents: Seq[ParentStagedRegion] = Seq(),
    description: String = "root"
  ): ParentStagedRegion =
    new ParentStagedRegion(r, parents, allowSubregions, description)

  def apply(r: Value[Region]): StagedRegion = new StagedRegion {
    val code = r
    def <=(that: ParentStagedRegion): Boolean = false
    def assertSubRegion(that: ParentStagedRegion) {
      assert(this <= that, s"root\n${that.description}")
    }
  }

  def swap(mb: EmitMethodBuilder[_], x: OwnedStagedRegion, y: OwnedStagedRegion)(implicit line: LineNumber): Code[Unit] = {
    x.parent assertEqual y.parent
    (x, y) match {
      case (x: RealOwnedStagedRegion, y: RealOwnedStagedRegion) =>
        val temp = mb.newLocal[Region]("sr_swap")
        Code(temp := x.r, x.r := y.r, y.r := temp)
      case (x: DummyOwnedStagedRegion, y: DummyOwnedStagedRegion) =>
        Code._empty
    }
  }
}

abstract class StagedRegion {
  def code: Value[Region]

  final def asParent(allowAllocations: Boolean, description: String): ParentStagedRegion = {
    val parents = this match {
      case child: ChildStagedRegion => child.otherAncestors :+ child.parent
      case parent: ParentStagedRegion => Seq(parent)
      case _ => Seq()
    }
    StagedRegion(code, allowAllocations, parents, description)
  }

  def <=(that: ParentStagedRegion): Boolean

  def assertSubRegion(that: ParentStagedRegion): Unit
}

class ParentStagedRegion(
  val code: Value[Region],
  val parents: Seq[ParentStagedRegion],
  val allowSubregions: Boolean,
  desc: String
) extends StagedRegion { self =>
  final def description: String = parents match {
    case Seq() => desc
    case Seq(p) => s"$desc < ${ p.description }"
    case ps => s"$desc < ${ ps.map(_.description).mkString(" {", " | ", " }") }"
  }

  def createChildRegion(mb: EmitMethodBuilder[_]): OwnedStagedRegion =
    if (allowSubregions) {
      val newR = mb.genFieldThisRef[Region]("staged_region_child")
      new RealOwnedStagedRegion(newR, this)
    } else {
      new DummyOwnedStagedRegion(code, this)
    }

  def createChildRegionArray(mb: EmitMethodBuilder[_], length: Int): OwnedStagedRegionArray =
    if (allowSubregions) {
      val regionArray = mb.genFieldThisRef[Array[Region]]("staged_region_child_array")

      def get(i: Value[Int]): Settable[Region] = new Settable[Region] {
        def get(implicit line: LineNumber): Code[Region] = regionArray(i)

        def store(rhs: Code[Region])(implicit line: LineNumber): Code[Unit] = regionArray.update(i, rhs)
      }

      new OwnedStagedRegionArray {
        def apply(i: Value[Int]): OwnedStagedRegion = new RealOwnedStagedRegion(get(i), self)

        def allocateRegions(mb: EmitMethodBuilder[_], size: Int)(implicit line: LineNumber): Code[Unit] = {
          val i = mb.newLocal[Int]("sora_alloc_i")
          Code(
            regionArray := Code.newArray(length),
            Code.forLoop(i := 0, i < length, i := i + 1, apply(i).allocateRegion(size)))
        }

        def freeAll(mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Unit] = {
          val i = mb.newLocal[Int]("sora_free_i")
          Code(
            Code.forLoop(i := 0, i < length, i := i + 1, apply(i).free()),
            regionArray := Code._null)
        }
      }
    } else {
      new OwnedStagedRegionArray {
        def apply(i: Value[Int]): OwnedStagedRegion = new DummyOwnedStagedRegion(code, self)

        def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit] =
          Code._empty

        def freeAll(mb: EmitMethodBuilder[_]): Code[Unit] =
          Code._empty
      }
    }

  override def equals(that: Any): Boolean = that match {
    case that: ParentStagedRegion =>
      (this.allowSubregions == that.allowSubregions) && (this.code eq that.code)
    case _ => false
  }

  final def <=(that: ParentStagedRegion): Boolean =
    (this == that) || parents.exists(_ <= that)

  def assertEqual(that: ParentStagedRegion) {
    assert(this == that, s"${this.description}\n${that.description}")
  }

  def assertSubRegion(that: ParentStagedRegion) {
    assert(this <= that, s"${this.description}\n${that.description}")
  }
}

abstract class ChildStagedRegion extends StagedRegion {
  def parent: ParentStagedRegion

  def otherAncestors: Seq[ParentStagedRegion]

  def asSubregionOf(that: ParentStagedRegion): ChildStagedRegion

  def createSiblingRegion(mb: EmitMethodBuilder[_]): OwnedStagedRegion

  final def createSiblingRegionArray(mb: EmitMethodBuilder[_], length: Int): OwnedStagedRegionArray =
    parent.createChildRegionArray(mb, length)

  final def copyToParent(mb: EmitMethodBuilder[_], value: PCode, destType: PType): PCode =
    copyTo(mb, value, parent, destType)

  final def copyToParent(mb: EmitMethodBuilder[_], value: PCode): PCode =
    copyTo(mb, value, parent)

  def copyTo(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion, destType: PType): PCode

  def copyTo(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion): PCode

  final def <=(that: ParentStagedRegion): Boolean =
    (this.parent <= that) || otherAncestors.exists(_ <= that)

  def description: String = if (otherAncestors.isEmpty)
    parent.description
  else
    otherAncestors.map(_.description).mkString(s"{ ${parent.description} | ", " | ", "}")

  def assertSubRegion(that: ParentStagedRegion) {
    assert(this <= that, s"${this.parent.description}\n${that.description}")
  }
}

trait OwnedStagedRegion extends ChildStagedRegion {
  def allocateRegion(size: Int): Code[Unit]

  def free(): Code[Unit]

  def clear(): Code[Unit]

  def giveToParent(): Code[Unit]

  def giveToSibling(dest: ChildStagedRegion): Code[Unit]

  def shareWithSibling(dest: ChildStagedRegion): Code[Unit]

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode)(implicit line: LineNumber): Code[Unit]
}

abstract class OwnedStagedRegionArray {
  def apply(i: Value[Int]): OwnedStagedRegion

  def allocateRegions(mb: EmitMethodBuilder[_], size: Int): Code[Unit]

  def freeAll(mb: EmitMethodBuilder[_]): Code[Unit]
}

class RealOwnedStagedRegion(
  val r: Settable[Region],
  val parent: ParentStagedRegion,
  val otherAncestors: Seq[ParentStagedRegion] = Seq()
) extends OwnedStagedRegion {
  assert(parent.allowSubregions)

  def code: Value[Region] = r

  def asSubregionOf(that: ParentStagedRegion): ChildStagedRegion =
    new RealOwnedStagedRegion(r, parent, otherAncestors :+ that)

  def createSiblingRegion(mb: EmitMethodBuilder[_]): OwnedStagedRegion = {
    val newR = mb.genFieldThisRef[Region]("staged_region_child")
    new RealOwnedStagedRegion(newR, parent, otherAncestors)
  }

  def allocateRegion(size: Int)(implicit line: LineNumber): Code[Unit] =
    r := Region.stagedCreate(size)

  def free()(implicit line: LineNumber): Code[Unit] =
    Code(r.invalidate(), r := Code._null)

  def clear()(implicit line: LineNumber): Code[Unit] =
    (r: Value[Region]).clear()

  def giveToParent()(implicit line: LineNumber): Code[Unit] =
    r.invoke[Region, Unit]("move", parent.code)

  def copyTo(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion, destType: PType): PCode = {
    dest assertSubRegion parent
    value.copyToRegion(mb, dest.code, destType)
  }

  def copyTo(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion): PCode =
    copyTo(mb, value, dest, value.pt)

  def giveToSibling(dest: ChildStagedRegion)(implicit line: LineNumber): Code[Unit] = {
    dest assertSubRegion  parent
    r.invoke[Region, Unit]("move", dest.code)
  }

  def shareWithSibling(dest: ChildStagedRegion)(implicit line: LineNumber): Code[Unit] = {
    dest assertSubRegion parent
    dest.code.invoke[Region, Unit]("addReferenceTo", r)
  }

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode)(implicit line: LineNumber): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = true)
}

class DummyOwnedStagedRegion(
  val code: Value[Region],
  val parent: ParentStagedRegion,
  val otherAncestors: Seq[ParentStagedRegion] = Seq()
) extends OwnedStagedRegion {
  assert(!parent.allowSubregions)

  def asSubregionOf(that: ParentStagedRegion): ChildStagedRegion =
    new DummyOwnedStagedRegion(code, parent, otherAncestors :+ that)

  def createSiblingRegion(mb: EmitMethodBuilder[_]): OwnedStagedRegion = {
    new DummyOwnedStagedRegion(code, parent, otherAncestors)
  }

  def allocateRegion(size: Int): Code[Unit] = Code._empty

  def free(): Code[Unit] = Code._empty

  def clear(): Code[Unit] = Code._empty

  def giveToParent(): Code[Unit] = Code._empty

  def copyTo(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion, destType: PType): PCode = {
    dest assertSubRegion parent
    value.castTo(mb, parent.code, destType)
  }

  def copyTo(mb: EmitMethodBuilder[_], value: PCode, dest: StagedRegion): PCode = {
    dest assertSubRegion parent
    value
  }

  def giveToSibling(dest: ChildStagedRegion): Code[Unit] = {
    dest assertSubRegion parent
    Code._empty
  }

  def shareWithSibling(dest: ChildStagedRegion): Code[Unit] = {
    dest assertSubRegion parent
    Code._empty
  }

  def addToParentRVB(srvb: StagedRegionValueBuilder, value: PCode)(implicit line: LineNumber): Code[Unit] =
    srvb.addIRIntermediate(value, deepCopy = false)
}
