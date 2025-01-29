package is.hail.utils.richUtils

import is.hail.annotations.{Region, RegionMemory, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder

class RichCodeRegion(val region: Code[Region]) extends AnyVal {
  def allocate(alignment: Code[Long], n: Code[Long]): Code[Long] =
    region.invoke[Long, Long, Long]("allocate", alignment, n)

  def clearRegion(): Code[Unit] =
    region.invoke[Unit]("clear")

  def getMemory(): Code[RegionMemory] = region.invoke[RegionMemory]("getMemory")

  def trackAndIncrementReferenceCountOf(other: Code[Region]): Code[Unit] =
    region.invoke[Region, Unit]("addReferenceTo", other)

  def takeOwnershipOfAndClear(other: Code[Region]): Code[Unit] =
    other.invoke[Region, Unit]("move", region)

  def setNumParents(n: Code[Int]): Code[Unit] =
    region.invoke[Int, Unit]("setNumParents", n)

  def setParentReference(r: Code[Region], i: Code[Int]): Code[Unit] =
    region.invoke[Region, Int, Unit]("setParentReference", r, i)

  def getParentReference(r: Code[Region], i: Code[Int], size: Int): Code[Region] =
    region.invoke[Int, Int, Region]("getParentReference", i, const(size))

  def setFromParentReference(r: Code[Region], i: Code[Int], size: Int): Code[Unit] =
    region.invoke[Region, Int, Int, Unit]("setFromParentReference", r, i, const(size))

  def unreferenceRegionAtIndex(i: Code[Int]): Code[Unit] =
    region.invoke[Int, Unit]("unreferenceRegionAtIndex", i)

  def isValid: Code[Boolean] = region.invoke[Boolean]("isValid")

  def invalidate(): Code[Unit] = region.invoke[Unit]("invalidate")

  def getNewRegion(blockSize: Code[Int]): Code[Unit] =
    region.invoke[Int, Unit]("getNewRegion", blockSize)

  def storeJavaObject(obj: Code[AnyRef]): Code[Int] =
    region.invoke[AnyRef, Int]("storeJavaObject", obj)

  def lookupJavaObject(idx: Code[Int]): Code[AnyRef] =
    region.invoke[Int, AnyRef]("lookupJavaObject", idx)

  def getPool(): Code[RegionPool] = region.invoke[RegionPool]("getPool")

  def totalManagedBytes(): Code[Long] = region.invoke[Long]("totalManagedBytes")

  def allocateSharedChunk(nBytes: Code[Long]): Code[Long] =
    region.invoke[Long, Long]("allocateSharedChunk", nBytes)

  def trackSharedChunk(cb: EmitCodeBuilder, addr: Code[Long]): Unit =
    cb += region.invoke[Long, Unit]("trackSharedChunk", addr)
}
