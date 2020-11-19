package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s._

class RichCodeRegion(val region: Code[Region]) extends AnyVal {
  def allocate(alignment: Code[Long], n: Code[Long])(implicit line: LineNumber): Code[Long] =
    region.invoke[Long, Long, Long]("allocate", alignment, n)

  def clear()(implicit line: LineNumber): Code[Unit] =
    region.invoke[Unit]("clear")

  def reference(other: Code[Region])(implicit line: LineNumber): Code[Unit] =
    region.invoke[Region, Unit]("reference", other)

  def setNumParents(n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    region.invoke[Int, Unit]("setNumParents", n)

  def setParentReference(r: Code[Region], i: Code[Int])(implicit line: LineNumber): Code[Unit] =
    region.invoke[Region, Int, Unit]("setParentReference", r, i)

  def getParentReference(r: Code[Region], i: Code[Int], size: Int)(implicit line: LineNumber): Code[Region] =
    region.invoke[Int, Int, Region]("getParentReference", i, const(size))

  def setFromParentReference(r: Code[Region], i: Code[Int], size: Int)(implicit line: LineNumber): Code[Unit] =
    region.invoke[Region, Int, Int, Unit]("setFromParentReference", r, i, const(size))

  def unreferenceRegionAtIndex(i: Code[Int])(implicit line: LineNumber): Code[Unit] =
    region.invoke[Int, Unit]("unreferenceRegionAtIndex", i)

  def isValid(implicit line: LineNumber): Code[Boolean] =
    region.invoke[Boolean]("isValid")

  def invalidate()(implicit line: LineNumber): Code[Unit] =
    region.invoke[Unit]("invalidate")

  def getNewRegion(blockSize: Code[Int])(implicit line: LineNumber): Code[Unit] =
    region.invoke[Int, Unit]("getNewRegion", blockSize)

  def storeJavaObject(obj: Code[AnyRef])(implicit line: LineNumber): Code[Int] =
    region.invoke[AnyRef, Int]("storeJavaObject", obj)

  def lookupJavaObject(idx: Code[Int])(implicit line: LineNumber): Code[AnyRef] =
    region.invoke[Int, AnyRef]("lookupJavaObject", idx)
}
