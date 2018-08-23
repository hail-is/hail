package is.hail.expr.types

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

abstract class PhysicalType {
  def virtualType: Type

  def byteSize: Long

  def alignment: Long

  def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering
}

case class PDefault(virtualType: Type) extends PhysicalType {
  def byteSize: Long = virtualType.byteSize

  def alignment: Long = virtualType.alignment

  def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering =
    virtualType.unsafeOrdering(rightType.virtualType, missingGreatest)

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = virtualType.codeOrdering(mb, other.virtualType)
}

abstract class PStruct extends PhysicalType {

  override def virtualType: TBaseStruct

  def allocate(region: Region): Long

  def isFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean]

  def isFieldDefined(rv: RegionValue, fieldIdx: Int): Boolean =
    isFieldDefined(rv.region, rv.offset, fieldIdx)

  def isFieldDefined(region: Region, offset: Long, fieldIdx: Int): Boolean

  def isFieldDefined(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(region, offset, fieldIdx)

  def setFieldMissing(region: Region, offset: Long, fieldIdx: Int): Unit

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit]

  def fieldOffset(offset: Long, fieldIdx: Int): Long

  def fieldOffset(offset: Code[Long], fieldIdx: Int): Code[Long]

  def loadField(rv: RegionValue, fieldIdx: Int): Long = loadField(rv.region, rv.offset, fieldIdx)

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long

  def loadField(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Long]

  def fieldType(field: Int): PhysicalType
}

abstract class PConstructableStruct extends PStruct {
  def clearMissingBits(region: Region, off: Long): Unit

  def clearMissingBits(region: Code[Region], off: Code[Long]): Code[Unit]

}

abstract class PArray extends PhysicalType {
  override def virtualType: TIterable

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int]

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] = !isElementDefined(region, aoff, i)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean]

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long]

  def elementOffset(aoff: Long, length: Int, i: Int): Long

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long]

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long]

  def elementType: PhysicalType
}

abstract class PConstructableArray extends PArray {
  def clearMissingBits(region: Region, aoff: Long, length: Int)

  def initialize(region: Region, aoff: Long, length: Int): Unit

  def initialize(region: Code[Region], aoff: Code[Long], length: Code[Int], a: Settable[Int]): Unit

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit]

  def setElementMissing(region: Region, aoff: Long, i: Int): Unit
}

// TODO: the primitive system can be relaxed to include non-canonical primitives
abstract class PPrimitive(val byteSize: Long) extends PhysicalType {
  def alignment: Long = byteSize
}

case object PInt32 extends PPrimitive(4) {
  def virtualType: Type = TInt32()

  override def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      Integer.compare(r1.loadInt(o1), r2.loadInt(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = {
    assert(other == this)
    new CodeOrdering {
      type T = Int

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", x, y)

      override def ltNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x < y

      override def lteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x <= y

      override def gtNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x > y

      override def gteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x >= y

      override def equivNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x.ceq(y)
    }
  }
}

case object PInt64 extends PPrimitive(8) {
  def virtualType: Type = TInt64()

  override def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Long.compare(r1.loadLong(o1), r2.loadLong(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = {
    assert(other == this)
    new CodeOrdering {
      type T = Long

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Long, Long, Long, Int]("compare", x, y)

      override def ltNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x < y

      override def lteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x <= y

      override def gtNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x > y

      override def gteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x >= y

      override def equivNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x.ceq(y)
    }
  }
}

case object PFloat32 extends PPrimitive(4) {
  def virtualType: Type = TFloat32()

  override def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Float.compare(r1.loadFloat(o1), r2.loadFloat(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = {
    assert(other == this)
    new CodeOrdering {
      type T = Float

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Float, Float, Float, Int]("compare", x, y)

      override def ltNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x < y

      override def lteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x <= y

      override def gtNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x > y

      override def gteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x >= y

      override def equivNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x.ceq(y)
    }
  }
}

case object PFloat64 extends PPrimitive(8) {
  def virtualType: Type = TFloat64()

  override def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Double.compare(r1.loadDouble(o1), r2.loadDouble(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = {
    assert(other == this)
    new CodeOrdering {
      type T = Double

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Double, Double, Double, Int]("compare", x, y)

      override def ltNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x < y

      override def lteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x <= y

      override def gtNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x > y

      override def gteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x >= y

      override def equivNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x.ceq(y)
    }
  }

}

case object PBool extends PPrimitive(1) {
  def virtualType: Type = TBoolean()

  override def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Boolean.compare(r1.loadBoolean(o1), r2.loadBoolean(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = {
    assert(other == this)
    new CodeOrdering {
      type T = Boolean

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Boolean, Boolean, Boolean, Int]("compare", x, y)
    }
  }
}

case object PBinary extends PhysicalType {
  def virtualType: Type = TBinary()

  val byteSize: Long = 8
  val alignment: Long = 8

  override def unsafeOrdering(rightType: PhysicalType, missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      val l1 = TBinary.loadLength(r1, o1)
      val l2 = TBinary.loadLength(r2, o2)

      val bOff1 = TBinary.bytesOffset(o1)
      val bOff2 = TBinary.bytesOffset(o2)

      val lim = math.min(l1, l2)
      var i = 0

      while (i < lim) {
        val b1 = r1.loadByte(bOff1 + i)
        val b2 = r2.loadByte(bOff2 + i)
        if (b1 != b2)
          return java.lang.Byte.compare(b1, b2)

        i += 1
      }
      Integer.compare(l1, l2)
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PhysicalType): CodeOrdering = {
    assert(other == this)
    new CodeOrdering {
      type T = Long

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] = {
        val l1 = mb.newLocal[Int]
        val l2 = mb.newLocal[Int]
        val lim = mb.newLocal[Int]
        val i = mb.newLocal[Int]
        val cmp = mb.newLocal[Int]

        Code(
          l1 := TBinary.loadLength(rx, x),
          l2 := TBinary.loadLength(ry, y),
          lim := (l1 < l2).mux(l1, l2),
          i := 0,
          cmp := 0,
          Code.whileLoop(cmp.ceq(0) && i < lim,
            cmp := Code.invokeStatic[java.lang.Byte, Byte, Byte, Int]("compare",
              rx.loadByte(TBinary.bytesOffset(x) + i.toL),
              ry.loadByte(TBinary.bytesOffset(y) + i.toL)),
            i += 1),
          cmp.ceq(0).mux(Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", l1, l2), cmp))
      }
    }
  }
}