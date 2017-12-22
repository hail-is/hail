package is.hail.expr.typ

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.utils._

import scala.reflect.{ClassTag, _}

/**
  * Created by dking on 12/21/17.
  */
class TBinary(override val required: Boolean) extends Type {
  def _toString = "Binary"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
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

  def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    val ord = Ordering.Iterable[Byte]

    annotationOrdering(extendOrderingToNull(missingGreatest)(
      new Ordering[Array[Byte]] {
        def compare(a: Array[Byte], b: Array[Byte]): Int = ord.compare(a, b)
      }))
  }

  override def byteSize: Long = 8
}

object TBinary {
  def apply(required: Boolean = false): TBinary = if (required) TBinaryRequired else TBinaryOptional

  def unapply(t: TBinary): Option[Boolean] = Option(t.required)

  def contentAlignment: Long = 4

  def contentByteSize(length: Int): Long = 4 + length

  def loadLength(region: Region, boff: Long): Int =
    region.loadInt(boff)

  def bytesOffset(boff: Long): Long = boff + 4

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentAlignment, contentByteSize(length))
  }

}