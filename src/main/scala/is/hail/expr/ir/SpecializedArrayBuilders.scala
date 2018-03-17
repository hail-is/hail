package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.types.Type

class StagedArrayBuilder(elt: Type, mb: MethodBuilder) {

  val ti = typeToTypeInfo(elt)

  val ref: LocalRef[Any] = coerce[Any](ti match {
    case BooleanInfo => mb.newLocal[BooleanArrayBuilder]("zab")
    case IntInfo => mb.newLocal[IntArrayBuilder]("iab")
    case LongInfo => mb.newLocal[LongArrayBuilder]("jab")
    case FloatInfo => mb.newLocal[FloatArrayBuilder]("fab")
    case DoubleInfo => mb.newLocal[DoubleArrayBuilder]("dab")
    case ti => throw new RuntimeException(s"unsupported type found: $elt whose type info is $ti")
  })

  def create(len: Code[Int]): Code[Unit] = {
    ref := {
      ti match {
        case BooleanInfo => Code.newInstance[BooleanArrayBuilder, Int](len)
        case IntInfo => Code.newInstance[IntArrayBuilder, Int](len)
        case LongInfo => Code.newInstance[LongArrayBuilder, Int](len)
        case FloatInfo => Code.newInstance[FloatArrayBuilder, Int](len)
        case DoubleInfo => Code.newInstance[DoubleArrayBuilder, Int](len)
        case ti => throw new RuntimeException(s"unsupported type found: $elt whose type info is $ti")
      }
    }
  }

  def add(i: Code[_]): Code[Unit] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Boolean, Unit]("add", coerce[Boolean](i))
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Int, Unit]("add", coerce[Int](i))
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Long, Unit]("add", coerce[Long](i))
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Float, Unit]("add", coerce[Float](i))
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Double, Unit]("add", coerce[Double](i))
  }

  def apply(i: Code[Int]): Code[_] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Int, Boolean]("apply", i)
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Int, Int]("apply", i)
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Int, Long]("apply", i)
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Int, Float]("apply", i)
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Int, Double]("apply", i)
  }

  def size: Code[Int] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Int]("size")
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Int]("size")
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Int]("size")
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Int]("size")
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Int]("size")
  }

  def clear: Code[Unit] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Unit]("clear")
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Unit]("clear")
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Unit]("clear")
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Unit]("clear")
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Unit]("clear")
  }

}

class ByteArrayBuilder(initialCapacity: Int) {
  private var b: Array[Byte] = new Array[Byte](initialCapacity)
  private var size_ : Int = 0

  def size: Int = size_

  def apply(i: Int): Byte = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Byte](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Byte) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }
}

class IntArrayBuilder(initialCapacity: Int) {
  private var b: Array[Int] = new Array[Int](initialCapacity)
  private var size_ : Int = 0

  def size: Int = size_

  def apply(i: Int): Int = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Int](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Int) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def clear() {  size_ = 0 }
}

class LongArrayBuilder(initialCapacity: Int) {
  private var b: Array[Long] = new Array[Long](initialCapacity)
  private var size_ : Int = 0

  def size: Int = size_

  def apply(i: Int): Long = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Long](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Long) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def clear() {  size_ = 0 }
}

class FloatArrayBuilder(initialCapacity: Int) {
  private var b: Array[Float] = new Array[Float](initialCapacity)
  private var size_ : Int = 0

  def size: Int = size_

  def apply(i: Int): Float = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Float](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Float) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def clear() {  size_ = 0 }
}

class DoubleArrayBuilder(initialCapacity: Int) {
  private var b: Array[Double] = new Array[Double](initialCapacity)
  private var size_ : Int = 0

  def size: Int = size_

  def apply(i: Int): Double = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Double](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Double) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def clear() {  size_ = 0 }
}

class BooleanArrayBuilder(initialCapacity: Int) {
  private var b: Array[Boolean] = new Array[Boolean](initialCapacity)
  private var size_ : Int = 0

  def size: Int = size_

  def apply(i: Int): Boolean = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int) {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Boolean](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Boolean) {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def clear() {  size_ = 0 }
}