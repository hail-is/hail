package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.types.Type

import scala.reflect.ClassTag

class StagedArrayBuilder(val elt: Type, mb: MethodBuilder, len: Code[Int]) {

  val ti = typeToTypeInfo(elt)

  val ref: Settable[Any] = coerce[Any](ti match {
    case BooleanInfo => mb.newLazyField[BooleanArrayBuilder]("zab")(Code.newInstance[BooleanArrayBuilder, Int](len))
    case IntInfo => mb.newLazyField[IntArrayBuilder]("iab")(Code.newInstance[IntArrayBuilder, Int](len))
    case LongInfo => mb.newLazyField[LongArrayBuilder]("jab")(Code.newInstance[LongArrayBuilder, Int](len))
    case FloatInfo => mb.newLazyField[FloatArrayBuilder]("fab")(Code.newInstance[FloatArrayBuilder, Int](len))
    case DoubleInfo => mb.newLazyField[DoubleArrayBuilder]("dab")(Code.newInstance[DoubleArrayBuilder, Int](len))
    case ti => throw new RuntimeException(s"unsupported type found: $elt whose type info is $ti")
  })

  def add(x: Code[_]): Code[Unit] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Boolean, Unit]("add", coerce[Boolean](x))
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Int, Unit]("add", coerce[Int](x))
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Long, Unit]("add", coerce[Long](x))
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Float, Unit]("add", coerce[Float](x))
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Double, Unit]("add", coerce[Double](x))
  }

  def apply(i: Code[Int]): Code[_] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Int, Boolean]("apply", i)
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Int, Int]("apply", i)
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Int, Long]("apply", i)
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Int, Float]("apply", i)
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Int, Double]("apply", i)
  }

  def update(i: Code[Int], x: Code[_]): Code[Unit] = ti match {
    case BooleanInfo => coerce[BooleanArrayBuilder](ref).invoke[Int, Boolean, Unit]("update", i, coerce[Boolean](x))
    case IntInfo => coerce[IntArrayBuilder](ref).invoke[Int, Int, Unit]("update", i, coerce[Int](x))
    case LongInfo => coerce[LongArrayBuilder](ref).invoke[Int, Long, Unit]("update", i, coerce[Long](x))
    case FloatInfo => coerce[FloatArrayBuilder](ref).invoke[Int, Float, Unit]("update", i, coerce[Float](x))
    case DoubleInfo => coerce[DoubleArrayBuilder](ref).invoke[Int, Double, Unit]("update", i, coerce[Double](x))
  }

  def sort(compare: Code[AsmFunction4[_, Boolean, _, Boolean, Boolean]]): Code[Unit] = {
    ti match {
      case BooleanInfo =>
        type F = AsmFunction4[Boolean, Boolean, Boolean, Boolean, Boolean]
        coerce[BooleanArrayBuilder](ref).invoke[F, Unit]("sort", coerce[F](compare))
      case IntInfo =>
        type F = AsmFunction4[Int, Boolean, Int, Boolean, Boolean]
        coerce[IntArrayBuilder](ref).invoke[F, Unit]("sort", coerce[F](compare))
      case LongInfo =>
        type F = AsmFunction4[Long, Boolean, Long, Boolean, Boolean]
        coerce[LongArrayBuilder](ref).invoke[F, Unit]("sort", coerce[F](compare))
      case FloatInfo =>
        type F = AsmFunction4[Float, Boolean, Float, Boolean, Boolean]
        coerce[FloatArrayBuilder](ref).invoke[F, Unit]("sort", coerce[F](compare))
      case DoubleInfo =>
        type F = AsmFunction4[Double, Boolean, Double, Boolean, Boolean]
        coerce[DoubleArrayBuilder](ref).invoke[F, Unit]("sort", coerce[F](compare))
    }
  }

  def addMissing(): Code[Unit] =
    coerce[MissingArrayBuilder](ref).invoke[Unit]("addMissing")

  def isMissing(i: Code[Int]): Code[Boolean] =
    coerce[MissingArrayBuilder](ref).invoke[Int, Boolean]("isMissing", i)

  def setMissing(i: Code[Int], m: Code[Boolean]): Code[Unit] =
    coerce[MissingArrayBuilder](ref).invoke[Int, Boolean, Unit]("setMissing", i, m)

  def size: Code[Int] = coerce[MissingArrayBuilder](ref).invoke[Int]("size")

  def setSize(n: Code[Int]): Code[Unit] = coerce[MissingArrayBuilder](ref).invoke[Int, Unit]("setSize", n)

  def clear: Code[Unit] = coerce[MissingArrayBuilder](ref).invoke[Unit]("clear")
}

sealed abstract class MissingArrayBuilder(initialCapacity: Int) {

  var missing: Array[Boolean] = new Array[Boolean](initialCapacity)
  var size_ : Int = 0

  def size: Int = size_

  def isMissing(i: Int): Boolean = {
    require(i >= 0 && i < size)
    missing(i)
  }

  def ensureCapacity(n: Int): Unit

  def setMissing(i: Int, m: Boolean): Unit = {
    require(i >= 0 && i < size)
    missing(i) = m
  }

  def addMissing() {
    ensureCapacity(size_ + 1)
    missing(size_) = true
    size_ += 1
  }

  def setSize(n: Int) {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def clear() {  size_ = 0 }
}

class ByteArrayBuilder(initialCapacity: Int) extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Byte] = new Array[Byte](initialCapacity)

  def apply(i: Int): Byte = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Byte](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def add(x: Byte): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def update(i: Int, x: Byte): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
    missing(i) = false
  }

  def sort(ordering: AsmFunction2[Byte, Byte, Boolean]): Unit = {
    var newend = 0
    var i = 0
    while (i < size_) {
      if (!missing(i)) {

      }

      i += 1
    }
    val (newmissing, newb) = missing.zip(b).take(size_).sortWith { (x, y) => ordering(x._2, x._1, y._2, y._1) }.unzip
    missing = newmissing
    b = newb
  }
}

class IntArrayBuilder(initialCapacity: Int) extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Int] = new Array[Int](initialCapacity)

  def apply(i: Int): Int = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Int](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def add(x: Int): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def update(i: Int, x: Int): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
    missing(i) = false
  }

  def sort(ordering: AsmFunction4[Int, Boolean, Int, Boolean, Boolean]): Unit = {
    val (newmissing, newb) = missing.zip(b).take(size_).sortWith { (x, y) => ordering(x._2, x._1, y._2, y._1) }.unzip
    missing = newmissing
    b = newb
  }
}

class LongArrayBuilder(initialCapacity: Int) extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Long] = new Array[Long](initialCapacity)

  def apply(i: Int): Long = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Long](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def add(x: Long): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def update(i: Int, x: Long): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
    missing(i) = false
  }

  def sort(ordering: AsmFunction4[Long, Boolean, Long, Boolean, Boolean]): Unit = {
    val (newmissing, newb) = missing.zip(b).take(size_).sortWith { (x, y) => ordering(x._2, x._1, y._2, y._1) }.unzip
    missing = newmissing
    b = newb
  }
}

class FloatArrayBuilder(initialCapacity: Int) extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Float] = new Array[Float](initialCapacity)

  def apply(i: Int): Float = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Float](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def add(x: Float): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def update(i: Int, x: Float): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
    missing(i) = false
  }

  def sort(ordering: AsmFunction4[Float, Boolean, Float, Boolean, Boolean]): Unit = {
    val (newmissing, newb) = missing.zip(b).take(size_).sortWith { (x, y) => ordering(x._2, x._1, y._2, y._1) }.unzip
    missing = newmissing
    b = newb
  }
}

class DoubleArrayBuilder(initialCapacity: Int) extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Double] = new Array[Double](initialCapacity)

  def apply(i: Int): Double = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Double](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def add(x: Double): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def update(i: Int, x: Double): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
    missing(i) = false
  }

  def sort(ordering: AsmFunction4[Double, Boolean, Double, Boolean, Boolean]): Unit = {
    val (newmissing, newb) = missing.zip(b).take(size_).sortWith { (x, y) => ordering(x._2, x._1, y._2, y._1) }.unzip
    missing = newmissing
    b = newb
  }
}

class BooleanArrayBuilder(initialCapacity: Int) extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Boolean] = new Array[Boolean](initialCapacity)

  def apply(i: Int): Boolean = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = (b.length * 2).max(n)
      val newb = new Array[Boolean](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
      val newmissing = new Array[Boolean](newCapacity)
      Array.copy(missing, 0, newmissing, 0, size_)
      missing = newmissing
    }
  }

  def add(x: Boolean): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    missing(size_) = false
    size_ += 1
  }

  def update(i: Int, x: Boolean): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
    missing(i) = false
  }

  def sort(ordering: AsmFunction4[Boolean, Boolean, Boolean, Boolean, Boolean]): Unit = {
    val (newmissing, newb) = missing.zip(b).take(size_).sortWith { (x, y) => ordering(x._2, x._1, y._2, y._1) }.unzip
    missing = newmissing
    b = newb
  }
}