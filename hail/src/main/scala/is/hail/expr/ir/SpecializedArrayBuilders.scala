package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.physical.stypes.SingleCodeType

import scala.reflect.ClassTag

class StagedArrayBuilder(
  cb: EmitCodeBuilder,
  val elt: SingleCodeType,
  val eltRequired: Boolean,
  len: Int,
) {

  def mb = cb.emb
  val ti: TypeInfo[_] = elt.ti

  val ref: Value[Any] = coerce[Any](ti match {
    case BooleanInfo => mb.genLazyFieldThisRef[BooleanMissingArrayBuilder](
        Code.newInstance[BooleanMissingArrayBuilder, Int](len),
        "zab",
      )
    case IntInfo => mb.genLazyFieldThisRef[IntMissingArrayBuilder](
        Code.newInstance[IntMissingArrayBuilder, Int](len),
        "iab",
      )
    case LongInfo => mb.genLazyFieldThisRef[LongMissingArrayBuilder](
        Code.newInstance[LongMissingArrayBuilder, Int](len),
        "jab",
      )
    case FloatInfo => mb.genLazyFieldThisRef[FloatMissingArrayBuilder](
        Code.newInstance[FloatMissingArrayBuilder, Int](len),
        "fab",
      )
    case DoubleInfo => mb.genLazyFieldThisRef[DoubleMissingArrayBuilder](
        Code.newInstance[DoubleMissingArrayBuilder, Int](len),
        "dab",
      )
    case ti => throw new RuntimeException(s"unsupported typeinfo found: $ti")
  })

  // If a method containing `new StagedArrayBuilder(...)` is called multiple times,
  // the invocations will share the same array builder at runtime. Clearing
  // here ensures a "new" array builder is always empty.
  clear(cb)
  ensureCapacity(cb, len)

  def add(cb: EmitCodeBuilder, x: Code[_]): Unit = cb.append(ti match {
    case BooleanInfo =>
      coerce[BooleanMissingArrayBuilder](ref).invoke[Boolean, Unit]("add", coerce[Boolean](x))
    case IntInfo => coerce[IntMissingArrayBuilder](ref).invoke[Int, Unit]("add", coerce[Int](x))
    case LongInfo => coerce[LongMissingArrayBuilder](ref).invoke[Long, Unit]("add", coerce[Long](x))
    case FloatInfo =>
      coerce[FloatMissingArrayBuilder](ref).invoke[Float, Unit]("add", coerce[Float](x))
    case DoubleInfo =>
      coerce[DoubleMissingArrayBuilder](ref).invoke[Double, Unit]("add", coerce[Double](x))
  })

  def apply(i: Code[Int]): Code[_] = ti match {
    case BooleanInfo => coerce[BooleanMissingArrayBuilder](ref).invoke[Int, Boolean]("apply", i)
    case IntInfo => coerce[IntMissingArrayBuilder](ref).invoke[Int, Int]("apply", i)
    case LongInfo => coerce[LongMissingArrayBuilder](ref).invoke[Int, Long]("apply", i)
    case FloatInfo => coerce[FloatMissingArrayBuilder](ref).invoke[Int, Float]("apply", i)
    case DoubleInfo => coerce[DoubleMissingArrayBuilder](ref).invoke[Int, Double]("apply", i)
  }

  def update(cb: EmitCodeBuilder, i: Code[Int], x: Code[_]): Unit = cb.append(ti match {
    case BooleanInfo => coerce[BooleanMissingArrayBuilder](ref).invoke[Int, Boolean, Unit](
        "update",
        i,
        coerce[Boolean](x),
      )
    case IntInfo =>
      coerce[IntMissingArrayBuilder](ref).invoke[Int, Int, Unit]("update", i, coerce[Int](x))
    case LongInfo =>
      coerce[LongMissingArrayBuilder](ref).invoke[Int, Long, Unit]("update", i, coerce[Long](x))
    case FloatInfo =>
      coerce[FloatMissingArrayBuilder](ref).invoke[Int, Float, Unit]("update", i, coerce[Float](x))
    case DoubleInfo => coerce[DoubleMissingArrayBuilder](ref).invoke[Int, Double, Unit](
        "update",
        i,
        coerce[Double](x),
      )
  })

  def addMissing(cb: EmitCodeBuilder): Unit =
    cb += coerce[MissingArrayBuilder](ref).invoke[Unit]("addMissing")

  def isMissing(i: Code[Int]): Code[Boolean] =
    coerce[MissingArrayBuilder](ref).invoke[Int, Boolean]("isMissing", i)

  def setMissing(cb: EmitCodeBuilder, i: Code[Int], m: Code[Boolean]): Unit =
    cb += coerce[MissingArrayBuilder](ref).invoke[Int, Boolean, Unit]("setMissing", i, m)

  def size: Code[Int] = coerce[MissingArrayBuilder](ref).invoke[Int]("size")

  def setSize(cb: EmitCodeBuilder, n: Code[Int]): Unit =
    cb += coerce[MissingArrayBuilder](ref).invoke[Int, Unit]("setSize", n)

  def ensureCapacity(cb: EmitCodeBuilder, n: Code[Int]): Unit =
    cb += coerce[MissingArrayBuilder](ref).invoke[Int, Unit]("ensureCapacity", n)

  def clear(cb: EmitCodeBuilder): Unit =
    cb += coerce[MissingArrayBuilder](ref).invoke[Unit]("clear")

  def loadFromIndex(cb: EmitCodeBuilder, r: Value[Region], i: Value[Int]): IEmitCode =
    IEmitCode(cb, isMissing(i), elt.loadToSValue(cb, cb.memoizeAny(apply(i), ti)))
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
    require(i >= 0 && i < size, i)
    missing(i) = m
  }

  def addMissing(): Unit = {
    ensureCapacity(size_ + 1)
    missing(size_) = true
    size_ += 1
  }

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def clear(): Unit = size_ = 0
}

final class IntMissingArrayBuilder(initialCapacity: Int)
    extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Int] = new Array[Int](initialCapacity)

  def apply(i: Int): Int = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
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

  def sort(ordering: AsmFunction2[Int, Int, Boolean]): Unit = {
    var newend = 0
    var i = 0
    while (i < size) {
      if (!isMissing(i)) {
        if (newend != i) {
          update(newend, b(i))
        }
        newend += 1
      }
      i += 1
    }
    i = newend
    while (i < size) {
      setMissing(i, true)
      i += 1
    }
    val newb = b.take(newend).sortWith(ordering(_, _))
    i = 0
    while (i < newend) {
      update(i, newb(i))
      i += 1
    }
  }
}

final class LongMissingArrayBuilder(initialCapacity: Int)
    extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Long] = new Array[Long](initialCapacity)

  def apply(i: Int): Long = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
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

  def sort(ordering: AsmFunction2[Long, Long, Boolean]): Unit = {
    var newend = 0
    var i = 0
    while (i < size) {
      if (!isMissing(i)) {
        if (newend != i) {
          update(newend, b(i))
        }
        newend += 1
      }
      i += 1
    }
    i = newend
    while (i < size) {
      setMissing(i, true)
      i += 1
    }
    val newb = b.take(newend).sortWith(ordering(_, _))
    i = 0
    while (i < newend) {
      update(i, newb(i))
      i += 1
    }
  }
}

final class FloatMissingArrayBuilder(initialCapacity: Int)
    extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Float] = new Array[Float](initialCapacity)

  def apply(i: Int): Float = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
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

  def sort(ordering: AsmFunction2[Float, Float, Boolean]): Unit = {
    var newend = 0
    var i = 0
    while (i < size) {
      if (!isMissing(i)) {
        if (newend != i) {
          update(newend, b(i))
        }
        newend += 1
      }
      i += 1
    }
    i = newend
    while (i < size) {
      setMissing(i, true)
      i += 1
    }
    val newb = b.take(newend).sortWith(ordering(_, _))
    i = 0
    while (i < newend) {
      update(i, newb(i))
      i += 1
    }
  }
}

final class DoubleMissingArrayBuilder(initialCapacity: Int)
    extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Double] = new Array[Double](initialCapacity)

  def apply(i: Int): Double = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
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

  def sort(ordering: AsmFunction2[Double, Double, Boolean]): Unit = {
    var newend = 0
    var i = 0
    while (i < size) {
      if (!isMissing(i)) {
        if (newend != i) {
          update(newend, b(i))
        }
        newend += 1
      }
      i += 1
    }
    i = newend
    while (i < size) {
      setMissing(i, true)
      i += 1
    }
    val newb = b.take(newend).sortWith(ordering(_, _))
    i = 0
    while (i < newend) {
      update(i, newb(i))
      i += 1
    }
  }
}

final class BooleanMissingArrayBuilder(initialCapacity: Int)
    extends MissingArrayBuilder(initialCapacity) {
  private var b: Array[Boolean] = new Array[Boolean](initialCapacity)

  def apply(i: Int): Boolean = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
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

  def sort(ordering: AsmFunction2[Boolean, Boolean, Boolean]): Unit = {
    var newend = 0
    var i = 0
    while (i < size) {
      if (!isMissing(i)) {
        if (newend != i) {
          update(newend, b(i))
        }
        newend += 1
      }
      i += 1
    }
    i = newend
    while (i < size) {
      setMissing(i, true)
      i += 1
    }
    val newb = b.take(newend).sortWith(ordering(_, _))
    i = 0
    while (i < newend) {
      update(i, newb(i))
      i += 1
    }
  }
}

final class ByteArrayArrayBuilder(initialCapacity: Int) {

  var size_ : Int = 0
  private var b: Array[Array[Byte]] = new Array[Array[Byte]](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Array[Byte] = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Array[Byte]](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: Array[Byte]): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Array[Byte]): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Array[Byte]] = b.slice(0, size_)
}

final class LongArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Long] = new Array[Long](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Long = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, (b.length * 2))
      val newb = new Array[Long](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: Long): Unit = add(x)

  def ++=(xs: Array[Long]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Long], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Long): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Long): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Long] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Long](initialCapacity)
  }

  def appendFrom(ab2: LongArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Long = {
    size_ -= 1
    b(size)
  }
}

final class IntArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Int] = new Array[Int](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Int = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Int](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def setSizeUninitialized(n: Int) = {
    ensureCapacity(n)
    size_ = n
  }

  def +=(x: Int): Unit = add(x)

  def ++=(xs: Array[Int]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Int], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Int): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Int): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Int] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Int](initialCapacity)
  }

  def appendFrom(ab2: IntArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Int = {
    size_ -= 1
    b(size)
  }
}

final class DoubleArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Double] = new Array[Double](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Double = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Double](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def setSizeUninitialized(n: Int) = {
    ensureCapacity(n)
    size_ = n
  }

  def +=(x: Double): Unit = add(x)

  def ++=(xs: Array[Double]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Double], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Double): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Double): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Double] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Double](initialCapacity)
  }

  def appendFrom(ab2: DoubleArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Double = {
    size_ -= 1
    b(size)
  }
}

final class ByteArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Byte] = new Array[Byte](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def setSizeUnchecked(n: Int): Unit =
    size_ = n

  def apply(i: Int): Byte = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Byte](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: Byte) = add(x)

  def ++=(xs: Array[Byte]) = {
    val newLen = size + xs.length
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, xs.length)
    size_ = newLen
  }

  def ++=(xs: Array[Byte], n: Int) = {
    val newLen = size + n
    ensureCapacity(newLen)
    System.arraycopy(xs, 0, b, size_, n)
    size_ = newLen
  }

  def add(x: Byte): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Byte): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Byte] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Byte](initialCapacity)
  }

  def appendFrom(ab2: ByteArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Byte = {
    size_ -= 1
    b(size)
  }
}

final class BooleanArrayBuilder(initialCapacity: Int = 16) {

  var size_ : Int = 0
  var b: Array[Boolean] = new Array[Boolean](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): Boolean = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[Boolean](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: Boolean) = add(x)

  def add(x: Boolean): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: Boolean): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[Boolean] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[Boolean](initialCapacity)
  }

  def appendFrom(ab2: BooleanArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): Boolean = {
    size_ -= 1
    b(size)
  }
}

final class StringArrayBuilder(initialCapacity: Int = 16) {
  var size_ : Int = 0
  var b: Array[String] = new Array[String](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): String = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[String](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def +=(x: String) = add(x)

  def add(x: String): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def update(i: Int, x: String): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[String] = {
    val a = new Array[String](size_)
    System.arraycopy(b, 0, a, 0, size_)
    a
  }

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[String](initialCapacity)
  }

  def appendFrom(ab2: StringArrayBuilder): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): String = {
    size_ -= 1
    b(size)
  }
}

final class AnyRefArrayBuilder[T <: AnyRef](initialCapacity: Int = 16)(implicit ct: ClassTag[T]) {

  var size_ : Int = 0
  var b: Array[T] = new Array[T](initialCapacity)

  def size: Int = size_

  def setSize(n: Int): Unit = {
    require(n >= 0 && n <= size)
    size_ = n
  }

  def apply(i: Int): T = {
    require(i >= 0 && i < size)
    b(i)
  }

  def ensureCapacity(n: Int): Unit = {
    if (b.length < n) {
      val newCapacity = math.max(n, b.length * 2)
      val newb = new Array[T](newCapacity)
      Array.copy(b, 0, newb, 0, size_)
      b = newb
    }
  }

  def add(x: T): Unit = {
    ensureCapacity(size_ + 1)
    b(size_) = x
    size_ += 1
  }

  def +=(x: T): Unit = add(x)

  def update(i: Int, x: T): Unit = {
    require(i >= 0 && i < size)
    b(i) = x
  }

  def clear(): Unit = size_ = 0

  def result(): Array[T] = b.slice(0, size_)

  def clearAndResize(): Unit = {
    size_ = 0
    if (b.length > initialCapacity)
      b = new Array[T](initialCapacity)
  }

  def appendFrom(ab2: AnyRefArrayBuilder[T]): Unit = {
    ensureCapacity(size_ + ab2.size_)
    System.arraycopy(ab2.b, 0, b, size_, ab2.size_)
    size_ = size_ + ab2.size_
  }

  def pop(): T = {
    size_ -= 1
    b(size)
  }

  def clearAndSetNull(): Unit = {
    clear()
    var i = 0
    while (i < b.length) {
      b(i) = (null.asInstanceOf[T])
      i += 1
    }
  }

  def setSizeUninitialized(size: Int): Unit = {
    ensureCapacity(size)
    size_ = size
  }
}
