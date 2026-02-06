package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.physical.stypes.SingleCodeType

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

  override def ensureCapacity(n: Int): Unit = {
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

  override def ensureCapacity(n: Int): Unit = {
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

  override def ensureCapacity(n: Int): Unit = {
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

  override def ensureCapacity(n: Int): Unit = {
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

  override def ensureCapacity(n: Int): Unit = {
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
