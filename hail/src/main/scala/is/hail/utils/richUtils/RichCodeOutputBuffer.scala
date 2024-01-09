package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.OutputBuffer
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SValue}
import is.hail.types.virtual._

class RichCodeOutputBuffer(
  val ob: Value[OutputBuffer]
) extends AnyVal {
  def flush(): Code[Unit] =
    ob.invoke[Unit]("flush")

  def close(): Code[Unit] =
    ob.invoke[Unit]("close")

  def indexOffset(): Code[Long] =
    ob.invoke[Long]("indexOffset")

  def writeByte(b: Code[Byte]): Code[Unit] =
    ob.invoke[Byte, Unit]("writeByte", b)

  def write(buf: Code[Array[Byte]]): Code[Unit] =
    ob.invoke[Array[Byte], Unit]("write", buf)

  def write(buf: Code[Array[Byte]], startPos: Code[Int], endPos: Code[Int]): Code[Unit] =
    ob.invoke[Array[Byte], Int, Int, Unit]("write", buf, startPos, endPos)

  def writeInt(i: Code[Int]): Code[Unit] =
    ob.invoke[Int, Unit]("writeInt", i)

  def writeLong(l: Code[Long]): Code[Unit] =
    ob.invoke[Long, Unit]("writeLong", l)

  def writeFloat(f: Code[Float]): Code[Unit] =
    ob.invoke[Float, Unit]("writeFloat", f)

  def writeDouble(d: Code[Double]): Code[Unit] =
    ob.invoke[Double, Unit]("writeDouble", d)

  def writeBytes(region: Code[Region], off: Code[Long], n: Code[Int]): Code[Unit] =
    ob.invoke[Region, Long, Int, Unit]("writeBytes", region, off, n)

  def writeBytes(addr: Code[Long], n: Code[Int]): Code[Unit] =
    ob.invoke[Long, Int, Unit]("writeBytes", addr, n)

  def writeDoubles(from: Code[Array[Double]], fromOff: Code[Int], n: Code[Int]): Code[Unit] =
    ob.invoke[Array[Double], Int, Int, Unit]("writeDoubles", from, fromOff, n)

  def writeDoubles(from: Code[Array[Double]]): Code[Unit] =
    ob.invoke[Array[Double], Unit]("writeDoubles", from)

  def writeBoolean(b: Code[Boolean]): Code[Unit] =
    ob.invoke[Boolean, Unit]("writeBoolean", b)

  def writeUTF(s: Code[String]): Code[Unit] =
    ob.invoke[String, Unit]("writeUTF", s)

  def writePrimitive(cb: EmitCodeBuilder, pc: SValue): Unit = pc.st.virtualType match {
    case TBoolean => cb += writeBoolean(pc.asBoolean.value)
    case TInt32 => cb += writeInt(pc.asInt.value)
    case TInt64 => cb += writeLong(pc.asLong.value)
    case TFloat32 => cb += writeFloat(pc.asFloat.value)
    case TFloat64 => cb += writeDouble(pc.asDouble.value)
  }
}
