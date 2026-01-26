package is.hail.io

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.FunctionWithPartitionRegion
import is.hail.types.encoded.EncoderAsmFunction

import java.io._

trait Encoder extends Closeable {
  def flush(): Unit

  override def close(): Unit

  def writeRegionValue(region: Region, offset: Long): Unit

  def writeByte(b: Byte): Unit

  def indexOffset(): Long
}

final class CompiledEncoder(
  out: OutputBuffer,
  theHailClassLoader: HailClassLoader,
  f: (HailClassLoader) => EncoderAsmFunction,
) extends Encoder {
  private[this] var poolSet: Boolean = false
  private[this] var partitionRegion: Region = _
  private[this] val compiled = f(theHailClassLoader)

  override def flush(): Unit =
    out.flush()

  override def close(): Unit = {
    compiled.asInstanceOf[FunctionWithPartitionRegion].setPool(null)
    compiled.asInstanceOf[FunctionWithPartitionRegion].addPartitionRegion(null)
    if (partitionRegion != null) partitionRegion.close()
    out.close()
  }

  private def setScratchPool(pool: RegionPool) = if (!poolSet) {
    compiled.asInstanceOf[FunctionWithPartitionRegion].setPool(pool)
    partitionRegion = pool.getRegion()
    compiled.asInstanceOf[FunctionWithPartitionRegion].addPartitionRegion(partitionRegion)
    poolSet = true
  }

  override def writeRegionValue(r: Region, offset: Long): Unit = {
    setScratchPool(r.pool)
    writeRegionValue(offset)
  }

  private def writeRegionValue(offset: Long): Unit = {
    require(poolSet)
    compiled(offset, out)
  }

  override def writeByte(b: Byte): Unit =
    out.writeByte(b)

  override def indexOffset(): Long = out.indexOffset()
}

final class ByteArrayEncoder(
  theHailClassLoader: HailClassLoader,
  makeEnc: (OutputStream, HailClassLoader) => Encoder,
) extends Closeable {
  private[this] val baos = new ByteArrayOutputStream()
  private[this] val enc = makeEnc(baos, theHailClassLoader)

  override def close(): Unit = {
    enc.close()
    baos.close()
  }

  def regionValueToBytes(region: Region, offset: Long): Array[Byte] = {
    reset()
    writeRegionValue(region, offset)
    result()
  }

  def reset(): Unit = baos.reset()
  def writeRegionValue(region: Region, offset: Long): Unit = enc.writeRegionValue(region, offset)

  def result(): Array[Byte] = {
    enc.flush()
    baos.toByteArray
  }
}
