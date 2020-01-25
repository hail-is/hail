package is.hail.utils

import is.hail.io._

final class MemoryBufferWrapper {
  private val mb = new MemoryBuffer
  private val ib = new MemoryInputBuffer(mb)

  def buffer: InputBuffer = ib

  def clearAndSetFrom(address: Long, size: Int): Unit = {
    if (mb.capacity < size)
      mb.grow(size)
    mb.clear()
    mb.writeBytes(address, size)
  }
}

final class MemoryWriterWrapper {
  private val mb = new MemoryBuffer
  private val ob = new MemoryOutputBuffer(mb)

  def buffer: OutputBuffer = ob

  def clear(): Unit = mb.clear()

  def length(): Int = mb.end

  def copyToAddress(addr: Long): Unit = {
    mb.readBytes(addr, mb.end)
  }
}
