package is.hail.io

import is.hail.rvd.AbstractRVDSpec
import java.io._

import is.hail.asm4s._
import is.hail.io.compress.LZ4Utils
import org.json4s.{ DefaultFormats, Formats, ShortTypeHints }

import org.json4s.jackson.JsonMethods
import org.json4s.{Extraction, JValue}

object BufferSpec {
  val default: BufferSpec = LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024,
      LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)))

  val defaultUncompressed: BufferSpec = BlockingBufferSpec(32 * 1024,
    new StreamBlockBufferSpec)

  val unblockedUncompressed: BufferSpec = new StreamBufferSpec

  val blockSpecs: Array[BufferSpec] = Array(
    BlockingBufferSpec(64 * 1024,
      new StreamBlockBufferSpec),
    BlockingBufferSpec(32 * 1024,
      LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec)),
    new StreamBufferSpec)

  val specs: Array[BufferSpec] = blockSpecs.flatMap { blockSpec =>
    Array(blockSpec, LEB128BufferSpec(blockSpec))
  }

  val wireSpec: BufferSpec = default
  val memorySpec: BufferSpec = default

  def parse(s: String): BufferSpec = {
    import AbstractRVDSpec.formats
    JsonMethods.parse(s).extract[BufferSpec]
  }

  def parseOrDefault(
    s: String,
    default: BufferSpec = BufferSpec.default
  ): BufferSpec = if (s == null) default else parse(s)

  val shortTypeHints = ShortTypeHints(List(
      classOf[BlockBufferSpec],
      classOf[LZ4BlockBufferSpec],
      classOf[StreamBlockBufferSpec],
      classOf[BufferSpec],
      classOf[LEB128BufferSpec],
      classOf[BlockingBufferSpec],
      classOf[StreamBufferSpec]
    ))
}

trait BufferSpec extends Spec {
  def buildInputBuffer(in: InputStream): InputBuffer

  def buildOutputBuffer(out: OutputStream): OutputBuffer

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer]

  def buildCodeOutputBuffer(in: Code[OutputStream]): Code[OutputBuffer]

  def nativeOutputBufferType: String

  def nativeInputBufferType(inputStreamType: String): String
}

final case class LEB128BufferSpec(child: BufferSpec) extends BufferSpec {
  def buildInputBuffer(in: InputStream): InputBuffer = new LEB128InputBuffer(child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new LEB128OutputBuffer(child.buildOutputBuffer(out))

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer] =
    Code.newInstance[LEB128InputBuffer, InputBuffer](child.buildCodeInputBuffer(in))

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBuffer] =
    Code.newInstance[LEB128OutputBuffer, OutputBuffer](child.buildCodeOutputBuffer(out))

  def nativeOutputBufferType: String = s"LEB128OutputBuffer<${ child.nativeOutputBufferType }>"

  def nativeInputBufferType(inputStreamType: String): String =
    s"LEB128InputBuffer<${ child.nativeInputBufferType(inputStreamType) }, $inputStreamType>"
}

final case class BlockingBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BufferSpec {
  require(blockSize <= (1 << 16))

  def buildInputBuffer(in: InputStream): InputBuffer = new BlockingInputBuffer(blockSize, child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBuffer = new BlockingOutputBuffer(blockSize, child.buildOutputBuffer(out))

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer] =
    Code.newInstance[BlockingInputBuffer, Int, InputBlockBuffer](blockSize, child.buildCodeInputBuffer(in))

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBuffer] =
    Code.newInstance[BlockingOutputBuffer, Int, OutputBlockBuffer](blockSize, child.buildCodeOutputBuffer(out))

  def nativeOutputBufferType: String = s"BlockingOutputBuffer<$blockSize, ${ child.nativeOutputBufferType }>"

  def nativeInputBufferType(inputStreamType: String): String =
    s"BlockingInputBuffer<$blockSize, ${ child.nativeInputBufferType(inputStreamType) }, $inputStreamType>"
}

trait BlockBufferSpec extends Spec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer]

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer]

  def nativeOutputBufferType: String

  def nativeInputBufferType(inputStreamType: String): String
}

final case class LZ4BlockBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BlockBufferSpec {
  require(blockSize <= (1 << 16))

  def buildInputBuffer(in: InputStream): InputBlockBuffer = new LZ4InputBlockBuffer(blockSize, child.buildInputBuffer(in))

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer = new LZ4OutputBlockBuffer(blockSize, child.buildOutputBuffer(out))

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[LZ4InputBlockBuffer, Int, InputBlockBuffer](blockSize, child.buildCodeInputBuffer(in))

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[LZ4OutputBlockBuffer, Int, OutputBlockBuffer](blockSize, child.buildCodeOutputBuffer(out))

  def nativeOutputBufferType: String = s"LZ4OutputBlockBuffer<${ 4 + LZ4Utils.maxCompressedLength(blockSize) }, ${ child.nativeOutputBufferType }>"

  def nativeInputBufferType(inputStreamType: String): String =
    s"LZ4InputBlockBuffer<${ 4 + LZ4Utils.maxCompressedLength(blockSize) }, ${ child.nativeInputBufferType(inputStreamType) }, $inputStreamType>"
}

object StreamBlockBufferSpec {
  def extract(jv: JValue): StreamBlockBufferSpec = new StreamBlockBufferSpec
}

final class StreamBlockBufferSpec extends BlockBufferSpec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer = new StreamBlockInputBuffer(in)

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer = new StreamBlockOutputBuffer(out)

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[StreamBlockInputBuffer, InputStream](in)

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[StreamBlockOutputBuffer, OutputStream](out)

  def nativeOutputBufferType: String = s"StreamOutputBlockBuffer"

  def nativeInputBufferType(inputStreamType: String): String = s"StreamInputBlockBuffer<$inputStreamType>"

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBlockBufferSpec]
}

final class StreamBufferSpec extends BufferSpec {
  override def buildInputBuffer(in: InputStream): InputBuffer = new StreamInputBuffer(in)

  override def buildOutputBuffer(out: OutputStream): OutputBuffer = new StreamOutputBuffer(out)

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer] =
    Code.newInstance[StreamInputBuffer, InputStream](in)

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBuffer] =
    Code.newInstance[StreamOutputBuffer, OutputStream](out)

  override def nativeOutputBufferType: String = s"StreamOutputBuffer"

  override def nativeInputBufferType(inputStreamType: String): String = s"StreamInputBuffer<$inputStreamType>"

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBufferSpec]
}
