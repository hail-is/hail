package is.hail.io

import is.hail.asm4s._
import is.hail.compatibility.{LEB128BufferSpec, LZ4BlockBufferSpec}
import is.hail.io.compress.LZ4
import is.hail.rvd.AbstractRVDSpec

import java.io._

import org.json4s.{JValue, ShortTypeHints}
import org.json4s.jackson.JsonMethods

object BufferSpec {
  // v1 legacy spec — preserved because the BGEN reader picks this for index
  // files written by older versions. Do not flip to v2.
  val zstdCompressionLEB: BufferSpec = LEB128BufferSpec(
    BlockingBufferSpec(64 * 1024, ZstdBlockBufferSpec(64 * 1024, new StreamBlockBufferSpec))
  )

  // v1 legacy spec — preserved for the same reason as zstdCompressionLEB.
  val lz4HCCompressionLEB: BufferSpec = LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024, LZ4HCBlockBufferSpec(32 * 1024, new StreamBlockBufferSpec))
  )

  val zstdCompression: BufferSpec =
    BlockingBufferSpec(64 * 1024, ZstdBlockBufferSpec(64 * 1024, new StreamBlockBufferSpec2))

  val default: BufferSpec = zstdCompression

  val blockedUncompressed: BufferSpec = BlockingBufferSpec(32 * 1024, new StreamBlockBufferSpec2)
  val unblockedUncompressed: BufferSpec = new StreamBufferSpec

  val wireSpec: BufferSpec =
    BlockingBufferSpec(
      64 * 1024,
      ZstdSizedBasedBlockBufferSpec(
        64 * 1024,
        /*minCompressionSize=*/ 256,
        new StreamBlockBufferSpec2,
      ),
    )

  val memorySpec: BufferSpec = wireSpec

  val specs: Array[BufferSpec] = Array(
    BlockingBufferSpec(64 * 1024, new StreamBlockBufferSpec2),
    BlockingBufferSpec(32 * 1024, LZ4HCBlockBufferSpec(32 * 1024, new StreamBlockBufferSpec2)),
    BlockingBufferSpec(32 * 1024, LZ4FastBlockBufferSpec(32 * 1024, new StreamBlockBufferSpec2)),
    BlockingBufferSpec(64 * 1024, ZstdBlockBufferSpec(64 * 1024, new StreamBlockBufferSpec2)),
  )

  def parse(s: String): BufferSpec = {
    import AbstractRVDSpec.formats
    JsonMethods.parse(s).extract[BufferSpec]
  }

  def parseOrDefault(
    s: String,
    default: BufferSpec = BufferSpec.default,
  ): BufferSpec = if (s == null) default else parse(s)

  val shortTypeHints = ShortTypeHints(
    List(
      classOf[BlockBufferSpec],
      classOf[LZ4BlockBufferSpec],
      classOf[LZ4HCBlockBufferSpec],
      classOf[LZ4FastBlockBufferSpec],
      classOf[LZ4SizeBasedBlockBufferSpec],
      classOf[ZstdBlockBufferSpec],
      classOf[StreamBlockBufferSpec],
      classOf[StreamBlockBufferSpec2],
      classOf[BufferSpec],
      classOf[LEB128BufferSpec],
      classOf[BlockingBufferSpec],
      classOf[StreamBufferSpec],
    ),
    typeHintFieldName = "name",
  )
}

trait BufferSpec extends Spec {
  def buildInputBuffer(in: InputStream): InputBuffer

  def buildOutputBuffer(out: OutputStream): OutputBuffer

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer]

  def buildCodeOutputBuffer(in: Code[OutputStream]): Code[OutputBuffer]
}

final case class BlockingBufferSpec(blockSize: Int, child: BlockBufferSpec) extends BufferSpec {
  require(blockSize <= (1 << 16))

  override def buildInputBuffer(in: InputStream): InputBuffer =
    new BlockingInputBuffer(blockSize, child.buildInputBuffer(in))

  override def buildOutputBuffer(out: OutputStream): OutputBuffer =
    new BlockingOutputBuffer(blockSize, child.buildOutputBuffer(out))

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer] =
    Code.newInstance[BlockingInputBuffer, Int, InputBlockBuffer](
      blockSize,
      child.buildCodeInputBuffer(in),
    )

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBuffer] =
    Code.newInstance[BlockingOutputBuffer, Int, OutputBlockBuffer](
      blockSize,
      child.buildCodeOutputBuffer(out),
    )
}

trait BlockBufferSpec extends Spec {
  def buildInputBuffer(in: InputStream): InputBlockBuffer

  def buildOutputBuffer(out: OutputStream): OutputBlockBuffer

  def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer]

  def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer]
}

abstract class LZ4BlockBufferSpecCommon extends BlockBufferSpec {
  require(blockSize <= (1 << 16))

  def typeName: String

  def lz4: LZ4

  def stagedlz4: Code[LZ4]

  def blockSize: Int

  def child: BlockBufferSpec

  override def buildInputBuffer(in: InputStream): InputBlockBuffer =
    new LZ4InputBlockBuffer(lz4, blockSize, child.buildInputBuffer(in))

  override def buildOutputBuffer(out: OutputStream): OutputBlockBuffer =
    new LZ4OutputBlockBuffer(lz4, blockSize, child.buildOutputBuffer(out))

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[LZ4InputBlockBuffer, LZ4, Int, InputBlockBuffer](
      stagedlz4,
      blockSize,
      child.buildCodeInputBuffer(in),
    )

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[LZ4OutputBlockBuffer, LZ4, Int, OutputBlockBuffer](
      stagedlz4,
      blockSize,
      child.buildCodeOutputBuffer(out),
    )
}

final case class LZ4HCBlockBufferSpec(blockSize: Int, child: BlockBufferSpec)
    extends LZ4BlockBufferSpecCommon {
  override def lz4 = LZ4.hc
  override def stagedlz4: Code[LZ4] = Code.invokeScalaObject0[LZ4](LZ4.getClass, "hc")
  override def typeName = "LZ4HCBlockBufferSpec"
}

final case class LZ4FastBlockBufferSpec(blockSize: Int, child: BlockBufferSpec)
    extends LZ4BlockBufferSpecCommon {
  override def lz4 = LZ4.fast
  override def stagedlz4: Code[LZ4] = Code.invokeScalaObject0[LZ4](LZ4.getClass, "fast")
  override def typeName = "LZ4FastBlockBufferSpec"
}

final case class LZ4SizeBasedBlockBufferSpec(
  compressorType: String,
  blockSize: Int,
  minCompressionSize: Int,
  child: BlockBufferSpec,
) extends BlockBufferSpec {
  def lz4: LZ4 = compressorType match {
    case "hc" => LZ4.hc
    case "fast" => LZ4.fast
  }

  def stagedlz4: Code[LZ4] = Code.invokeScalaObject0[LZ4](LZ4.getClass, "fast")
  def typeName = "LZ4SizeBasedBlockBufferSpec"

  override def buildInputBuffer(in: InputStream): InputBlockBuffer =
    new LZ4SizeBasedCompressingInputBlockBuffer(lz4, blockSize, child.buildInputBuffer(in))

  override def buildOutputBuffer(out: OutputStream): OutputBlockBuffer =
    new LZ4SizeBasedCompressingOutputBlockBuffer(
      lz4,
      blockSize,
      minCompressionSize,
      child.buildOutputBuffer(out),
    )

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[LZ4SizeBasedCompressingInputBlockBuffer, LZ4, Int, InputBlockBuffer](
      stagedlz4,
      blockSize,
      child.buildCodeInputBuffer(in),
    )

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[LZ4SizeBasedCompressingOutputBlockBuffer, LZ4, Int, Int, OutputBlockBuffer](
      stagedlz4,
      blockSize,
      minCompressionSize,
      child.buildCodeOutputBuffer(out),
    )
}

final case class ZstdBlockBufferSpec(blockSize: Int, child: BlockBufferSpec)
    extends BlockBufferSpec {
  require(blockSize <= (1 << 16))

  override def buildInputBuffer(in: InputStream): InputBlockBuffer =
    new ZstdInputBlockBuffer(blockSize, child.buildInputBuffer(in))

  override def buildOutputBuffer(out: OutputStream): OutputBlockBuffer =
    new ZstdOutputBlockBuffer(blockSize, child.buildOutputBuffer(out))

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[ZstdInputBlockBuffer, Int, InputBlockBuffer](
      blockSize,
      child.buildCodeInputBuffer(in),
    )

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[ZstdOutputBlockBuffer, Int, OutputBlockBuffer](
      blockSize,
      child.buildCodeOutputBuffer(out),
    )
}

final case class ZstdSizedBasedBlockBufferSpec(
  blockSize: Int,
  minCompressionSize: Int,
  child: BlockBufferSpec,
) extends BlockBufferSpec {
  require(blockSize <= (1 << 16))

  override def buildInputBuffer(in: InputStream): InputBlockBuffer =
    new ZstdSizedBasedInputBlockBuffer(blockSize, child.buildInputBuffer(in))

  override def buildOutputBuffer(out: OutputStream): OutputBlockBuffer =
    new ZstdSizedBasedOutputBlockBuffer(blockSize, minCompressionSize, child.buildOutputBuffer(out))

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[ZstdSizedBasedInputBlockBuffer, Int, InputBlockBuffer](
      blockSize,
      child.buildCodeInputBuffer(in),
    )

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[ZstdSizedBasedOutputBlockBuffer, Int, Int, OutputBlockBuffer](
      blockSize,
      minCompressionSize,
      child.buildCodeOutputBuffer(out),
    )
}

object StreamBlockBufferSpec {
  def extract(jv: JValue): StreamBlockBufferSpec = new StreamBlockBufferSpec
}

final class StreamBlockBufferSpec extends BlockBufferSpec {
  override def buildInputBuffer(in: InputStream): InputBlockBuffer = new StreamBlockInputBuffer(in)

  override def buildOutputBuffer(out: OutputStream): OutputBlockBuffer =
    new StreamBlockOutputBuffer(out)

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[StreamBlockInputBuffer, InputStream](in)

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[StreamBlockOutputBuffer, OutputStream](out)

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBlockBufferSpec]
}

object StreamBlockBufferSpec2 {
  def extract(jv: JValue): StreamBlockBufferSpec2 = new StreamBlockBufferSpec2
}

final class StreamBlockBufferSpec2 extends BlockBufferSpec {
  override def buildInputBuffer(in: InputStream): InputBlockBuffer = new StreamBlockInputBuffer2(in)

  override def buildOutputBuffer(out: OutputStream): OutputBlockBuffer =
    new StreamBlockOutputBuffer2(out)

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBlockBuffer] =
    Code.newInstance[StreamBlockInputBuffer2, InputStream](in)

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBlockBuffer] =
    Code.newInstance[StreamBlockOutputBuffer2, OutputStream](out)

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBlockBufferSpec2]
}

final class StreamBufferSpec extends BufferSpec {
  override def buildInputBuffer(in: InputStream): InputBuffer = new StreamInputBuffer(in)

  override def buildOutputBuffer(out: OutputStream): OutputBuffer = new StreamOutputBuffer(out)

  override def buildCodeInputBuffer(in: Code[InputStream]): Code[InputBuffer] =
    Code.newInstance[StreamInputBuffer, InputStream](in)

  override def buildCodeOutputBuffer(out: Code[OutputStream]): Code[OutputBuffer] =
    Code.newInstance[StreamOutputBuffer, OutputStream](out)

  override def equals(other: Any): Boolean = other.isInstanceOf[StreamBufferSpec]
}
