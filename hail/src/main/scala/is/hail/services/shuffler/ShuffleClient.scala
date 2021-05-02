package is.hail.services.shuffler

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.service.ServiceBackendContext
import is.hail.expr.ir._
import is.hail.io._
import is.hail.services._
import is.hail.types.physical._
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.SCanonicalShufflePointerSettable
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.log4j.Logger

import java.io.{InputStream, OutputStream}
import java.net.Socket

object ShuffleClient {
  private[this] val log = Logger.getLogger(getClass.getName())

  def socket(ctx: Option[ExecuteContext]): Socket = ctx match {
    case None =>
      DeployConfig.get.socket("shuffler")
    case Some(ctx) =>
      DeployConfig.get.socket(
        "shuffler",
        ctx.backendContext.asInstanceOf[ServiceBackendContext].tokens())
  }

  def codeSocket(): Code[Socket] =
    Code.invokeScalaObject0[Socket](ShuffleClient.getClass, "socket")
}

object CompileTimeShuffleClient {
  def getSocketOnWorker(): Socket = DeployConfig.get.socket("shuffler")

  def create(cb: EmitCodeBuilder, shuffle: SCanonicalShufflePointerSettable): CompileTimeShuffleClient = {
    val ts = shuffle.st.virtualType

    val socket: Value[Socket] = cb.newField[Socket]("socket",
      Code.invokeScalaObject0[Socket](CompileTimeShuffleClient.getClass, "getSocketOnWorker"))

    val in: Value[InputBuffer] = cb.newField[InputBuffer]("ib", shuffleBufferSpec.buildCodeInputBuffer(socket.invoke[InputStream]("getInputStream")))
    val out: Value[OutputBuffer] = cb.newField[OutputBuffer]("ob", shuffleBufferSpec.buildCodeOutputBuffer(socket.invoke[OutputStream]("getOutputStream")))

    new CompileTimeShuffleClient(shuffle, socket, in, out)
  }
}

class CompileTimeShuffleClient(
  val uuid: SCanonicalShufflePointerSettable,
  socket: Value[Socket],
  in: Value[InputBuffer],
  out: Value[OutputBuffer]) {

  private[this] val ts = uuid.st.virtualType.asInstanceOf[TShuffle]

  def start(cb: EmitCodeBuilder, region: Value[Region]): Unit = {
    startOperation(cb, Wire.START)
    cb.logInfo(s"start")
    cb += Wire.writeTStruct(out, ts.rowType)
    cb += Wire.writeEBaseStruct(out, ts.rowEType)
    cb += Wire.writeSortFieldArray(out, ts.keyFields)
    cb.logInfo(s"using ${ts.keyEType}")
    cb += Wire.writeEBaseStruct(out, ts.keyEType)
    cb += out.flush()

    val bytes = cb.newLocal[Array[Byte]]("shuff_start_bytes", Wire.readByteArray(in))
    cb.ifx(bytes.length().cne(Wire.ID_SIZE),
      cb._fatal(s"shuffle error: uuid length mismatch: expect ${Wire.ID_SIZE}, got ", bytes.length().toS))
    uuid.storeFromBytes(cb, region, bytes)
    cb.logInfo(s"start done")
  }


  private def startOperation(cb: EmitCodeBuilder, op: Byte): Unit = {
    assert(op != Wire.EOS)
    assert(op != Wire.START)
    cb += out.writeByte(op)
    if (op != Wire.START) {
      val uuidBytes = cb.newLocal[Array[Byte]]("shuffle_uuid", uuid.loadBytes())
      cb.logInfo(s"operation $op uuid ", uuidToString(uuidBytes))
      cb += Wire.writeByteArray(out, uuidBytes)
    }
  }

  def startPut(cb: EmitCodeBuilder):Unit = {
    // will define later
    cb.logInfo(s"put")
    startOperation(cb, Wire.PUT)
    out.flush()
  }

  def putValue(cb: EmitCodeBuilder, value: SCode): Unit = {
    cb += out.writeByte(1.toByte)
    ts.rowEType.buildEncoder(value.st, cb.emb.ecb)
      .apply(cb, value, out)
  }

  def finishPut(cb: EmitCodeBuilder): Unit = {
    cb += out.writeByte(0.toByte)

    // fixme: server needs to send uuid for the successful partition
    cb += out.flush()
    val b = cb.newLocal[Byte]("finishPut_b", in.readByte())
    cb.ifx(b.get.toI.cne(const(0)), cb._fatal(s"bad shuffle put"))
    cb.logInfo(s"put done")
  }

  def startGet(cb: EmitCodeBuilder,
    _start: SCode,
    _startInclusive: Code[Boolean],
    _end: SCode,
    _endInclusive: Code[Boolean]
  ): Unit = {
    // will define later

    val start = _start.memoize(cb, "shuffle_start")
    val end = _end.memoize(cb, "shuffle_end")
    val startIncl = cb.newLocal[Boolean]("startIncl", _startInclusive)
    val endIncl = cb.newLocal[Boolean]("endIncl", _endInclusive)

    cb.logInfo("shuffle get: start=", cb.strValue(start.get), ", startInclusive=", startIncl.toS, ", end=", cb.strValue(end.get), ", endInclusive=", endIncl.toS)

    startOperation(cb, Wire.GET)

    ts.keyEType.buildEncoder(start.st, cb.emb.ecb)
      .apply(cb, start.get, out)
    cb.ifx(startIncl, cb += out.writeByte(1.toByte), cb += out.writeByte(0.toByte))
    ts.keyEType.buildEncoder(end.st, cb.emb.ecb)
      .apply(cb, end.get, out)
    cb.ifx(endIncl, cb += out.writeByte(1.toByte), cb += out.writeByte(0.toByte))

    cb += out.flush()
    cb.logInfo(s"get receiving values")
  }

  def getValueFinished(cb: EmitCodeBuilder): Code[Boolean] = in.readByte() ceq 0.toByte

  def readValue(cb: EmitCodeBuilder, region: Value[Region]): SCode = {
    ts.rowEType.buildDecoder(ts.rowType, cb.emb.ecb)
      .apply(cb, region, in)
  }

  def finishGet(cb: EmitCodeBuilder): Unit = {
    cb.logInfo(s"get done")
  }

  def startPartitionBounds(cb: EmitCodeBuilder, nPartitions: Code[Int]): Unit = {
      cb.logInfo(s"partitionBounds")
      startOperation(cb, Wire.PARTITION_BOUNDS)
      cb += out.writeInt(nPartitions)
      cb += out.flush()
      cb.logInfo(s"partitionBounds receiving values")
  }

  def readPartitionBound(cb: EmitCodeBuilder, region: Value[Region]): SCode = {
    ts.keyEType.buildDecoder(ts.keyType, cb.emb.ecb)
      .apply(cb, region, in)
  }

  def partitionBoundsFinished(cb: EmitCodeBuilder): Code[Boolean] = in.readByte() ceq 0.toByte

  def stop(cb: EmitCodeBuilder): Unit = {
    startOperation(cb, Wire.STOP)
    cb += out.flush()
    cb.logInfo(s"stop")
    val byte = cb.newLocal[Int]("shuffle_stop_byte", in.readByte().toI)
    cb.ifx(byte.cne(0), cb._fatal(s"bad byte in stop: ", byte.toS))
    cb.logInfo(s"stop done")
  }

  def close(cb: EmitCodeBuilder): Unit = {
    cb += out.writeByte(Wire.EOS)
    cb += out.flush()
    val byte = in.readByte()
    cb.ifx(byte.toI.cne(Wire.EOS.toInt), cb._fatal("bad shuffle close"))
    cb += socket.invoke[Unit]("close")
  }
}

class ShuffleClient (
  shuffleType: TShuffle,
  var uuid: Array[Byte],
  rowEncodingPType: Option[PType],
  keyEncodingPType: Option[PType],
  ctx: Option[ExecuteContext]
) extends AutoCloseable {
  private[this] val log = Logger.getLogger(getClass.getName())

  def this(shuffleType: TShuffle) = this(shuffleType, null, None, None, None)

  def this(shuffleType: TShuffle, ctx: ExecuteContext) = this(shuffleType, null, None, None, Some(ctx))

  def this(shuffleType: TShuffle, rowEncodingPType: PType, keyEncodingPType: PType) =
    this(shuffleType, null, Option(rowEncodingPType), Option(keyEncodingPType), None)

  def this(shuffleType: TShuffle, uuid: Array[Byte], rowEncodingPType: PType, keyEncodingPType: PType) =
    this(shuffleType, uuid, Option(rowEncodingPType), Option(keyEncodingPType), None)

  def this(shuffleType: TShuffle, uuid: Array[Byte], ctx: ExecuteContext) =
    this(shuffleType, uuid, None, None, Some(ctx))

  def this(shuffleType: TShuffle, uuid: Array[Byte]) =
    this(shuffleType, uuid, None, None, None)

  val codecs = ctx match {
    case None =>
      ExecutionTimer.logTime("ShuffleClient.codecs") { timer =>
        ExecuteContext.scoped("/tmp", "file:///tmp", null, null, timer, null) { ctx =>
          new ShuffleCodecSpec(ctx, shuffleType, rowEncodingPType, keyEncodingPType)
        }
      }
    case Some(ctx) =>
      new ShuffleCodecSpec(ctx, shuffleType, rowEncodingPType, keyEncodingPType)
  }

  private[this] val s = ShuffleClient.socket(ctx)
  private[this] val in = shuffleBufferSpec.buildInputBuffer(s.getInputStream())
  private[this] val out = shuffleBufferSpec.buildOutputBuffer(s.getOutputStream())

  private[this] def startOperation(op: Byte) = {
    assert(op != Wire.EOS)
    out.writeByte(op)
    if (op != Wire.START) {
      assert(uuid != null)
      log.info(s"operation $op uuid ${uuidToString(uuid)}")
      Wire.writeByteArray(out, uuid)
    }
  }

  def start(): Unit = {
    startOperation(Wire.START)
    log.info(s"start")
    Wire.writeTStruct(out, shuffleType.rowType)
    Wire.writeEBaseStruct(out, shuffleType.rowEType)
    Wire.writeSortFieldArray(out, shuffleType.keyFields)
    log.info(s"using ${shuffleType.keyEType}")
    Wire.writeEBaseStruct(out, shuffleType.keyEType)
    out.flush()
    uuid = Wire.readByteArray(in)
    assert(uuid.length == Wire.ID_SIZE, s"${uuid.length} ${Wire.ID_SIZE}")
    log.info(s"start done")
  }

  private[this] var encoder: Encoder = null

  def startPut(): Unit = {
    log.info(s"put")
    startOperation(Wire.PUT)
    out.flush()
    encoder = codecs.makeRowEncoder(out)
  }

  def put(values: Array[Long]): Unit = {
    startPut()
    writeRegionValueArray(encoder, values)
    endPut()
  }

  def putValue(value: Long): Unit = {
    encoder.writeByte(1)
    encoder.writeRegionValue(value)
  }

  def putValueDone(): Unit = {
    encoder.writeByte(0)
  }

  def endPut(): Unit = {
    // fixme: server needs to send uuid for the successful partition
    encoder.flush()
    assert(in.readByte() == 0.toByte)
    log.info(s"put done")
  }

  private[this] var decoder: Decoder = null

  def startGet(
    start: Long,
    startInclusive: Boolean,
    end: Long,
    endInclusive: Boolean
  ): Unit = {
    log.info(s"get ${Region.pretty(codecs.keyEncodingPType, start)} ${startInclusive} " +
      s"${Region.pretty(codecs.keyEncodingPType, end)} ${endInclusive}")
    val keyEncoder = codecs.makeKeyEncoder(out)
    decoder = codecs.makeRowDecoder(in)
    startOperation(Wire.GET)
    keyEncoder.writeRegionValue(start)
    keyEncoder.writeByte(if (startInclusive) 1.toByte else 0.toByte)
    keyEncoder.writeRegionValue(end)
    keyEncoder.writeByte(if (endInclusive) 1.toByte else 0.toByte)
    keyEncoder.flush()
    log.info(s"get receiving values")
  }

  def get(
    region: Region,
    start: Long,
    startInclusive: Boolean,
    end: Long,
    endInclusive: Boolean
  ): Array[Long] = {
    startGet(start, startInclusive, end, endInclusive)
    val values = readRegionValueArray(region, decoder)
    getDone()
    values
  }

  def getValue(region: Region): Long = {
    decoder.readRegionValue(region)
  }

  def getValueFinished(): Boolean = {
    decoder.readByte() == 0.toByte
  }

  def getDone(): Unit = {
    log.info(s"get done")
  }

  private[this] var keyDecoder: Decoder = null

  def startPartitionBounds(
    nPartitions: Int
  ): Unit = {
    log.info(s"partitionBounds")
    startOperation(Wire.PARTITION_BOUNDS)
    out.writeInt(nPartitions)
    out.flush()
    log.info(s"partitionBounds receiving values")
    keyDecoder = codecs.makeKeyDecoder(in)
  }

  def partitionBounds(region: Region, nPartitions: Int): Array[Long] = {
    startPartitionBounds(nPartitions)
    val keys = readRegionValueArray(region, keyDecoder, nPartitions + 1)
    endPartitionBounds()
    keys
  }

  def partitionBoundsValue(region: Region): Long = {
    keyDecoder.readRegionValue(region)
  }

  def partitionBoundsValueFinished(): Boolean = {
    val b = keyDecoder.readByte()
    assert(b == 0.toByte || b == 1.toByte, b)
    b == 0.toByte
  }

  def endPartitionBounds(): Unit = {
    log.info(s"partitionBounds done")
  }

  def stop(): Unit = {
    startOperation(Wire.STOP)
    out.flush()
    log.info(s"stop")
    val byte = in.readByte()
    assert(byte == 0.toByte, byte)
    log.info(s"stop done")
  }

  def close(): Unit = {
    out.writeByte(Wire.EOS)
    out.flush()
    val byte = in.readByte()
    assert(byte == Wire.EOS, byte)
    s.close()
  }
}
