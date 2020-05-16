package is.hail.shuffler

import java.net.Socket

import is.hail._
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.expr.types.encoded._
import is.hail.expr.types.virtual._
import is.hail.utils._
import javax.net.ssl._
import org.apache.log4j.Logger

object ShuffleClient {
  private[this] val log = Logger.getLogger(getClass.getName())

  lazy val host = {
    val x = HailContext.get.flags.getOrNull("shuffler_host")
    if (x == null) "localhost" else x
  }
  lazy val port = {
    val x = HailContext.get.flags.getOrNull("shuffler_port")
    if (x == null) 8080 else x.toInt
  }
  lazy val sslContext = {
    val key = HailContext.get.flags.getOrNull("shuffler_ssl_client_key_file")
    val cert = HailContext.get.flags.getOrNull("shuffler_ssl_client_cert_file")
    if (key == null && cert != null ||
      cert == null && key != null) {
      fatal("you must specify both or neither of the hail context flags: " +
        "shuffler_ssl_client_key_file and shuffler_ssl_client_cert_file")
    }
    if (key == null) {
      is.hail.shuffler.sslContext(
        getClass.getResourceAsStream("/non-secret-key-and-trust-stores/client-keystore.p12"),
        "hailhail",
        getClass.getResourceAsStream("/non-secret-key-and-trust-stores/client-truststore.p12"),
        "hailhail"
      )
    } else {
      is.hail.shuffler.sslContext(key, "", cert, "")
    }
  }

  def socket(): Socket = socket(host, port)

  def socket(host: String, port: Int): Socket = {
    val s = sslContext.getSocketFactory().createSocket(host, port)
    log.info(s"CLNT connected to ${host}:${port} (socket())")
    s
  }
}

class ShuffleClient (
  keyFields: Array[SortField],
  rowType: TStruct,
  rowEType: EBaseStruct,
  keyEType: EBaseStruct,
  ssl: SSLContext,
  host: String = ShuffleClient.host,
  port: Int = ShuffleClient.port
) {
  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] var uuid: Array[Byte] = null
  private[this] val ctx = new ExecuteContext("/tmp", "file:///tmp", null, null, Region(), new ExecutionTimer())

  val codecs = new ShuffleCodecSpec(ctx, keyFields, rowType, rowEType, keyEType)

  private[this] val s = ShuffleClient.socket(host, port)
  private[this] val in = shuffleBufferSpec.buildInputBuffer(s.getInputStream())
  private[this] val out = shuffleBufferSpec.buildOutputBuffer(s.getOutputStream())

  private[this] def startOperation(op: Byte): Unit = {
    out.writeByte(op)
    if (op != Wire.START) {
      assert(uuid != null)
      log.info(s"CLNT operation $op uuid ${uuidToString(uuid)}")
      Wire.writeByteArray(out, uuid)
    }
  }

  def start(): Unit = {
    log.info(s"CLNT start")
    startOperation(Wire.START)
    Wire.writeTStruct(out, rowType)
    Wire.writeEBaseStruct(out, rowEType)
    Wire.writeSortFieldArray(out, keyFields)
    Wire.writeEBaseStruct(out, keyEType)
    out.flush()
    uuid = Wire.readByteArray(in)
    assert(uuid.length == Wire.ID_SIZE, s"${uuid.length} ${Wire.ID_SIZE}")
    log.info(s"CLNT start done")
  }

  def put(values: Array[Long]): Unit = {
    log.info(s"CLNT put")
    startOperation(Wire.PUT)
    out.flush()
    val encoder = codecs.makeRowEncoder(out)
    writeRegionValueArray(encoder, values)
    // fixme: server needs to send uuid for the successful partition
    out.flush()
    assert(in.readByte() == 0.toByte)
    log.info(s"CLNT put done")
  }

  def get(
    region: Region,
    start: Long,
    startInclusive: Boolean,
    end: Long,
    endInclusive: Boolean
  ): Array[Long] = {
    log.info(s"CLNT get ${Region.pretty(codecs.keyDecodedPType, start)} ${startInclusive} " +
      s"${Region.pretty(codecs.keyDecodedPType, end)} ${endInclusive}")
    val keyEncoder = codecs.makeKeyEncoder(out)
    val decoder = codecs.makeRowDecoder(in)
    startOperation(Wire.GET)
    out.flush()
    keyEncoder.writeRegionValue(start)
    keyEncoder.writeByte(if (startInclusive) 1.toByte else 0.toByte)
    keyEncoder.writeRegionValue(end)
    keyEncoder.writeByte(if (endInclusive) 1.toByte else 0.toByte)
    keyEncoder.flush()
    log.info(s"CLNT get receiving values")
    val values = readRegionValueArray(region, decoder)
    log.info(s"CLNT get done")
    values
  }

  def partitionBounds(region: Region, nPartitions: Int): Array[Long] = {
    log.info(s"CLNT partitionBounds")
    val keyDecoder = codecs.makeKeyDecoder(in)
    startOperation(Wire.PARTITION_BOUNDS)
    out.writeLong(nPartitions)
    out.flush()
    log.info(s"CLNT partitionBounds receiving values")
    val keys = readRegionValueArray(region, keyDecoder, nPartitions + 1)
    log.info(s"CLNT partitionBounds done")
    keys
  }

  def stop(): Unit = {
    log.info(s"CLNT stop")
    out.writeByte(Wire.STOP)
    Wire.writeByteArray(out, uuid)
    out.flush()
    assert(in.readByte() == 0.toByte)
    log.info(s"CLNT stop done")
  }

  def close(): Unit = {
    try {
      out.writeByte(Wire.EOS)
      out.flush()
      assert(in.readByte() == Wire.EOS)
    } finally {
      try {
        s.close()
      } finally {
        ctx.close()
      }
    }
  }
}
