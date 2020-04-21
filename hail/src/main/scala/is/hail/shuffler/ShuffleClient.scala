package is.hail.shuffler

import java.io._

import is.hail.annotations._
import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.virtual.TStruct
import is.hail.io.TypedCodecSpec
import is.hail.utils._
import javax.net.ssl._
import org.apache.log4j.Logger

class ShuffleClient (
  t: TStruct,
  codecSpec: TypedCodecSpec,
  key: Array[String],
  ssl: SSLContext,
  host: String,
  port: Int
) {
  // FIXME close
  private[this] val ctx = new ExecuteContext("/tmp", "file:///tmp", null, null, Region(), new ExecutionTimer())

  val log = Logger.getLogger(getClass.getName)

  val sf = ssl.getSocketFactory
  val s = sf.createSocket(host, port)
  val in = new DataInputStream(s.getInputStream)
  val out = new DataOutputStream(s.getOutputStream)
  log.info(s"CLNT connected to ${host}:${port}")

  val keyedCodecSpec = new KeyedCodecSpec(ctx, t, codecSpec, key)
  import keyedCodecSpec._

  var uuid: Array[Byte] = null

  def start(): Unit = {
    log.info(s"CLNT start")
    out.write(Wire.START)
    Wire.writeTStruct(out, t)
    Wire.writeTypedCodecSpec(out, codecSpec)
    Wire.writeStringArray(out, key)
    out.flush()
    uuid = Wire.readByteArray(in)
    log.info(s"CLNT start done")
  }

  def put(values: Iterator[Long]): Unit = {
    log.info(s"CLNT put")
    val encoder = makeEnc(out)
    encoder.writeByte(Wire.PUT)
    encoder.flush()
    Wire.writeByteArray(out, uuid)
    out.flush()
    while (values.hasNext) {
      encoder.writeByte(1)
      val off = values.next
      encoder.writeRegionValue(off)
    }
    encoder.writeByte(0)
    encoder.flush()
    assert(in.read() == 0)
    log.info(s"CLNT put done")
  }

  def get(region: Region, start: Long, end: Long): Array[Long] = {
    log.info(s"CLNT get")
    val keyEncoder = makeKeyEnc(out)
    val decoder = makeDec(in)
    keyEncoder.writeByte(Wire.GET)
    keyEncoder.flush()
    Wire.writeByteArray(out, uuid)
    out.flush()
    keyEncoder.writeRegionValue(start)
    keyEncoder.writeRegionValue(end)
    keyEncoder.flush()

    log.info(s"CLNT get receiving values")
    val ab = new ArrayBuilder[Long]()
    var hasNext = decoder.readByte()
    assert(hasNext >= 0)
    while (hasNext == 1) {
      ab += decoder.readRegionValue(region)
      hasNext = decoder.readByte()
      assert(hasNext >= 0)
    }
    assert(hasNext == 0)
    log.info(s"CLNT get done")
    ab.result()
  }

  def stop(): Unit = {
    log.info(s"CLNT stop")
    out.write(Wire.STOP)
    Wire.writeByteArray(out, uuid)
    out.flush()
    log.info(s"CLNT stop done")
  }
}
