package is.hail.shuffler

import org.apache.log4j.Logger;
import is.hail.shuffler.ShuffleUtils._
import is.hail.utils._
import is.hail.annotations._
import is.hail.expr.types.physical.{ PStruct, PType }
import is.hail.expr.types.virtual.TStruct
import is.hail.io.TypedCodecSpec
import java.io._
import java.net._
import java.security.KeyStore;
import java.util.UUID
import java.util.concurrent.{ ConcurrentSkipListMap, Executors }
import javax.net._
import javax.net.ssl._
import javax.security.cert.X509Certificate;
import org.json4s.jackson.{ JsonMethods, Serialization }

class ShuffleClient (
  t: TStruct,
  codecSpec: TypedCodecSpec,
  key: Array[String],
  ssl: SSLContext,
  host: String,
  port: Int
) {
  val log = Logger.getLogger(this.getClass.getName());

  val sf = ssl.getSocketFactory()
  val s = sf.createSocket(host, port)
  val in = new DataInputStream(s.getInputStream())
  val out = new DataOutputStream(s.getOutputStream())
  log.info(s"CLNT connected to ${host}:${port}")

  val keyedCodecSpec = new KeyedCodecSpec(t, codecSpec, key)
  import keyedCodecSpec._

  var uuid: String = null

  def start(): Unit = {
    log.info(s"CLNT start")
    out.write(Wire.START)
    Wire.writeTStruct(out, t)
    Wire.writeTypedCodecSpec(out, codecSpec)
    Wire.writeListOfStrings(out, key)
    out.flush()
    uuid = in.readUTF()
    log.info(s"CLNT start done")
  }

  def put(values: Iterator[Long]): Unit = {
    log.info(s"CLNT put")
    val encoder = makeEnc(out)
    encoder.writeByte(Wire.PUT)
    encoder.flush()
    out.writeUTF(uuid)
    out.flush()
    while (values.hasNext) {
      encoder.writeByte(1)
      val off = values.next
      encoder.writeRegionValue(null, off)
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
    out.writeUTF(uuid)
    out.flush()
    keyEncoder.writeRegionValue(null, start)
    keyEncoder.writeRegionValue(null, end)
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
}
