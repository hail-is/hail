package is.hail.backend

import java.io.File
import java.net.Socket
import java.nio.charset.StandardCharsets

import is.hail.utils._
import is.hail.annotations.Memory
import is.hail.expr.ir.{BaseIR, BlockMatrixIR, ExecuteContext, IR, IRParser, IRParserEnvironment, MatrixIR, TableIR}
import is.hail.types.virtual.Type
import is.hail.utils.toRichInputStream
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods
import org.newsclub.net.unix.{AFUNIXSocket, AFUNIXSocketAddress}

import scala.annotation.switch
import scala.collection.mutable
import scala.collection.JavaConverters._

class EndOfInputException extends RuntimeException

class Py4JSocketThread(backend: Py4JBackend, socket: Socket) extends Thread {
  private[this] val PARSE_VALUE_IR = 1
  private[this] val VALUE_TYPE = 2
  private[this] val EXECUTE = 3
  private[this] val REMOVE_IR = 4

  private[this] val in = socket.getInputStream
  private[this] val out = socket.getOutputStream

  private[this] val dummy = new Array[Byte](8)

  def read(bytes: Array[Byte], off: Int, n: Int): Unit = {
    assert(off + n <= bytes.length)
    var read = 0
    while (read < n) {
      val r = in.read(bytes, off + read, n - read)
      if (r < 0) {
        if (socket.isClosed)
          throw new EndOfInputException
      } else {
        read += r
      }
    }
  }

  def readInt(): Int = {
    read(dummy, 0, 4)
    Memory.loadInt(dummy, 0)
  }

  def readLong(): Long = {
    read(dummy, 0, 8)
    Memory.loadLong(dummy, 0)
  }

  def readBytes(): Array[Byte] = {
    val n = readInt()
    val bytes = new Array[Byte](n)
    read(bytes, 0, n)
    bytes
  }

  def readString(): String = new String(readBytes(), StandardCharsets.UTF_8)

  def writeBool(b: Boolean): Unit = {
    out.write(if (b) 1 else 0)
  }

  def writeInt(v: Int): Unit = {
    Memory.storeInt(dummy, 0, v)
    out.write(dummy, 0, 4)
  }

  def writeLong(v: Long): Unit = {
    Memory.storeLong(dummy, 0, v)
    out.write(dummy)
  }

  def writeBytes(bytes: Array[Byte]): Unit = {
    writeInt(bytes.length)
    out.write(bytes)
  }

  def writeString(s: String): Unit = writeBytes(s.getBytes(StandardCharsets.UTF_8))

  def trySaveException[T](block: => T): Option[T] = {
    try {
      val v: T = block
      Some(v)
    } catch {
      case t: Throwable =>
        backend.setSavedException(t)
        None
    }
  }

  def eventLoop(): Unit = {
    while (true) {
      val cmd = readInt()
      (cmd: @switch) match {
        case PARSE_VALUE_IR =>
          val irStr = readString()
          val typeEnvStr = readString()
          val idOpt = trySaveException {
            implicit val formats: Formats = DefaultFormats
            val typeEnv = JsonMethods.parse(typeEnvStr).extract[Map[String, String]].mapValues(IRParser.parseType)
            backend.parseValueIR(irStr, typeEnv)
          }
          idOpt match {
            case Some(id) =>
              writeBool(true)
              writeLong(id)
            case None =>
              writeBool(false)
          }
        case VALUE_TYPE =>
          val id = readLong()
          val succeeded = trySaveException {
            backend.irMap(id).asInstanceOf[IR].typ.toString
          }
          succeeded match {
            case Some(typeStr) =>
              writeBool(true)
              writeString(typeStr)
            case None =>
              writeBool(false)
          }
        case EXECUTE =>
          val id = readLong()
          val strOpt = trySaveException {
            backend.executeJSON(id)
          }
          strOpt match {
            case Some(s) =>
              writeBool(true)
              writeString(s)
            case None =>
              writeBool(false)
          }
        case REMOVE_IR =>
          val id = readLong()
          val succeeded = trySaveException {
            backend.removeIR(id)
          }
          writeBool(succeeded.isDefined)
      }
    }
  }

  override def run(): Unit = {
    try {
      eventLoop()
    } catch {
      case t: Throwable =>
        log.info("py4 backend thread caught exception", t)
    } finally {
      socket.close()
    }
  }
}

abstract class Py4JBackend extends Backend {
  var irMap: mutable.Map[Long, BaseIR] = new mutable.HashMap[Long, BaseIR]()

  private[this] var irCounter: Long = 0

  def addIR(x: BaseIR): Long = {
    val id = irCounter
    irCounter += 1
    irMap(id) = x
    id
  }

  def removeIR(id: Long): Unit = {
    irMap -= id
  }

  def withExecuteContext[T]()(f: ExecuteContext => T): T

  def parseValueIR(s: String, typeEnv: Map[String, Type]): Long = {
    val ir = withExecuteContext() { ctx =>
      IRParser.parse_value_ir(s, IRParserEnvironment(ctx, typeEnv, irMap))
    }
    addIR(ir)
  }

  def pyParseTableIR(s: String, refMap: java.util.Map[String, String]): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_table_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyParseMatrixIR(s: String, refMap: java.util.Map[String, String]): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyParseBlockMatrixIR(
    s: String, refMap: java.util.Map[String, String]
  ): Long = {
    withExecuteContext() { ctx =>
      addIR(IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap)))
    }
  }

  def pyTableType(id: Long): String = {
    JsonMethods.compact(
      irMap(id).asInstanceOf[TableIR].typ.toJSON)
  }

  def pyMatrixType(id: Long): String = {
    JsonMethods.compact(
      irMap(id).asInstanceOf[MatrixIR].typ.toJSON)
  }

  def pyBlockMatrixType(id: Long): String = {
    JsonMethods.compact(
      irMap(id).asInstanceOf[BlockMatrixIR].typ.toJSON)
  }

  def executeJSON(id: Long): String

  override def stop(): Unit = {
    irMap = null
  }

  def pyBlockMatrixIsSparse(id: Int): Boolean = {
    irMap(id).asInstanceOf[BlockMatrixIR].typ.isSparse
  }

  def connectUNIXSocket(udsAddress: String): AFUNIXSocket = {
    val socket = AFUNIXSocket.newInstance()
    socket.connect(new AFUNIXSocketAddress(new File(udsAddress)))
    assert(!socket.isClosed)
    socket
  }

  def startUNIXSocketThread(socket: AFUNIXSocket): Unit = {
    new Py4JSocketThread(this, socket).start()
  }

  private[this] var savedException: Throwable = _

  def setSavedException(e: Throwable): Unit = synchronized {
    assert(savedException == null)
    savedException = e
  }

  def reraiseSavedException(): Unit = synchronized {
    assert(savedException != null)
    val e = savedException
    savedException = null
    throw e
  }
}
