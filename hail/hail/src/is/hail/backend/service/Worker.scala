package is.hail.backend.service

import is.hail.{HAIL_REVISION, HailFeatureFlags}
import is.hail.asm4s._
import is.hail.backend.Backend.PartitionFn
import is.hail.backend.HailTaskContext
import is.hail.io.fs._
import is.hail.services._
import is.hail.utils._

import scala.collection.mutable
import scala.concurrent.{Await, ExecutionContext, ExecutionContextExecutorService, Future}
import scala.concurrent.duration.Duration

import java.io._
import java.nio.charset._
import java.nio.file.Path
import java.util
import java.util.{concurrent => javaConcurrent}

import org.apache.log4j.Logger

class ServiceTaskContext(val partitionId: Int) extends HailTaskContext {
  override def stageId(): Int = 0

  override def attemptNumber(): Int = 0
}

object WorkerTimer {
  private val log = Logger.getLogger(getClass.getName())
}

class WorkerTimer() {
  import WorkerTimer._

  var startTimes: mutable.Map[String, Long] = mutable.Map()

  def start(label: String): Unit =
    startTimes.update(label, System.nanoTime())

  def end(label: String): Unit = {
    val endTime = System.nanoTime()
    val startTime = startTimes.get(label)
    startTime.foreach { s =>
      val durationMS = "%.6f".format((endTime - s).toDouble / 1000000.0)
      log.info(s"$label took $durationMS ms.")
    }
  }
}

// Java's ObjectInputStream does not properly use the context classloader that
// we set on the thread and knows about the hail jar. We need to explicitly force
// the ObjectInputStream to use the correct classloader, but it is not configurable
// so we override the behavior ourselves.
// For more context, see: https://github.com/scala/bug/issues/9237#issuecomment-292436652
object ExplicitClassLoaderInputStream {
  val primClasses: util.HashMap[String, Class[_]] = {
    val m = new util.HashMap[String, Class[_]](8, 1.0f)
    m.put("boolean", Boolean.getClass)
    m.put("byte", Byte.getClass)
    m.put("char", Char.getClass)
    m.put("short", Short.getClass)
    m.put("int", Int.getClass)
    m.put("long", Long.getClass)
    m.put("float", Float.getClass)
    m.put("double", Double.getClass)
    // FIXME: I (ps) don't understand what this code is doing. In Scala 2.13 the Unit object isn't
    // allowed to be named anymore. Why are we loading scala companion objects here?
    m.put("void", ().getClass)
    m
  }
}

class ExplicitClassLoaderInputStream(is: InputStream, cl: ClassLoader)
    extends ObjectInputStream(is) {

  override def resolveClass(desc: ObjectStreamClass): Class[_] = {
    val name = desc.getName
    try return Class.forName(name, false, cl)
    catch {
      case ex: ClassNotFoundException =>
        val cl = ExplicitClassLoaderInputStream.primClasses.get(name)
        if (cl != null) return cl
        else throw ex
    }
  }
}

object Worker {
  private[this] val log = Logger.getLogger(getClass.getName())

  implicit private[this] val ec: ExecutionContextExecutorService =
    ExecutionContext.fromExecutorService(
      javaConcurrent.Executors.newCachedThreadPool()
    )

  private[this] def writeString(out: DataOutputStream, s: String): Unit = {
    val bytes = s.getBytes(StandardCharsets.UTF_8)
    out.writeInt(bytes.length)
    out.write(bytes)
  }

  def writeException(out: DataOutputStream, e: Throwable): Unit = {
    val (shortMessage, expandedMessage, errorId) = handleForPython(e)
    out.writeBoolean(false)
    writeString(out, shortMessage)
    writeString(out, expandedMessage)
    out.writeInt(errorId)
  }

  def main(argv: Array[String]): Unit = {
    val theHailClassLoader = new HailClassLoader(getClass().getClassLoader())

    if (argv.length != 7) {
      throw new IllegalArgumentException(s"expected seven arguments, not: ${argv.length}")
    }
    val scratchDir = argv(0)
    // val logFile = argv(1)
    // var jarLocation = argv(2)
    val kind = argv(3)
    assert(kind == Main.WORKER)
    val root = argv(4)
    val i = argv(5).toInt
    val n = argv(6).toInt
    val timer = new WorkerTimer()

    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    log.info(s"is.hail.backend.service.Worker $HAIL_REVISION")
    log.info(s"running job $i/$n at root $root with scratch directory '$scratchDir'")

    timer.start(s"Job $i/$n")

    timer.start("readInputs")
    val fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        HailFeatureFlags.fromEnv(),
      )
    )

    def open(x: String): SeekableDataInputStream =
      fs.openNoCompression(x)

    def write(x: String)(writer: PositionedDataOutputStream => Unit): Unit =
      fs.writePDOS(x)(writer)

    val gFuture = Future {
      retryTransientErrors {
        using(open(s"$root/globals"))(_.readAllBytes())
      }
    }

    val fFuture = Future {
      retryTransientErrors {
        using(new ExplicitClassLoaderInputStream(open(s"$root/f"), theHailClassLoader)) {
          _.readObject().asInstanceOf[PartitionFn]
        }
      }
    }

    val contextFuture = Future {
      retryTransientErrors {
        using(open(s"$root/contexts")) { is =>
          is.seek(i.toLong * 12)
          val offset = is.readLong()
          val length = is.readInt()
          is.seek(offset)
          val context = new Array[Byte](length)
          is.readFully(context)
          context
        }
      }
    }

    val globals = Await.result(gFuture, Duration.Inf)
    val f = Await.result(fFuture, Duration.Inf)
    val context = Await.result(contextFuture, Duration.Inf)

    timer.end("readInputs")
    timer.start("executeFunction")

    val result: Either[Throwable, Array[Byte]] =
      try
        using(new ServiceTaskContext(i)) { htc =>
          retryTransientErrors {
            Right(f(globals, context, htc, theHailClassLoader, fs))
          }
        }
      catch {
        case t: Throwable => Left(t)
      }

    timer.end("executeFunction")
    timer.start("writeOutputs")

    retryTransientErrors {
      write(s"$root/result.$i") { dos =>
        result match {
          case Right(bytes) =>
            dos.writeBoolean(true)
            dos.write(bytes)
          case Left(throwableWhileExecutingUserCode) =>
            writeException(dos, throwableWhileExecutingUserCode)
        }
      }
    }

    timer.end("writeOutputs")
    timer.end(s"Job $i")
    log.info(s"finished job $i at root $root")

    result.left.foreach { throwableWhileExecutingUserCode =>
      log.info("throwing the exception so that this Worker job is marked as failed.")
      throw throwableWhileExecutingUserCode
    }
  }
}
