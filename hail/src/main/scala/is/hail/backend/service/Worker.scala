package is.hail.backend.service

import java.io._
import java.nio.charset._

import is.hail.{HAIL_REVISION, HailContext}
import is.hail.backend.HailTaskContext
import is.hail.io.fs._
import is.hail.services._
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.concurrent.duration.{Duration, MILLISECONDS}
import scala.concurrent.{Future, Await}

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
  def start(label: String): Unit = {
    startTimes.put(label, System.nanoTime())
  }

  def end(label: String): Unit = {
    val endTime = System.nanoTime()
    val startTime = startTimes.get(label)
    startTime.foreach { s =>
      val durationMS = "%.6f".format((endTime - s).toDouble / 1000000.0)
      log.info(s"$label took $durationMS ms.")
    }
  }
}

object Worker {
  private[this] val log = Logger.getLogger(getClass.getName())
  private[this] val myRevision = HAIL_REVISION
  private[this] val scratchDir = sys.env.get("HAIL_WORKER_SCRATCH_DIR").getOrElse("")

  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      throw new IllegalArgumentException(s"expected at least four arguments, not: ${ args.length }")
    }
    val root = args(2)
    val i = args(3).toInt
    val timer = new WorkerTimer()

    log.info(s"is.hail.backend.service.Worker $myRevision")
    log.info(s"running job $i at root $root with scratch directory '$scratchDir'")

    timer.start(s"Job $i")

    timer.start("readInputs")
    val fs = retryTransientErrors {
      using(new FileInputStream(s"$scratchDir/gsa-key/key.json")) { is =>
        new GoogleStorageFS(IOUtils.toString(is, Charset.defaultCharset().toString())).asCacheable()
      }
    }

    val fileRetrievalExecutionContext = scala.concurrent.ExecutionContext.global
    val fFuture = Future {
      retryTransientErrors {
        using(new ObjectInputStream(fs.openCachedNoCompression(s"$root/f"))) { is =>
          is.readObject().asInstanceOf[(Array[Byte], HailTaskContext, FS) => Array[Byte]]
        }
      }
    }(fileRetrievalExecutionContext)

    val contextFuture = Future {
      retryTransientErrors {
        using(fs.openCachedNoCompression(s"$root/contexts")) { is =>
          is.seek(i * 12)
          val offset = is.readLong()
          val length = is.readInt()
          is.seek(offset)
          val context = new Array[Byte](length)
          is.readFully(context)
          context
        }
      }
    }(fileRetrievalExecutionContext)

    // retryTransientErrors handles timeout and exception throwing logic
    val f = Await.result(fFuture, Duration.Inf)
    val context = Await.result(contextFuture, Duration.Inf)

    timer.end("readInputs")
    timer.start("executeFunction")

    val hailContext = HailContext(
      // FIXME: workers should not have backends, but some things do need hail contexts
      new ServiceBackend(null), skipLoggingConfiguration = true, quiet = true)
    val htc = new ServiceTaskContext(i)
    val result = f(context, htc, fs)
    htc.finish()

    timer.end("executeFunction")
    timer.start("writeOutputs")

    using(fs.createCachedNoCompression(s"$root/result.$i")) { os =>
      os.write(result)
    }
    timer.end("writeOutputs")
    timer.end(s"Job $i")
    log.info(s"finished job $i at root $root")
  }
}
