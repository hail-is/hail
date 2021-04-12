package is.hail.backend.service

import java.io._
import java.nio.charset.Charset

import is.hail.HailContext
import is.hail.backend._
import is.hail.io.fs._
import is.hail.services._
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger

import scala.collection.mutable

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
  private val log = Logger.getLogger(getClass.getName())

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      throw new IllegalArgumentException(s"expected two arguments, not: ${ args.length }")
    }
    val root = args(0)
    val i = args(1).toInt
    val timer = new WorkerTimer()

    var scratchDir = System.getenv("HAIL_WORKER_SCRATCH_DIR")
    if (scratchDir == null)
      scratchDir = ""

    log.info(s"running job $i at root $root wih scratch directory '$scratchDir'")

    timer.start(s"Job $i")
    timer.start("readInputs")

    val fs = retryTransientErrors {
      using(new FileInputStream(s"$scratchDir/gsa-key/key.json")) { is =>
        new GoogleStorageFS(IOUtils.toString(is, Charset.defaultCharset().toString()))
      }
    }

    val f = retryTransientErrors {
      using(new ObjectInputStream(fs.openNoCompression(s"$root/f"))) { is =>
        is.readObject().asInstanceOf[(Array[Byte], HailTaskContext) => Array[Byte]]
      }
    }

    var offset = 0L
    var length = 0

    retryTransientErrors {
      using(fs.openNoCompression(s"$root/context.offsets")) { is =>
        is.seek(i * 12)
        offset = is.readLong()
        length = is.readInt()
      }
    }

    val context = retryTransientErrors {
      using(fs.openNoCompression(s"$root/contexts")) { is =>
        is.seek(offset)
        val context = new Array[Byte](length)
        is.readFully(context)
        context
      }
    }
    timer.end("readInputs")
    timer.start("executeFunction")

    val hailContext = HailContext(
      ServiceBackend(), skipLoggingConfiguration = true, quiet = true)
    val htc = new ServiceTaskContext(i)
    HailTaskContext.setTaskContext(htc)
    val result = f(context, htc)
    HailTaskContext.finish()

    timer.end("executeFunction")
    timer.start("writeOutputs")

    using(fs.createNoCompression(s"$root/result.$i")) { os =>
      os.write(result)
    }
    timer.end("writeOutputs")
    timer.end(s"Job $i")
    log.info(s"finished job $i at root $root")
  }
}
