package is.hail.scheduler

import java.io._
import java.net.Socket
import java.util.concurrent.Executors
import scala.collection.mutable

import is.hail.utils._

class RemoteException(cause: Throwable) extends Exception(cause)

object ExecutorMessage {
  // Scheduler => Executor
  val EXECUTE = 2

  // Executor => Scheduler
  val PING = 1
  val TASKRESULT = 4
}

class TaskThread[T](ex: Executor, taskId: Int, f: () => T) extends Runnable {
  def run(): Unit = {
    val v = try {
      f()
    } catch {
      case e: Exception =>
        val re = new RemoteException(e)
        ex.sendTaskResult(taskId, re)
        return
    }
    ex.sendTaskResult(taskId, v)
  }
}

class PingThread(ex: Executor) extends Runnable {
  def run(): Unit = {
    while (true) {
      try {
        ex.sendPing()
      } catch {
        case e: Exception =>
          log.info(s"PingThread.run: sendPing failed due to exception $e")
      }
      Thread.sleep(5000)
    }
  }
}

class TaskResult[T](val taskId: Int, val result: T)


class DataSocket(val socket: Socket) {
  val in = new DataInputStream(new BufferedInputStream(socket.getInputStream))
  val out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream))
}

class Executor(host: String, nCores: Int) extends Runnable {
  private var pool = Executors.newFixedThreadPool(nCores)

  @volatile private var socket: DataSocket = _
  private val outLock = new Object

  private var pendingResults = new mutable.ArrayBuffer[TaskResult[_]]()

  def sendPing(): Unit = outLock.synchronized {
    val s = socket
    if (s != null) {
      try {
        s.out.writeInt(ExecutorMessage.PING)
        s.out.flush()
      } catch {
        case e: Exception =>
          log.error(s"Client.sendPing: failed due to exception $e")
          closeConnection()
          throw e
      }
    }
  }

  def sendTaskResult[T](taskId: Int, result: T): Unit = outLock.synchronized {
    val s = socket
    if (s != null) {
      try {
        s.out.writeInt(ExecutorMessage.TASKRESULT)
        s.out.writeInt(taskId)
        writeObject(result, s.out)
        s.out.flush()
        log.info(s"sent task $taskId result")
      } catch {
        case e: Exception =>
          log.error(s"Client.sendTaskResult: queuing result, send failed due to exception: $e")
          pendingResults += new TaskResult(taskId, result)
          closeConnection()
      }
    } else {
      pendingResults += new TaskResult(taskId, result)
    }
  }

  def handleExecute[T](): Unit = {
    val s = socket
    val taskId = s.in.readInt()
    val f = readObject[() => _](s.in)
    log.info(s"received task $taskId")
    pool.execute(new TaskThread(this, taskId, f))
  }

  def run1(): Unit = {
    try {
      val s = new DataSocket(new Socket(host, 5051))

      s.out.writeInt(nCores)
      s.out.flush()

      socket = s
      log.info(s"Client.run1: connected to $host:5051")

      while (pendingResults.nonEmpty) {
        val tr = pendingResults.last
        sendTaskResult(tr.taskId, tr.result)
        pendingResults.reduceToSize(pendingResults.size - 1)
      }

      while (true) {
        val msg = s.in.readInt()
        msg match {
          case ExecutorMessage.EXECUTE =>
            handleExecute()
        }
      }
    } finally {
      closeConnection()
    }
  }

  def run(): Unit = {
    retry(run1)
  }


  def closeConnection(): Unit = {
    val s = socket
    socket = null
    if (s != null)
      s.socket.close()
  }
}

object Executor {
  def main(args: Array[String]): Unit = {
    val host = args(0)
    val nCores = args(1).toInt

    val ex = new Executor(host, nCores)
    val pt = new Thread(new PingThread(ex))
    pt.start()

    ex.run()
  }
}
