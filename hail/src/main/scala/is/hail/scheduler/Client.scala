package is.hail.scheduler

import java.net.Socket
import java.util.concurrent.LinkedBlockingQueue

import scala.util.Random
import scala.collection.mutable
import org.apache.spark.ExposedUtils
import is.hail.utils._

object ClientMessage {
  // Client => Scheduler
  val SUBMIT = 5

  // Scheduler => Client
  val APPTASKRESULT = 6
  val ACKTASKRESULT = 7
}

class ReceiverThread(host: String, token: Array[Byte], q: LinkedBlockingQueue[Int]) extends Runnable {
  private var socket: DataSocket = _

  private var nTasks: Int = 0
  private var receivedTasks: mutable.BitSet = _
  private var nComplete: Int = 0
  private var callback: (Int, Any) => Unit = _

  private def getConnection(): DataSocket = {
    var s = socket
    if (s == null) {
      s = new DataSocket(new Socket(host, 5053))
      println(s"ReceiverThread.getConnection: connected to $host:5053")
      writeByteArray(token, s.out)
      s.out.flush()
      socket = s
    }
    s
  }

  private def closeConnection(): Unit = {
    val s = socket
    socket = null
    if (s != null)
      s.socket.close()
  }

  private def withConnection(f: DataSocket => Unit): Unit = {
    try {
      val s = retry(getConnection)
      f(s)
    } catch {
      case e: Exception =>
        closeConnection()
        throw e
    }
  }

  def startJob(n: Int, cb: (Int, Any) => Unit): Unit = synchronized {
    println(s"in startJob $cb")
    assert(callback == null)
    nTasks = n
    callback = cb
    nComplete = 0
    receivedTasks = mutable.BitSet(nTasks)
  }

  def endJob(): Unit = synchronized {
    q.put(0)
    nTasks = 0
    receivedTasks = null
    nComplete = 0
    callback = null
  }

  private def sendAckTask(s: DataSocket, index: Int): Unit = {
    s.out.writeInt(ClientMessage.ACKTASKRESULT)
    s.out.writeInt(index)
    s.out.flush()
  }

  private def handleAppTaskResult(s: DataSocket): Unit = {
    println("in handleAppTaskResult")
    val index = s.in.readInt()
    val res = readObject[Any](s.in)
    if (callback == null)
      println(s"ignoring $index")
    if (callback != null && !receivedTasks.contains(index)) {
      callback(index, res)
      receivedTasks += index
      nComplete += 1
      println(s"ReceiverThread.handleAppTaskResult: $index done, $nComplete/$nTasks complete")
    }
    sendAckTask(s, index)
    if (nComplete == nTasks)
      endJob()
  }

  private def run1(): Unit = {
    withConnection { s =>
      while (true) {
        val msg = s.in.readInt()
        msg match {
          case ClientMessage.APPTASKRESULT =>
            handleAppTaskResult(s)
        }
      }
    }
  }

  def run(): Unit = retry(run1)
}

class SchedulerAppClient(host: String) {
  private val token = new Array[Byte](16)
  Random.nextBytes(token)

  private val q = new LinkedBlockingQueue[Int]()

  private val receiver = new ReceiverThread(host, token, q)
  private val rt = new Thread(receiver)
  rt.start()

  private var socket: DataSocket = _

  private def getConnection(): DataSocket = {
    var s = socket
    if (s == null) {
      s = new DataSocket(new Socket(host, 5052))
      println(s"SchedulerAppClient.getConnection: connected to $host:5052")
      writeByteArray(token, s.out)
      s.out.flush()
      socket = s
    }
    s
  }

  private def closeConnection(): Unit = {
    val s = socket
    socket = null
    if (s != null)
      s.socket.close()
  }

  private def withConnection(f: DataSocket => Unit): Unit = {
    try {
      val s = retry(getConnection)
      f(s)
    } catch {
      case e: Exception =>
        closeConnection()
        throw e
    }
  }

  private def submit1[C, T](da: DArray[T], cb: (Int, T) => Unit): Unit = {
    withConnection { s =>
      s.out.writeInt(ClientMessage.SUBMIT)
      val n = da.nTasks
      s.out.writeInt(n)
      s.out.flush()

      var i = s.in.readInt()
      while (i < n) {
        val context = da.contexts(i)
        val localBody = da.body
        val f: () => T = () => localBody(context)
        ExposedUtils.clean(f, checkSerializable = true)
        writeObject(f, s.out)
        i += 1
      }
      s.out.flush()

      val ack = s.in.readInt()
      assert(ack == 0)
    }
  }

  def submit[C, T](da: DArray[T], cb: (Int, T) => Unit): Unit = synchronized {
    receiver.startJob(da.nTasks, (i: Int, a: Any) => cb(i, a.asInstanceOf[T]))

    retry(() => submit1(da, cb))

    val t = q.take()
    assert(t == 0)
  }
}
