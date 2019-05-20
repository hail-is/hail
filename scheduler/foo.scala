package org.apache.spark.foo

import java.io._
import java.net.Socket
import java.util.concurrent.{Executors, LinkedBlockingQueue}
import scala.util.Random
import scala.collection.mutable
import scala.reflect.ClassTag
import org.apache.spark.util.ClosureCleaner

class RemoteException(cause: Throwable) extends Exception(cause)

object Message {
  // Scheduler => Executor
  val EXECUTE = 2

  // Executor => Scheduler
  val PING = 1
  val TASKRESULT = 4

  // App => Scheduler
  val SUBMIT = 5

  // Scheduler => App
  val APPTASKRESULT = 6
  val ACKTASK = 7

  def writeByteArray(b: Array[Byte], out: DataOutputStream): Unit = {
    out.writeInt(b.length)
    out.write(b)
  }

  def writeObject[T](v: T, out: DataOutputStream): Unit = {
    val bos = new ByteArrayOutputStream()
    val boos = new ObjectOutputStream(bos)
    boos.writeObject(v)
    writeByteArray(bos.toByteArray, out)
  }

  def readByteArray(in: DataInputStream): Array[Byte] = {
    val n = in.readInt()
    val b = new Array[Byte](n)
    in.readFully(b)
    b
  }

  def readObject[T](in: DataInputStream): T = {
    val b = readByteArray(in)
    val bis = new ByteArrayInputStream(b)
    val bois = new ObjectInputStream(bis)
    bois.readObject().asInstanceOf[T]
  }

  def partition(n: Int, k: Int): Array[Int] = {
    if (k == 0) {
      assert(n == 0)
      return Array.empty[Int]
    }

    assert(n >= 0)
    assert(k > 0)
    val parts = Array.tabulate(k)(i => (n - i + k - 1) / k)
    assert(parts.sum == n)
    assert(parts.max - parts.min <= 1)
    parts
  }

  def time[A](f: => A): (A, Long) = {
    val t0 = System.nanoTime()
    val result = f
    val t1 = System.nanoTime()
    (result, t1 - t0)
  }

  final val msPerMinute = 60 * 1e3
  final val msPerHour = 60 * msPerMinute
  final val msPerDay = 24 * msPerHour

  def formatTime(dt: Long): String = {
    val tMilliseconds = dt / 1e6
    if (tMilliseconds < 1000)
      ("%.3f" + "ms").format(tMilliseconds)
    else if (tMilliseconds < msPerMinute)
      ("%.3f" + "s").format(tMilliseconds / 1e3)
    else if (tMilliseconds < msPerHour) {
      val tMins = (tMilliseconds / msPerMinute).toInt
      val tSec = (tMilliseconds % msPerMinute) / 1e3
      ("%d" + "m" + "%.1f" + "s").format(tMins, tSec)
    }
    else {
      val tHrs = (tMilliseconds / msPerHour).toInt
      val tMins = ((tMilliseconds % msPerHour) / msPerMinute).toInt
      val tSec = (tMilliseconds % msPerMinute) / 1e3
      ("%d" + "h" + "%d" + "m" + "%.1f" + "s").format(tHrs, tMins, tSec)
    }
  }

  def printTime[T](name: String)(block: => T) = {
    val timed = time(block)
    println(s"time: $name: ${ formatTime(timed._2) }")
    timed._1
  }

  def retry[T](f: () => T, exp: Double = 2.0, maxWait: Double = 60.0): T = {
    var minWait = 1.0
    var w = minWait
    while (true) {
      val startTime = System.nanoTime()
      try {
        return f()
      } catch {
        case e: Exception =>
          println(s"retry: restarting due to exception: $e")
          e.printStackTrace()
      }
      val endTime = System.nanoTime()
      val duration = (endTime - startTime) / 1e-9
      w = math.min(maxWait, math.max(minWait, w * exp - duration))
      val t = (1000 * w * Random.nextDouble).toLong
      println(s"retry: waiting ${ formatTime(t * 1000000) }")
      Thread.sleep(t)
    }
    null.asInstanceOf[T]
  }
}

class TaskThread[T](client: Client, taskId: Int, f: () => T) extends Runnable {
  def run(): Unit = {
    val v = try {
      f()
    } catch {
      case e: Exception =>
        val re = new RemoteException(e)
        client.sendTaskResult(taskId, re)
        return
    }
    client.sendTaskResult(taskId, v)
  }
}

class PingThread(client: Client) extends Runnable {
  def run(): Unit = {
    while (true) {
      try {
        client.sendPing()
      } catch {
        case e: Exception =>
        println(s"PingThread.run: sendPing failed due to exception $e")
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

class Client(host: String, nCores: Int) extends Runnable {
  private var pool = Executors.newFixedThreadPool(nCores)

  @volatile private var socket: DataSocket = null
  private val outLock = new Object

  private var pendingResults = new mutable.ArrayBuffer[TaskResult[_]]()

  def sendPing(): Unit = outLock.synchronized {
    val s = socket
    if (s != null) {
      try {
        s.out.writeInt(Message.PING)
        s.out.flush()
      } catch {
        case e: Exception =>
          println(s"Client.sendPing: failed due to exception $e")
          closeConnection()
          throw e
      }
    }
  }

  def sendTaskResult[T](taskId: Int, result: T): Unit = outLock.synchronized {
    val s = socket
    if (s != null) {
      try {
        s.out.writeInt(Message.TASKRESULT)
        s.out.writeInt(taskId)
        Message.writeObject(result, s.out)
        s.out.flush()
      } catch {
        case e: Exception =>
          println(s"Client.sendTaskResult: queuing result, send failed due to exception: $e")
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
    val f = Message.readObject[() => _](s.in)
    pool.execute(new TaskThread(this, taskId, f))
  }

  def closeConnection(): Unit = {
    // race condition, worth locking?
    val s = socket
    socket = null
    if (s != null)
      s.socket.close()
  }

  def run1(): Unit = {
    try {
      val s = new DataSocket(new Socket(host, 5051))

      s.out.writeInt(nCores)
      s.out.flush()

      socket = s
      println(s"Client.run1: connected to $host:5051")

      while (pendingResults.nonEmpty) {
        val tr = pendingResults.last
        sendTaskResult(tr.taskId, tr.result)
        pendingResults.reduceToSize(pendingResults.size - 1)
      }

      while (true) {
        val msg = s.in.readInt()
        msg match {
          case Message.EXECUTE =>
            handleExecute()
        }
      }
    } finally {
      closeConnection()
    }
  }

  def run(): Unit = {
    Message.retry(run1)
  }
}

object Client {
  def main(args: Array[String]): Unit = {
    val host = args(0)
    val nCores = args(1).toInt

    val client = new Client(host, nCores)
    val pt = new Thread(new PingThread(client))
    pt.start()

    client.run()
  }
}

abstract class DArray[T] {
  type Context

  val contexts: Array[Context]
  val body: (Context) => T

  def nTasks: Int = contexts.length
}

class DSeqContext(host: String) {
  val conn = new SchedulerAppClient(host)

  def parallelize[T](a: IndexedSeq[T], p: Int)(implicit tct: ClassTag[T]): DSeq[T] = {
    val counts = Message.partition(a.length, p)
    val starts = counts.scanLeft(0)(_ + _)
    new DSeq(this, new DArray[Seq[T]] {
      type Context = Seq[T]
      val contexts = Array.tabulate[Seq[T]](p)(i => a.slice(starts(i), starts(i + 1)))
      val body = (a: Seq[T]) => a
    })
  }

  def range(n: Int, p: Int): DSeq[Int] = {
    val counts = Message.partition(n, p)
    val starts = counts.scanLeft(0)(_ + _)
    new DSeq(this, new DArray[Seq[Int]] {
      type Context = Range
      val contexts = Array.tabulate(p)(i => starts(i) until starts(i + 1))
      val body = (r: Range) => r
    })
  }
}

class DSeq[T](conn: DSeqContext, da: DArray[Seq[T]]) {
  def map[U](f: (T) => U): DSeq[U] = {
    new DSeq(conn, new DArray[Seq[U]] {
      type Context = da.Context
      val contexts = da.contexts
      val body = (c: da.Context) => da.body(c).map(f)
    })
  }

  // filter, flatMap, collect, (sum)
}

class ReceiverThread(host: String, token: Array[Byte], q: LinkedBlockingQueue[Int]) extends Runnable {
  println("starting receiver")

  private var socket: DataSocket = null

  private var nTasks: Int = 0
  private var receivedTasks: mutable.BitSet = null
  private var nComplete: Int = 0
  private var callback: (Int, Any) => Unit = null

  private def getConnection(): DataSocket = {
    var s = socket
    if (s == null) {
      s = new DataSocket(new Socket(host, 5053))
      println(s"ReceiverThread.getConnection: connected to $host:5053")
      Message.writeByteArray(token, s.out)
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

  private def withConnection(f: (DataSocket) => Unit): Unit = {
    try {
      val s = Message.retry(getConnection)
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
    s.out.writeInt(Message.ACKTASK)
    s.out.writeInt(index)
    s.out.flush()
  }

  private def handleAppTaskResult(s: DataSocket): Unit = {
    println("in handleAppTaskResult")
    val index = s.in.readInt()
    val res = Message.readObject[Any](s.in)
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
          case Message.APPTASKRESULT =>
            handleAppTaskResult(s)
        }
      }
    }
  }

  def run(): Unit = Message.retry(run1)
}

class SchedulerAppClient(host: String) {
  private val token = new Array[Byte](16)
  Random.nextBytes(token)

  private val q = new LinkedBlockingQueue[Int]()

  private val receiver = new ReceiverThread(host, token, q)
  private val rt = new Thread(receiver)
  rt.start()

  private var socket: DataSocket = null

  private def getConnection(): DataSocket = {
    var s = socket
    if (s == null) {
      s = new DataSocket(new Socket(host, 5052))
      println(s"SchedulerAppClient.getConnection: connected to $host:5052")
      Message.writeByteArray(token, s.out)
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

  private def withConnection(f: (DataSocket) => Unit): Unit = {
    try {
      val s = Message.retry(getConnection)
      f(s)
    } catch {
      case e: Exception =>
        closeConnection()
        throw e
    }
  }

  private def submit1[C, T](da: DArray[T], cb: (Int, T) => Unit): Unit = {
    withConnection { s =>
      s.out.writeInt(Message.SUBMIT)
      val n = da.nTasks
      s.out.writeInt(n)
      s.out.flush()

      var i = s.in.readInt()
      while (i < n) {
        val context = da.contexts(i)
        val localBody = da.body
        val f: () => T = () => localBody(context)
        // FIXME clean
        ClosureCleaner.clean(f, checkSerializable = true)
        Message.writeObject(f, s.out)
        i += 1
      }
      s.out.flush()

      val ack = s.in.readInt()
      assert(ack == 0)
    }
  }

  def submit[C, T](da: DArray[T], cb: (Int, T) => Unit): Unit = synchronized {
    receiver.startJob(da.nTasks, (i: Int, a: Any) => cb(i, a.asInstanceOf[T]))

    Message.retry(() => submit1(da, cb))

    val t = q.take()
    assert(t == 0)
  }
}

object Submit {
  def main(args: Array[String]): Unit = {
    val host = args(0)
    val n = args(1).toInt
    println(s"n $n")

    val conn = new SchedulerAppClient(host)

    val da = new DArray[Int] {
      type Context = Range
      val contexts = Array.tabulate(n)(i => i until i + 1)
      val body = (r: Range) => r.sum
    }

    var t = 0
    val callback = (i: Int, s: Int) => {
      t += s
    }

    Message.printTime("submit") {
      conn.submit(da, callback)
    }
    println(s"t = $t")

    t = 0
    Message.printTime("submit2") {
      conn.submit(da, callback)
    }
    println(s"t = $t")

    // because client connection is still running
    System.exit(0)
  }
}
