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

class SubmitterThread(host: String, token: Token, q: LinkedBlockingQueue[Option[(Token, DArray[_])]]) extends Runnable {
  private var socket: DataSocket = _

  private def connect(): DataSocket = {
    var s = socket
    if (s == null) {
      s = new DataSocket(new Socket(host, 5052))
      info(s"SubmitterThread.getConnection: connected to $host:5052")
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
      val s = retry(connect)
      f(s)
    } catch {
      case e: Exception =>
        closeConnection()
        throw e
    }
  }

  def submit(taskID: Token, da: DArray[_]): Unit = {
    withConnection { s =>
      log.info(s"SubmitterThread.submit: submitting task with ${ da.nTasks } tasks")
      s.out.writeInt(ClientMessage.SUBMIT)
      writeByteArray(taskID, s.out)
      val n = da.nTasks
      s.out.writeInt(n)
      s.out.flush()

      var i = s.in.readInt()
      while (i < n) {
        val context = da.contexts(i)
        val localBody = da.body
        val f = () => localBody(context)
        ExposedUtils.clean(f, checkSerializable = true)
        writeObject(f, s.out)
        i += 1
      }
      s.out.flush()

      log.info(s"SubmitterThread.submit: finished submitting task")
      val ack = s.in.readInt()
      assert(ack == 0)
      val returnToken = readByteArray(s.in)
      assert(returnToken sameElements taskID)
      log.info(s"SubmitterThread.submit: task was acknowledged by the scheduler.")
    }
  }

  def run(): Unit = {
    while (true) {
      q.take() match {
        case Some((taskID, da)) =>
          retry(() => submit(taskID, da))
        case None =>
          return
      }
    }
  }
}

class SchedulerAppClient(host: String) {
  private val token = newToken()
  private val q = new LinkedBlockingQueue[Option[(Token, DArray[_])]]()

  private val submitter = new SubmitterThread(host, token, q)
  private val st = new Thread(submitter)
  st.start()

  private var socket: DataSocket = _

  var nTasks: Int = 0
  private var receivedTasks: mutable.BitSet = _
  var nComplete: Int = 0
  var callback: (Int, Any)  => Unit = _
  var callbackExc: Exception = _

  private def newToken(): Token = {
    val t = new Array[Byte](16)
    Random.nextBytes(t)
    t
  }

  private def connect(): DataSocket = {
    var s = socket
    if (s == null) {
      s = new DataSocket(new Socket(host, 5053))
      info(s"SchedulerAppClient.getConnection: connected to $host:5053")
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
      val s = retry(connect)
      f(s)
    } catch {
      case e: Exception =>
        closeConnection()
        throw e
    }
  }

  private def sendAckTask(s: DataSocket, index: Int, taskID: Token): Unit = {
    s.out.writeInt(ClientMessage.ACKTASKRESULT)
    writeByteArray(taskID, s.out)
    s.out.writeInt(index)
    s.out.flush()
  }

  private def clear(): Unit = {
    callback = null
    receivedTasks = null
    callbackExc = null
  }

  private def receive(taskID: Token): Unit = {
    withConnection { s =>
      while (nComplete < nTasks) {
        val msg = s.in.readInt()
        assert(msg == ClientMessage.APPTASKRESULT)
        val taskToken = readByteArray(s.in)
        assert(taskID sameElements taskToken)

        val index = s.in.readInt()
        val res = readObject[Any](s.in)
        log.info(s"SchedulerAppClient.receive: received task result for index $index / $nTasks.")

        res match {
          case re: RemoteException =>
            // we are done here
            clear()
            throw new BreakRetryException(re.getCause)
          case _ =>
            if (!receivedTasks.contains(index)) {
              try {
                callback(index, res)
              } catch {
                case e: Exception =>
                  log.info(s"SchedulerAppClient.receive: got callback exception for task $index / $nTasks:\n$e")
                  callbackExc = e
              }
              receivedTasks += index
              nComplete += 1
            }
            log.info(s"SchedulerAppClient.receive: tasks completed: $nComplete / $nTasks.")
            sendAckTask(s, index, taskID)
        }
      }
    }
  }

  def submit[T](da: DArray[T], cb: (Int, T) => Unit): Unit = synchronized {
    val taskID = newToken()
    log.info(s"SchedulerAppClient.submit: submitting task with ${ da.nTasks } tasks")
    q.put(Some(taskID -> da))

    assert(callback == null)
    nTasks = da.nTasks
    nComplete = 0
    callback = (i: Int, x: Any) => cb(i, x.asInstanceOf[T])
    receivedTasks = mutable.BitSet(nTasks)

    log.info(s"SchedulerAppClient.submit: finished submitting task with ${ da.nTasks } tasks")
    retry(() => receive(taskID))
    val ce = callbackExc
    clear()

    if (ce != null)
      throw ce
  }

  def close(): Unit = {
    q.put(None)
  }
}
