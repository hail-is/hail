package is.hail.utils

import java.io.{IOException, InterruptedIOException, ObjectOutputStream, OutputStream}
import java.net.{ConnectException, InetAddress, Socket}

import org.apache.log4j.{AppenderSkeleton, PatternLayout}
import org.apache.log4j.helpers.LogLog
import org.apache.log4j.spi.{ErrorCode, LoggingEvent}

/** This class was translated and streamlined from org.apache.log4j.net.SocketAppender */

object StringSocketAppender {
  // low reconnection delay because everything is local
  val DEFAULT_RECONNECTION_DELAY = 100

  var theAppender: StringSocketAppender = _

  def get(): StringSocketAppender = theAppender
}

class StringSocketAppender() extends AppenderSkeleton {
  private var address: InetAddress = _
  private var port: Int = _
  private var os: OutputStream = _
  private val reconnectionDelay = StringSocketAppender.DEFAULT_RECONNECTION_DELAY
  private var connector: SocketConnector = null
  private var patternLayout: PatternLayout = _
  private var initialized: Boolean = false

  StringSocketAppender.theAppender = this

  def connect(host: String, port: Int, format: String): Unit = {
    this.port = port
    this.address = InetAddress.getByName(host)
    this.patternLayout = new PatternLayout(format)
    connect(address, port)
    initialized = true
  }

  override def close(): Unit = {
    if (closed) return
    this.closed = true
    cleanUp()
  }

  private def cleanUp(): Unit = {
    if (os != null) {
      try
        os.close()
      catch {
        case e: IOException =>
          if (e.isInstanceOf[InterruptedIOException]) Thread.currentThread.interrupt()
          LogLog.error("Could not close os.", e)
      }
      os = null
    }
    if (connector != null) {
      connector.interrupted = true
      connector = null // allow gc
    }
  }

  private def connect(address: InetAddress, port: Int): Unit = {
    if (this.address == null) return
    try { // First, close the previous connection if any.
      cleanUp()
      os = new Socket(address, port).getOutputStream
    } catch {
      case e: IOException =>
        if (e.isInstanceOf[InterruptedIOException]) Thread.currentThread.interrupt()
        var msg = "Could not connect to remote log4j server at [" + address.getHostName + "]."
        if (reconnectionDelay > 0) {
          msg += " We will try again later."
          fireConnector() // fire the connector thread

        } else {
          msg += " We are not retrying."
          errorHandler.error(msg, e, ErrorCode.GENERIC_FAILURE)
        }
        LogLog.error(msg)
    }
  }

  override def append(event: LoggingEvent): Unit = {
    if (!initialized) return
    if (event == null) return
    if (address == null) {
      errorHandler.error("No remote host is set for SocketAppender named \"" + this.name + "\".")
      return
    }
    if (os != null)
      try {
        event.getLevel
        val str = patternLayout.format(event)
        os.write(str.getBytes("ISO-8859-1"))
        os.flush()
      } catch {
        case e: IOException =>
          if (e.isInstanceOf[InterruptedIOException]) Thread.currentThread.interrupt()
          os = null
          LogLog.warn("Detected problem with connection: " + e)
          if (reconnectionDelay > 0) fireConnector()
          else errorHandler.error(
            "Detected problem with connection, not reconnecting.",
            e,
            ErrorCode.GENERIC_FAILURE,
          )
      }
  }

  private def fireConnector(): Unit = {
    if (connector == null) {
      LogLog.debug("Starting a new connector thread.")
      connector = new SocketConnector
      connector.setDaemon(true)
      connector.setPriority(Thread.MIN_PRIORITY)
      connector.start()
    }
  }

  /** The SocketAppender does not use a layout. Hence, this method returns <code>false</code>. */
  override def requiresLayout = false

  class SocketConnector extends Thread {
    var interrupted = false

    override def run(): Unit = {
      var socket: Socket = null
      var c = true
      while (c && !interrupted)
        try {
          Thread.sleep(reconnectionDelay)
          LogLog.debug("Attempting connection to " + address.getHostName)
          socket = new Socket(address, port)
          this.synchronized {
            os = new ObjectOutputStream(socket.getOutputStream)
            connector = null
            LogLog.debug("Connection established. Exiting connector thread.")
            c = false
          }
        } catch {
          case _: InterruptedException =>
            LogLog.debug("Connector interrupted. Leaving loop.")
            return
          case _: ConnectException =>
            LogLog.debug("Remote host " + address.getHostName + " refused connection.")
          case e: IOException =>
            if (e.isInstanceOf[InterruptedIOException]) Thread.currentThread.interrupt()
            LogLog.debug("Could not connect to " + address.getHostName + ". Exception is " + e)
        }
    }
  }
}
