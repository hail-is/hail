package is.hail

import javax.net.ssl.SSLException
import java.net._
import java.io.EOFException
import java.util.concurrent.TimeoutException
import is.hail.utils._

import org.apache.http.NoHttpResponseException
import org.apache.http.ConnectionClosedException
import org.apache.http.conn.HttpHostConnectException
import org.apache.log4j.{LogManager, Logger}

import reactor.core.Exceptions.ReactiveException
import scala.util.Random
import java.io._
import com.google.cloud.storage.StorageException
import com.google.api.client.googleapis.json.GoogleJsonResponseException
import com.google.api.client.http.HttpResponseException

package object services {
  lazy val log: Logger = LogManager.getLogger("is.hail.services")

  val RETRYABLE_HTTP_STATUS_CODES: Set[Int] = {
    val s = Set(408, 429, 500, 502, 503, 504)
    if (System.getenv("HAIL_DONT_RETRY_500") == "1")
      s - 500
    else
      s
  }

  def sleepAndBackoff(delay: Double): Double = {
    val t = delay * Random.nextDouble()
    Thread.sleep((t * 1000).toInt)  // in ms
    math.min(delay * 2, 60.0)
  }

  def isLimitedRetriesError(_e: Throwable): Boolean = {
    // An exception is a "retry once error" if a rare, known bug in a dependency or in a cloud
    // provider can manifest as this exception *and* that manifestation is indistinguishable from a
    // true error.
    val e = reactor.core.Exceptions.unwrap(_e)
    e match {
      case e: SocketException =>
        e.getMessage != null && e.getMessage.contains("Connection reset")
      case e: HttpResponseException =>
        e.getStatusCode() == 400 && e.getMessage != null && (
          e.getMessage.contains("Invalid grant: account not found") ||
            e.getMessage.contains("{\"error\":\"unhandled_canonical_code_14\"}")
        )
      case e @ (_: SSLException | _: StorageException | _: IOException) =>
        val cause = e.getCause
        cause != null && isLimitedRetriesError(cause)
      case _ =>
        false
    }
  }

  def isTransientError(_e: Throwable): Boolean = {
    // ReactiveException is package private inside reactore.core.Exception so we cannot access
    // it directly for an isInstance check. AFAICT, this is the only way to check if we received
    // a ReactiveException.
    //
    // If the argument is a ReactiveException, it returns its cause. If the argument is not a
    // ReactiveException it returns the exception unmodified.
    val e = reactor.core.Exceptions.unwrap(_e)
    e match {
      case e: NoHttpResponseException =>
        true
      case e: HttpResponseException =>
        RETRYABLE_HTTP_STATUS_CODES.contains(e.getStatusCode())
      case e: ClientResponseException =>
        RETRYABLE_HTTP_STATUS_CODES.contains(e.status)
      case e: GoogleJsonResponseException =>
        RETRYABLE_HTTP_STATUS_CODES.contains(e.getStatusCode())
      case e: HttpHostConnectException =>
        true
      case e: NoRouteToHostException =>
        true
      case e: SocketTimeoutException =>
        true
      case e: java.util.concurrent.TimeoutException =>
        true
      case e: UnknownHostException =>
        true
      case e: ConnectionClosedException =>
        true
      case e: SocketException =>
        e.getMessage != null && (
          e.getMessage.contains("Connection timed out (Read failed)") ||
            e.getMessage.contains("Broken pipe") ||
            e.getMessage.contains("Connection refused"))
      case e: EOFException =>
        e.getMessage != null && (
          e.getMessage.contains("SSL peer shut down incorrectly"))
      case e: IllegalStateException =>
        // Caused by: java.lang.IllegalStateException: Timeout on blocking read for 30000000000 NANOSECONDS
        // reactor.core.publisher.BlockingSingleSubscriber.blockingGet(BlockingSingleSubscriber.java:123)
        // reactor.core.publisher.Mono.block(Mono.java:1727)
        // com.azure.storage.common.implementation.StorageImplUtils.blockWithOptionalTimeout(StorageImplUtils.java:130)
        // com.azure.storage.blob.specialized.BlobClientBase.downloadStreamWithResponse(BlobClientBase.java:731)
        // is.hail.io.fs.AzureStorageFS$$anon$1.fill(AzureStorageFS.scala:152)
        // is.hail.io.fs.FSSeekableInputStream.read(FS.scala:141)
        // ...
        e.getMessage.contains("Timeout on blocking read")
      case e @ (_: SSLException | _: StorageException | _: IOException) =>
        val cause = e.getCause

        (
          NettyProxy.isRetryableNettyIOException(e)
        ) || (
          cause != null && isTransientError(cause)
        )
      case _ =>
        false
    }
  }

  def retryTransientErrors[T](f: => T): T = {
    var delay = 0.1
    var errors = 0
    while (true) {
      try {
        return f
      } catch {
        case e: Exception =>
          errors += 1
          if (errors <= 5 && isLimitedRetriesError(e)) {
            log.warn(
              s"A limited retry error has occured. We will automatically retry " +
                s"${5 - errors} more times. Do not be alarmed. (current delay: " +
                s"$delay). The most recent error was $e.")
          } else if (!isTransientError(e)) {
            throw e
          } else if (errors % 10 == 0) {
            log.warn(s"Encountered $errors transient errors, most recent one was $e.")
          }
      }
      delay = sleepAndBackoff(delay)
    }

    throw new AssertionError("unreachable")
  }

  def formatException(e: Throwable): String = {
    using(new StringWriter()) { sw =>
      using(new PrintWriter(sw)) { pw =>
        e.printStackTrace(pw)
        sw.toString
      }
    }
  }
}
