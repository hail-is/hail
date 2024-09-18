package is.hail.backend

import is.hail.types.virtual.Kinds.{BlockMatrix, Matrix, Table, Value}
import is.hail.utils._
import is.hail.utils.ExecutionTimer.Timings

import java.io.Closeable
import java.net.InetSocketAddress
import java.util.concurrent._

import com.google.api.client.http.HttpStatusCodes
import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

class BackendServer(backend: Backend) extends Closeable {
  // 0 => let the OS pick an available port
  private[this] val httpServer = HttpServer.create(new InetSocketAddress(0), 10)

  private[this] val thread = {
    // This HTTP server *must not* start non-daemon threads because such threads keep the JVM
    // alive. A living JVM indicates to Spark that the job is incomplete. This does not manifest
    // when you run jobs in a local pyspark (because you'll Ctrl-C out of Python regardless of the
    // JVM's state) nor does it manifest in a Notebook (again, you'll kill the Notebook kernel
    // explicitly regardless of the JVM). It *does* manifest when submitting jobs with
    //
    //     gcloud dataproc submit ...
    //
    // or
    //
    //     spark-submit
    //
    // setExecutor(null) ensures the server creates no new threads:
    //
    // > If this method is not called (before start()) or if it is called with a null Executor, then
    // > a default implementation is used, which uses the thread which was created by the start()
    // > method.
    //
    /* Source:
     * https://docs.oracle.com/en/java/javase/11/docs/api/jdk.httpserver/com/sun/net/httpserver/HttpServer.html#setExecutor(java.util.concurrent.Executor) */
    //
    httpServer.createContext("/", Handler)
    httpServer.setExecutor(null)
    val t = Executors.defaultThreadFactory().newThread(new Runnable() {
      def run(): Unit =
        httpServer.start()
    })
    t.setDaemon(true)
    t
  }

  def port: Int = httpServer.getAddress.getPort

  def start(): Unit =
    thread.start()

  override def close(): Unit =
    httpServer.stop(10)

  private case class Request(exchange: HttpExchange, payload: JValue)

  private[this] object Handler extends HttpHandler with HttpLikeBackendRpc[Request] {

    override def handle(exchange: HttpExchange): Unit = {
      val payload = using(exchange.getRequestBody)(JsonMethods.parse(_))
      runRpc(Request(exchange, payload))
    }

    implicit override protected object Ask extends Routing {

      import Routes._

      override def route(a: Request): Route =
        a.exchange.getRequestURI.getPath match {
          case "/value/type" => TypeOf(Value)
          case "/table/type" => TypeOf(Table)
          case "/matrixtable/type" => TypeOf(Matrix)
          case "/blockmatrix/type" => TypeOf(BlockMatrix)
          case "/execute" => Execute
          case "/vcf/metadata/parse" => ParseVcfMetadata
          case "/fam/import" => ImportFam
          case "/references/load" => LoadReferencesFromDataset
          case "/references/from_fasta" => LoadReferencesFromFASTA
        }

      override def payload(a: Request): JValue = a.payload
    }

    implicit override protected object Write extends Write[Request] with ErrorHandling {

      override def timings(req: Request)(t: Timings): Unit = {
        val ts = Serialization.write(Map("timings" -> t))
        req.exchange.getResponseHeaders.add("X-Hail-Timings", ts)
      }

      override def result(req: Request)(result: Array[Byte]): Unit =
        respond(req)(HttpStatusCodes.STATUS_CODE_OK, result)

      override def error(req: Request)(t: Throwable): Unit =
        respond(req)(
          HttpStatusCodes.STATUS_CODE_SERVER_ERROR,
          jsonToBytes {
            val (shortMessage, expandedMessage, errorId) = handleForPython(t)
            JObject(
              "short" -> JString(shortMessage),
              "expanded" -> JString(expandedMessage),
              "error_id" -> JInt(errorId),
            )
          },
        )

      private[this] def respond(req: Request)(code: Int, payload: Array[Byte]): Unit = {
        req.exchange.sendResponseHeaders(code, payload.length)
        using(req.exchange.getResponseBody)(_.write(payload))
      }
    }

    implicit override protected object Context extends Context[Request] {
      override def scoped[A](req: Request)(f: ExecuteContext => A): (A, Timings) =
        backend.withExecuteContext(f)
    }
  }
}
