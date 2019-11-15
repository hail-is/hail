package is.hail.shuffler

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.Route
import akka.stream.ActorMaterializer
import akka.stream.ActorMaterializerSettings
import akka.util.ByteString
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir.{IRParser, SortOrder}
import is.hail.expr.types.physical._
import is.hail.io.{ByteArrayDecoder, BufferSpec}
import is.hail.utils._
import java.io.{ ByteArrayInputStream, ByteArrayOutputStream, InputStream }
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import org.apache.log4j.LogManager
import scala.collection.mutable
import scala.io.StdIn
import scala.collection.JavaConverters._
import com.typesafe.config.ConfigFactory
import sys.process._
import scala.language.postfixOps

object WebServer {
  private[this] val log = LogManager.getLogger("Shuffler")

  def main(args: Array[String]) {
    implicit val system = ActorSystem("shuffler")
    implicit val materializer = ActorMaterializer()
    implicit val executionContext = system.dispatcher
    val shuffles = new ConcurrentHashMap[Long, Shuffle]()

    def shuffleOr404[T](
      id: Long
    )(route: Shuffle => Route
    ): Route = {
      if (!shuffles.containsKey(id)) {
        complete(StatusCodes.NotFound -> s"no shuffle $id")
      } else {
        route(shuffles.get(id))
      }
    }

    var nextId: AtomicLong = new AtomicLong(0L)
    val route = concat(
      path("api" / "v1alpha") {
        concat(
          post {
            extractDataBytes { byteSource =>
              onSuccess(byteSource.runFold(ByteString())(_ ++ _)) { bytes =>
                val bb = bytes.asByteBuffer
                val id = nextId.getAndIncrement()
                val inPartitions = ByteUtils.readInt(bb)
                val s = ByteUtils.readString(bb)
                println(s)
                val bufferSpec = BufferSpec.parse(s)
                val wireKeyType = IRParser.parsePType(ByteUtils.readString(bb)).asInstanceOf[PBaseStruct]
                val serializedSortOrder = new Array[Byte](wireKeyType.size)
                bb.get(serializedSortOrder, 0, serializedSortOrder.size)
                val sortOrder = serializedSortOrder.map(SortOrder.deserialize)
                val hasPartitioner = bb.get() == 1.toByte
                val outPartitioning = if (!hasPartitioner) {
                  ByteUtils.skipInt(bb)
                  val nOutPartitions = ByteUtils.readInt(bb)
                  Left(nOutPartitions)
                } else {
                  Right(ByteUtils.readByteArray(bb))
                }
                log.info(s"POST api/v1alpha $inPartitions $outPartitioning $sortOrder")
                shuffles.put(id,
                  new Shuffle(id,
                    wireKeyType,
                    sortOrder,
                    bufferSpec,
                    inPartitions,
                    outPartitioning))
                complete(id.toString)
              }
            }
          }
        )
      },
      path("api" / "v1alpha" / LongNumber) { id =>
        concat(
          post {
            extractDataBytes { byteSource =>
              onSuccess(byteSource.runFold(ByteString())(_ ++ _)) { bytes =>
                shuffleOr404(id) { shuffle =>
                  val bb = bytes.asByteBuffer
                  val partitionId = ByteUtils.readInt(bb)
                  val attemptId = ByteUtils.readLong(bb)
                  val pairs = ByteUtils.readInt(bb)
                  log.info(s"POST api/v1alpha/$id $partitionId $attemptId $pairs ...")
                  shuffle.addMany(partitionId, attemptId, pairs, bb)
                  complete("")
                }
              }
            }
          },
          delete {
            log.info(s"DELETE api/v1alpha/$id")
            var shuffle = shuffles.remove(id)
            if (shuffle != null) {
              shuffle.close()
            }
            shuffle = null
            System.gc()
            complete("")
          })
      },
      path("api" / "v1alpha" / LongNumber / "finish_partition") { id =>
        post {
          extractDataBytes { byteSource =>
            onSuccess(byteSource.runFold(ByteString())(_ ++ _)) { bytes =>
              shuffleOr404(id) { shuffle =>
                val bb = bytes.asByteBuffer
                val partitionId = ByteUtils.readInt(bb)
                val attemptId = ByteUtils.readLong(bb)
                log.info(s"POST api/v1alpha/$id/finish_partition $partitionId $attemptId")
                shuffle.finishPartition(partitionId, attemptId)
                complete("")
              }
            }
          }
        }
      },
      path("api" / "v1alpha" / LongNumber / "close") { id =>
        post {
          log.info(s"POST api/v1alpha/$id/close")
          shuffleOr404(id) { shuffle =>
            val out = new ByteArrayOutputStream()
            shuffle.close(out)
            complete(out.toByteArray())
          }
        }
      },
      path("api" / "v1alpha" / LongNumber / IntNumber) { (id, partitionId) =>
        concat(
          get {
            log.info(s"GET api/v1alpha/$id/$partitionId")
            shuffleOr404(id) { shuffle =>
              if (!shuffle.closed()) {
                complete(
                  StatusCodes.BadRequest -> s"shuffle $id is already closed")
              }
              val out = new ByteArrayOutputStream()
              shuffle.get(partitionId, out)
              complete(out.toByteArray())
            }
          }// ,
           // delete {
           //   log.info(s"DELETE api/v1alpha/$id/$partitionId")
           //   shuffleOr404(id) { shuffle =>
           //     shuffle.deletePartition(partitionId)
           //     if (shuffle.allPartitionsDeleted()) {
           //       shuffles.remove(id)
           //     }
           //     complete("")
           //   }
           // }
        )
      },
      path("healthcheck") {
        get {
          complete("")
        }
      }
    )

    val host = "0.0.0.0"
    val port = 80
    log.info(s"serving at ${host}:${port}")

    val basePath = (Seq("python3",
      "-c",
      "from hailtop.config import get_deploy_config; print(get_deploy_config().base_path('shuffler-0.shuffler'))") !!
    ).trim

    val namespacedRoute = if (basePath == "") {
      route
    } else {
      log.info(s"basePath $basePath")
      pathPrefix(separateOnSlashes(basePath.substring(1)))(route)
    }
    val bindingFuture = Http().bindAndHandle(namespacedRoute, host, port)
  }
}
