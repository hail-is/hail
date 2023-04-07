package is.hail.io.fs


import java.io.{ByteArrayInputStream, FileNotFoundException, IOException}
import java.net.URI
import java.nio.ByteBuffer
import java.nio.file.FileSystems
import org.apache.log4j.Logger
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.{ReadChannel, WriteChannel}
import com.google.cloud.storage.Storage.{BlobListOption, BlobWriteOption, BlobSourceOption}
import com.google.cloud.storage.{Option => StorageOption, _}
import com.google.cloud.http.HttpTransportOptions
import com.google.api.client.googleapis.json.GoogleJsonResponseException
import is.hail.io.fs.FSUtil.{containsWildcard, dropTrailingSlash}
import is.hail.services.retryTransientErrors
import is.hail.utils.fatal

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

object GoogleStorageFS {
  private val log = Logger.getLogger(getClass.getName())

  def getBucketPath(filename: String): (String, String) = {
    val uri = new URI(filename).normalize()

    val scheme = uri.getScheme
    assert(scheme != null && scheme == "gs", (uri.getScheme, filename))

    val bucket = uri.getAuthority
    assert(bucket != null, (filename, uri.toString(), uri.getScheme, uri.getAuthority, uri.getRawAuthority(), uri.getUserInfo()))

    var path = uri.getPath
    if (path.nonEmpty && path.head == '/')
      path = path.drop(1)

    (bucket, path)
  }
}

object GoogleStorageFileStatus {
  def apply(blob: Blob): BlobStorageFileStatus = {
    val isDir = blob.isDirectory

    val name = dropTrailingSlash(blob.getName)

    new BlobStorageFileStatus(
      s"gs://${ blob.getBucket }/$name",
      if (isDir)
        null
      else
        blob.getUpdateTimeOffsetDateTime.toInstant().toEpochMilli(),
      blob.getSize,
      isDir)
  }
}

object RequesterPaysConfiguration {
  def fromFlags(requesterPaysProject: String, requesterPaysBuckets: String): Option[RequesterPaysConfiguration] = {
    if (requesterPaysProject == null) {
      if (requesterPaysBuckets == null) {
        None
      } else {
        fatal(s"Expected gcs_requester_pays_buckets flag to be unset when gcs_requester_pays_project is unset, but instead found: $requesterPaysBuckets")
      }
    } else {
      val buckets = if (requesterPaysBuckets == null) {
        None
      } else {
        Some(requesterPaysBuckets.split(",").toSet)
      }
      Some(RequesterPaysConfiguration(requesterPaysProject, buckets))
    }
  }
}

case class RequesterPaysConfiguration(
  val project: String,
  val buckets: Option[Set[String]] = None
) extends Serializable

class GoogleStorageFS(
  private[this] val serviceAccountKey: Option[String] = None,
  private[this] var requesterPaysConfiguration: Option[RequesterPaysConfiguration] = None
) extends FS {
  import GoogleStorageFS._

  def getConfiguration(): Option[RequesterPaysConfiguration] = {
    requesterPaysConfiguration
  }

  def setConfiguration(config: Any): Unit = {
    requesterPaysConfiguration = config.asInstanceOf[Option[RequesterPaysConfiguration]]
  }

  private[this] def requesterPaysOptions[T](bucket: String, makeUserProjectOption: String => T): Seq[T] = {
    requesterPaysConfiguration match {
      case None =>
        Seq()
      case Some(RequesterPaysConfiguration(project, None)) =>
        Seq(makeUserProjectOption(project))
      case Some(RequesterPaysConfiguration(project, Some(buckets))) =>
        if (buckets.contains(bucket)) {
          Seq(makeUserProjectOption(project))
        } else {
          Seq()
        }
    }
  }

  private[this] def retryIfRequesterPays[T, U](
    exc: Exception,
    message: String,
    code: Int,
    makeRequest: Seq[U] => T,
    makeUserProjectOption: String => U,
    bucket: String
  ): T = {
    if (message == null) {
      throw exc
    }

    val probablyNeedsRequesterPays = message.equals("userProjectMissing") || (code == 400 && message.contains("requester pays"))
    if (!probablyNeedsRequesterPays) {
      throw exc
    }

    makeRequest(requesterPaysOptions(bucket, makeUserProjectOption))
  }

  def retryIfRequesterPays[T, U](
    exc: Throwable,
    makeRequest: Seq[U] => T,
    makeUserProjectOption: String => U,
    bucket: String
  ): T = exc match {
    case exc: IOException if exc.getCause() != null =>
      retryIfRequesterPays(exc.getCause(), makeRequest, makeUserProjectOption, bucket)
    case exc: StorageException =>
      retryIfRequesterPays(exc, exc.getMessage(), exc.getCode(), makeRequest, makeUserProjectOption, bucket)
    case exc: GoogleJsonResponseException =>
      retryIfRequesterPays(exc, exc.getMessage(), exc.getStatusCode(), makeRequest, makeUserProjectOption, bucket)
    case exc: Throwable =>
      throw exc
  }

  private[this] def handleRequesterPays[T, U](
    makeRequest: Seq[U] => T,
    makeUserProjectOption: String => U,
    bucket: String
  ): T = {
    try {
      makeRequest(Seq())
    } catch {
      case exc: Throwable =>
        retryIfRequesterPays(exc, makeRequest, makeUserProjectOption, bucket)
    }
  }

  private lazy val storage: Storage = {
    val transportOptions = HttpTransportOptions.newBuilder()
      .setConnectTimeout(5000)
      .setReadTimeout(5000)
      .build()
    serviceAccountKey match {
      case None =>
        log.info("Initializing google storage client from latent credentials")
        StorageOptions.newBuilder()
          .setTransportOptions(transportOptions)
          .build()
          .getService
      case Some(keyData) =>
        log.info("Initializing google storage client from service account key")
        StorageOptions.newBuilder()
          .setCredentials(
            ServiceAccountCredentials.fromStream(new ByteArrayInputStream(keyData.getBytes)))
          .setTransportOptions(transportOptions)
          .build()
          .getService
    }
  }

  def asCacheable(): CacheableGoogleStorageFS = new CacheableGoogleStorageFS(serviceAccountKey, requesterPaysConfiguration, null)

  def asCacheable(sessionID: String): CacheableGoogleStorageFS = new CacheableGoogleStorageFS(serviceAccountKey, requesterPaysConfiguration, sessionID)

  def openNoCompression(filename: String, _debug: Boolean = false): SeekableDataInputStream = retryTransientErrors {
    assert(!_debug)
    val (bucket, path) = getBucketPath(filename)

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] var reader: ReadChannel = null
      private[this] var lazyPosition: Long = 0L

      private[this] def readHandlingRequesterPays(bb: ByteBuffer): Int = {
        if (reader != null) {
          reader.read(bb)
        } else {
          handleRequesterPays(
            { (options: Seq[BlobSourceOption]) =>
              reader = storage.reader(bucket, path, options:_*)
              reader.seek(lazyPosition)
              reader.read(bb)
            },
            BlobSourceOption.userProject _,
            bucket
          )
        }
      }

      private[this] def seekHandlingRequesterPays(newPos: Long): Unit = {
        if (reader != null) {
          reader.seek(newPos)
        } else {
          lazyPosition = newPos
        }
      }

      override def close(): Unit = {
        if (!closed) {
          if (reader != null) {
            reader.close()
          }
          closed = true
        }
      }

      override def fill(): Int = {
        bb.clear()

        // read some bytes
        var n = 0
        while (n == 0) {
          n = readHandlingRequesterPays(bb)
          if (n == -1) {
            return -1
          }
        }
        bb.flip()

        assert(bb.position() == 0 && bb.remaining() > 0)
        return n
      }

      override def physicalSeek(newPos: Long): Unit = {
        seekHandlingRequesterPays(newPos)
      }
    }

    new WrappedSeekableDataInputStream(is)
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = retryTransientErrors {
    log.info(f"createNoCompression: ${filename}")
    val (bucket, path) = getBucketPath(filename)

    val blobId = BlobId.of(bucket, path)
    val blobInfo = BlobInfo.newBuilder(blobId)
      .build()

    val os: PositionedOutputStream = new FSPositionedOutputStream(8 * 1024 * 1024) {
      private[this] var writer: WriteChannel = null

      private[this] def writeHandlingRequesterPays(): Int = {
        if (writer != null) {
          writer.write(bb)
        } else {
          handleRequesterPays(
            { (options: Seq[BlobWriteOption]) =>
              writer = storage.writer(blobInfo, options:_*)
              writer.write(bb)
            },
            BlobWriteOption.userProject _,
            bucket
          )
        }
      }

      override def flush(): Unit = {
        bb.flip()

        while (bb.remaining() > 0)
          writeHandlingRequesterPays()

        bb.clear()
      }

      override def close(): Unit = {
        log.info(f"close: ${filename}")
        if (!closed) {
          flush()
          if (writer != null) {
            retryTransientErrors {
              writer.close()
            }
          }
          closed = true
        }
        log.info(f"closed: ${filename}")
      }
    }

    new WrappedPositionedDataOutputStream(os)
  }

  override def copy(src: String, dst: String, deleteSource: Boolean = false): Unit = {
    val (srcBucket, srcPath) = getBucketPath(src)
    val (dstBucket, dstPath) = getBucketPath(dst)
    val srcId = BlobId.of(srcBucket, srcPath)
    val dstId = BlobId.of(dstBucket, dstPath)

    // There is only one userProject for the whole request, the source takes precedence over the target.
    // https://github.com/googleapis/java-storage/blob/0bd17b1f70e47081941a44f018e3098b37ba2c47/google-cloud-storage/src/main/java/com/google/cloud/storage/spi/v1/HttpStorageRpc.java#L1016-L1019
    def retryCopyIfRequesterPays(exc: Exception, message: String, code: Int): Unit = {
      if (message == null) {
        throw exc
      }

      val probablyNeedsRequesterPays = message.equals("userProjectMissing") || (code == 400 && message.contains("requester pays"))
      if (!probablyNeedsRequesterPays) {
        throw exc
      }

      val config = requesterPaysConfiguration match {
        case None =>
          throw exc
        case Some(RequesterPaysConfiguration(project, None)) =>
          Storage.CopyRequest.newBuilder()
            .setSourceOptions(BlobSourceOption.userProject(project))
            .setSource(srcId)
            .setTarget(dstId)
            .build()
        case Some(RequesterPaysConfiguration(project, Some(buckets))) =>
          if (buckets.contains(srcBucket) && buckets.contains(dstBucket)) {
            Storage.CopyRequest.newBuilder()
              .setSourceOptions(BlobSourceOption.userProject(project))
              .setSource(srcId)
              .setTarget(dstId)
              .build()
          } else if (buckets.contains(srcBucket) || buckets.contains(dstBucket)) {
            throw new RuntimeException(s"both $srcBucket and $dstBucket must be specified in the requester_pays_buckets to copy between these buckets", exc)
          } else {
            throw exc
          }
      }
      storage.copy(config).getResult() // getResult is necessary to cause this to go to completion
    }

    def discoverExceptionThenRetryCopyIfRequesterPays(exc: Throwable): Unit = exc match {
      case exc: IOException if exc.getCause() != null =>
        discoverExceptionThenRetryCopyIfRequesterPays(exc.getCause())
      case exc: StorageException =>
        retryCopyIfRequesterPays(exc, exc.getMessage(), exc.getCode())
      case exc: GoogleJsonResponseException =>
        retryCopyIfRequesterPays(exc, exc.getMessage(), exc.getStatusCode())
      case exc: Throwable =>
        throw exc
    }


    try {
      storage.copy(
        Storage.CopyRequest.newBuilder()
          .setSource(srcId)
          .setTarget(dstId)
          .build()
      ).getResult() // getResult is necessary to cause this to go to completion
    } catch {
      case exc: Throwable =>
        discoverExceptionThenRetryCopyIfRequesterPays(exc)
    }

    if (deleteSource)
      storage.delete(srcId)
  }

  def delete(filename: String, recursive: Boolean): Unit = retryTransientErrors {
    val (bucket, path) = getBucketPath(filename)
    if (recursive) {
      var page = retryTransientErrors {
        handleRequesterPays(
          (options: Seq[BlobListOption]) => storage.list(bucket, (BlobListOption.prefix(path) +: options):_*),
          BlobListOption.userProject _,
          bucket
        )
      }
      while (page != null) {
        retryTransientErrors {
          val blobs = page.getValues.asScala.map(_.getBlobId).asJava
          if (blobs.iterator().hasNext) {
            handleRequesterPays(
              { (options: Seq[BlobSourceOption]) =>
                if (options.size == 0) {
                  storage.delete(blobs)
                } else {
                  blobs.asScala.foreach(storage.delete(_, options:_*))
                }
              },
              BlobSourceOption.userProject _,
              bucket
            )
          }
        }
        page = page.getNextPage()
      }
    } else {
      // Storage.delete is idempotent. it returns a Boolean which is false if the file did not exist
      handleRequesterPays(
        (options: Seq[BlobSourceOption]) => storage.delete(bucket, path, options:_*),
        BlobSourceOption.userProject _,
        bucket
      )
    }
  }

  def glob(filename: String): Array[FileStatus] = retryTransientErrors {
    var (bucket, path) = getBucketPath(filename)
    path = dropTrailingSlash(path)

    globWithPrefix(prefix = s"gs://$bucket", path = path)
  }

  def listStatus(filename: String): Array[FileStatus] = retryTransientErrors {
    var (bucket, path) = getBucketPath(filename)
    if (!path.endsWith("/"))
      path = path + "/"

    val blobs = retryTransientErrors {
      handleRequesterPays(
        (options: Seq[BlobListOption]) => storage.list(bucket, (BlobListOption.prefix(path) +: BlobListOption.currentDirectory() +: options):_*),
        BlobListOption.userProject _,
        bucket
      )
    }

    blobs.getValues.iterator.asScala
      .filter(b => b.getName != path) // elide directory markers created by Hadoop
      .map(b => GoogleStorageFileStatus(b))
      .toArray
  }

  def fileStatus(filename: String): FileStatus = retryTransientErrors {
    var (bucket, path) = getBucketPath(filename)
    path = dropTrailingSlash(path)

    if (path == "")
      return new BlobStorageFileStatus(s"gs://$bucket", null, 0, true)

    val blobs = retryTransientErrors {
      handleRequesterPays(
        (options: Seq[BlobListOption]) => storage.list(bucket, (BlobListOption.prefix(path) +: BlobListOption.currentDirectory() +: options):_*),
        BlobListOption.userProject _,
        bucket
      )
    }

    val it = blobs.getValues.iterator.asScala
    while (it.hasNext) {
      val b = it.next()
      var name = b.getName
      while (name.endsWith("/"))
        name = name.dropRight(1)
      if (name == path)
        return GoogleStorageFileStatus(b)
    }

    throw new FileNotFoundException(filename)
  }

  def makeQualified(filename: String): String = {
    if (!filename.startsWith("gs://"))
      throw new IllegalArgumentException(s"Invalid path, expected gs://bucket/path $filename")
    filename
  }
}

class CacheableGoogleStorageFS(
  serviceAccountKey: Option[String],
  requesterPaysConfiguration: Option[RequesterPaysConfiguration] = None,
  @transient val sessionID: String
) extends GoogleStorageFS(serviceAccountKey, requesterPaysConfiguration) with ServiceCacheableFS {
}
