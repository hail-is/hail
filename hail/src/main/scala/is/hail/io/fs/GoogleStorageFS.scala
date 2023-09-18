package is.hail.io.fs


import com.google.api.client.googleapis.json.GoogleJsonResponseException
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.http.HttpTransportOptions
import com.google.cloud.storage.Storage.{BlobGetOption, BlobListOption, BlobWriteOption, BlobSourceOption}
import com.google.cloud.storage.{Blob, BlobId, BlobInfo, Storage, StorageException, StorageOptions}
import com.google.cloud.{ReadChannel, WriteChannel}
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.services.retryTransientErrors
import is.hail.utils.fatal
import org.apache.log4j.Logger

import java.io.{ByteArrayInputStream, FileNotFoundException, IOException}
import java.net.URI
import java.nio.ByteBuffer
import java.nio.file.Paths
import scala.jdk.CollectionConverters.{asJavaIterableConverter, asScalaIteratorConverter, iterableAsScalaIterableConverter}


case class GoogleStorageFSURL(bucket: String, path: String) extends FSURL[GoogleStorageFSURL] {
  def addPathComponent(c: String): GoogleStorageFSURL = {
    if (path == "")
      withPath(c)
    else
      withPath(s"$path/$c")
  }
  def withPath(newPath: String): GoogleStorageFSURL = GoogleStorageFSURL(bucket, newPath)
  def fromString(s: String): GoogleStorageFSURL = GoogleStorageFS.parseUrl(s)

  def getPath: String = path

  override def toString(): String = if (path.isEmpty) {
    s"gs://$bucket"
  } else {
    s"gs://$bucket/$path"
  }
}


object GoogleStorageFS {
  private val log = Logger.getLogger(getClass.getName())
  private[this] val GCS_URI_REGEX = "^gs:\\/\\/([a-z0-9_\\-\\.]+)(\\/.*)?".r

  def parseUrl(filename: String): GoogleStorageFSURL = {
    val scheme = new URI(filename).getScheme
    if (scheme == null || scheme != "gs") {
      throw new IllegalArgumentException(s"Invalid scheme, expected gs: $scheme")
    }

    GCS_URI_REGEX.findFirstMatchIn(filename) match {
      case Some(m) =>
        val bucket = m.group(1)
        val maybePath = m.group(2)
        val path = Paths.get(if (maybePath == null) "" else maybePath.stripPrefix("/"))
        GoogleStorageFSURL(bucket, path.normalize().toString)
      case None => throw new IllegalArgumentException(s"GCS URI must be of the form: gs://bucket/path, found $filename")
    }
  }
}

object GoogleStorageFileListEntry {
  def apply(blob: Blob): BlobStorageFileListEntry = {
    val isDir = blob.isDirectory

    val name = dropTrailingSlash(blob.getName)

    new BlobStorageFileListEntry(
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
  type URL = GoogleStorageFSURL

  import GoogleStorageFS._

  override def validUrl(filename: String): Boolean =
    filename.startsWith("gs://")

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

  def openNoCompression(filename: String): SeekableDataInputStream = retryTransientErrors {
    val url = parseUrl(filename)

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] var reader: ReadChannel = null

      private[this] def retryingRead(): Int = {
        retryTransientErrors(
          { reader.read(bb) },
          reset = Some({ () => reader.seek(getPosition) })
        )
      }

      private[this] def readHandlingRequesterPays(bb: ByteBuffer): Int = {
        if (reader != null) {
          retryingRead()
        } else {
          handleRequesterPays(
            { (options: Seq[BlobSourceOption]) =>
              reader = retryTransientErrors {
                storage.reader(url.bucket, url.path, options:_*)
              }
              reader.seek(getPosition)
              retryingRead()
            },
            BlobSourceOption.userProject,
            url.bucket
          )
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
        if (reader != null) {
          reader.seek(newPos)
        }
      }
    }

    new WrappedSeekableDataInputStream(is)
  }

  override def readNoCompression(filename: String): Array[Byte] = retryTransientErrors {
    val url = parseUrl(filename)
    storage.readAllBytes(url.bucket, url.path)
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = retryTransientErrors {
    log.info(f"createNoCompression: ${filename}")
    val url = parseUrl(filename)

    val blobId = BlobId.of(url.bucket, url.path)
    val blobInfo = BlobInfo.newBuilder(blobId)
      .build()

    val os: PositionedOutputStream = new FSPositionedOutputStream(8 * 1024 * 1024) {
      private[this] var writer: WriteChannel = null

      private[this] def doHandlingRequesterPays(f: => Unit): Unit = {
        if (writer != null) {
          f
        } else {
          handleRequesterPays(
            { (options: Seq[BlobWriteOption]) =>
              writer = retryTransientErrors { storage.writer(blobInfo, options:_*) }
              f
            },
            BlobWriteOption.userProject,
            url.bucket
          )
        }
      }

      override def flush(): Unit = {
        bb.flip()

        while (bb.remaining() > 0)
          doHandlingRequesterPays {
            writer.write(bb)
          }

        bb.clear()
      }

      override def close(): Unit = {
        log.info(f"close: ${filename}")
        if (!closed) {
          flush()
          retryTransientErrors {
            doHandlingRequesterPays {
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
    val srcUrl = parseUrl(src)
    val dstUrl = parseUrl(dst)
    val srcId = BlobId.of(srcUrl.bucket, srcUrl.path)
    val dstId = BlobId.of(dstUrl.bucket, dstUrl.path)

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
          if (buckets.contains(srcUrl.bucket) && buckets.contains(dstUrl.bucket)) {
            Storage.CopyRequest.newBuilder()
              .setSourceOptions(BlobSourceOption.userProject(project))
              .setSource(srcId)
              .setTarget(dstId)
              .build()
          } else if (buckets.contains(srcUrl.bucket) || buckets.contains(dstUrl.bucket)) {
            throw new RuntimeException(s"both ${srcUrl.bucket} and ${dstUrl.bucket} must be specified in the requester_pays_buckets to copy between these buckets", exc)
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
    val url = parseUrl(filename)
    if (recursive) {
      var page = retryTransientErrors {
        handleRequesterPays(
          (options: Seq[BlobListOption]) => storage.list(url.bucket, (BlobListOption.prefix(url.path) +: options):_*),
          BlobListOption.userProject,
          url.bucket
        )
      }
      while (page != null) {
        retryTransientErrors {
          val blobs = page.getValues.asScala.map(_.getBlobId).asJava
          if (blobs.iterator().hasNext) {
            handleRequesterPays(
              { (options: Seq[BlobSourceOption]) =>
                if (options.isEmpty) {
                  storage.delete(blobs)
                } else {
                  blobs.asScala.foreach(storage.delete(_, options:_*))
                }
              },
              BlobSourceOption.userProject,
              url.bucket
            )
          }
        }
        page = page.getNextPage()
      }
    } else {
      // Storage.delete is idempotent. it returns a Boolean which is false if the file did not exist
      handleRequesterPays(
        (options: Seq[BlobSourceOption]) => storage.delete(url.bucket, url.path, options:_*),
        BlobSourceOption.userProject,
        url.bucket
      )
    }
  }

  def glob(filename: String): Array[FileListEntry] = retryTransientErrors {
    val url = parseUrl(filename)
    globWithPrefix(url.withPath(""), path = dropTrailingSlash(url.path))
  }

  def listDirectory(filename: String): Array[FileListEntry] = listDirectory(parseUrl(filename))

  override def listDirectory(url: GoogleStorageFSURL): Array[FileListEntry] = retryTransientErrors {
    val path = if (url.path.endsWith("/")) url.path else url.path + "/"

    val blobs = retryTransientErrors {
      handleRequesterPays(
        (options: Seq[BlobListOption]) => storage.list(url.bucket, (BlobListOption.prefix(path) +: BlobListOption.currentDirectory() +: options):_*),
        BlobListOption.userProject,
        url.bucket
      )
    }

    blobs.iterateAll().iterator.asScala
      .filter(b => b.getName != path) // elide directory markers created by Hadoop
      .map(b => GoogleStorageFileListEntry(b))
      .toArray
  }

  def fileListEntry(filename: String): FileListEntry = fileListEntry(parseUrl(filename))

  override def fileListEntry(url: GoogleStorageFSURL): FileListEntry = retryTransientErrors {
    val path = dropTrailingSlash(url.path)

    if (url.path == "")
      return new BlobStorageFileListEntry(s"gs://${url.bucket}", null, 0, true)

    val blobs = retryTransientErrors {
      handleRequesterPays(
        (options: Seq[BlobListOption]) => storage.list(url.bucket, (BlobListOption.prefix(path) +: BlobListOption.currentDirectory() +: options):_*),
        BlobListOption.userProject,
        url.bucket
      )
    }

    val it = blobs.getValues.iterator.asScala
    while (it.hasNext) {
      val b = it.next()
      var name = b.getName
      while (name.endsWith("/"))
        name = name.dropRight(1)
      if (name == path)
        return GoogleStorageFileListEntry(b)
    }

    throw new FileNotFoundException(url.toString())
  }

  override def eTag(filename: String): Some[String] = {
    val GoogleStorageFSURL(bucket, blob) = parseUrl(filename)
    handleRequesterPays(
      (options: Seq[BlobGetOption]) =>
        retryTransientErrors {
          Some(storage.get(bucket, blob, options:_*).getEtag)
        },
      BlobGetOption.userProject,
      bucket
    )
  }

  def makeQualified(filename: String): String = {
    if (!filename.startsWith("gs://"))
      throw new IllegalArgumentException(s"Invalid path, expected gs://bucket/path $filename")
    filename
  }
}
