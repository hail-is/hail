package is.hail.io.fs

import is.hail.HailFeatureFlags
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.services.{isTransientError, retryTransientErrors}
import is.hail.utils._

import scala.jdk.CollectionConverters._

import java.io.{ByteArrayInputStream, FileNotFoundException, IOException}
import java.nio.ByteBuffer
import java.nio.file.Paths

import com.google.api.client.googleapis.json.GoogleJsonResponseException
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.{ReadChannel, WriteChannel}
import com.google.cloud.http.HttpTransportOptions
import com.google.cloud.storage.{Blob, BlobId, BlobInfo, Storage, StorageException, StorageOptions}
import com.google.cloud.storage.Storage.{
  BlobGetOption, BlobListOption, BlobSourceOption, BlobWriteOption,
}
import org.apache.log4j.Logger

case class GoogleStorageFSURL(bucket: String, path: String) extends FSURL {
  def addPathComponent(c: String): GoogleStorageFSURL =
    if (path == "")
      withPath(c)
    else
      withPath(s"$path/$c")

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
  object EnvVars {
    val GoogleApplicationCredentials = "GOOGLE_APPLICATION_CREDENTIALS"
  }

  private val log = Logger.getLogger(getClass.getName())
  private[this] val GCS_URI_REGEX = "^gs:\\/\\/([a-z0-9_\\-\\.]+)(\\/.*)?".r

  def parseUrl(filename: String): GoogleStorageFSURL = {
    val scheme = filename.split(":")(0)
    if (scheme == null || scheme != "gs") {
      throw new IllegalArgumentException(s"Invalid scheme, expected gs: $scheme")
    }

    GCS_URI_REGEX.findFirstMatchIn(filename) match {
      case Some(m) =>
        val bucket = m.group(1)
        val maybePath = m.group(2)
        val path = Paths.get(if (maybePath == null) "" else maybePath.stripPrefix("/"))
        GoogleStorageFSURL(bucket, path.normalize().toString)
      case None => throw new IllegalArgumentException(
          s"GCS URI must be of the form: gs://bucket/path, found $filename"
        )
    }
  }
}

object GoogleStorageFileListEntry {
  def apply(blob: Blob): BlobStorageFileListEntry = {
    val isDir = blob.isDirectory

    new BlobStorageFileListEntry(
      s"gs://${blob.getBucket}/${blob.getName}",
      if (isDir)
        null
      else
        blob.getUpdateTimeOffsetDateTime.toInstant().toEpochMilli(),
      blob.getSize,
      isDir,
    )
  }

  def dir(url: GoogleStorageFSURL): BlobStorageFileListEntry =
    new BlobStorageFileListEntry(url.toString, null, 0, true)
}

object RequesterPaysConfig {
  object Flags {
    val RequesterPaysProject = "gcs_requester_pays_project"
    val RequesterPaysBuckets = "gcs_requester_pays_buckets"
  }

  def fromFlags(flags: HailFeatureFlags): Option[RequesterPaysConfig] =
    FastSeq(Flags.RequesterPaysProject, Flags.RequesterPaysBuckets).map(flags.lookup) match {
      case Seq(Some(project), buckets) =>
        Some(RequesterPaysConfig(project, buckets.map(_.split(",").toSet)))
      case Seq(None, Some(buckets)) =>
        fatal(
          s"'${Flags.RequesterPaysBuckets}' requires '${Flags.RequesterPaysProject}'." +
            s"Expected: <undefined>" +
            s"  Actual: '$buckets'"
        )
      case _ =>
        None
    }
}

case class RequesterPaysConfig(project: String, buckets: Option[Set[String]] = None)
    extends Serializable

class GoogleStorageFS(
  private[this] val serviceAccountKey: Option[String] = None,
  private[this] var requesterPaysConfig: Option[RequesterPaysConfig] = None,
) extends FS {
  type URL = GoogleStorageFSURL

  import GoogleStorageFS.log

  override def parseUrl(filename: String): URL = GoogleStorageFS.parseUrl(filename)

  override def validUrl(filename: String): Boolean =
    filename.startsWith("gs://")

  def urlAddPathComponent(url: URL, component: String): URL = url.addPathComponent(component)

  def getConfiguration(): Option[RequesterPaysConfig] =
    requesterPaysConfig

  def setConfiguration(config: Any): Unit =
    requesterPaysConfig = config.asInstanceOf[Option[RequesterPaysConfig]]

  private[this] def requesterPaysOptions[T](bucket: String, makeUserProjectOption: String => T)
    : Seq[T] =
    requesterPaysConfig match {
      case Some(RequesterPaysConfig(project, buckets)) if buckets.fold(true)(_.contains(bucket)) =>
        Seq(makeUserProjectOption(project))
      case _ =>
        Seq()
    }

  object RequesterPaysFailure {
    def unapply(t: Throwable): Option[Throwable] =
      Some(t).filter {
        case e: IOException =>
          Option(e.getCause).exists {
            case RequesterPaysFailure(_) => true
            case _ => false
          }

        case exc: StorageException =>
          Option(exc.getMessage).exists { message =>
            message == "userProjectMissing" ||
            (exc.getCode == 400 && message.contains("requester pays"))
          }

        case exc: GoogleJsonResponseException =>
          Option(exc.getMessage).exists { message =>
            message == "userProjectMissing" ||
            (exc.getStatusCode == 400 && message.contains("requester pays"))
          }

        case _ =>
          false
      }
  }

  private[this] def handleRequesterPays[T, U](
    makeRequest: Seq[U] => T,
    makeUserProjectOption: String => U,
    bucket: String,
  ): T =
    try
      makeRequest(Seq())
    catch {
      case RequesterPaysFailure(_) =>
        makeRequest(requesterPaysOptions(bucket, makeUserProjectOption))
      case t: Throwable =>
        throw t
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
            ServiceAccountCredentials.fromStream(new ByteArrayInputStream(keyData.getBytes))
          )
          .setTransportOptions(transportOptions)
          .build()
          .getService
    }
  }

  def openNoCompression(url: URL): SeekableDataInputStream = retryTransientErrors {
    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] var reader: ReadChannel = null
      private[this] var options: Option[Seq[BlobSourceOption]] = None

      private[this] def readHandlingRequesterPays(bb: ByteBuffer): Int = {
        while (true) {
          try {
            if (reader == null) {
              val opts = options.getOrElse(FastSeq())
              reader = storage.reader(url.bucket, url.path, opts: _*)
              reader.seek(getPosition)
            }
            return reader.read(bb)
          } catch {
            case RequesterPaysFailure(_) if options.isEmpty =>
              reader = null
              bb.clear()
              options = Some(requesterPaysOptions(url.bucket, BlobSourceOption.userProject))
            case exc: Exception if isTransientError(exc) =>
              reader = null
              bb.clear()
          }
        }
        throw new RuntimeException("unreachable")
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

      override def physicalSeek(newPos: Long): Unit =
        if (reader != null) {
          reader.seek(newPos)
        }
    }

    new WrappedSeekableDataInputStream(is)
  }

  override def readNoCompression(url: URL): Array[Byte] = retryTransientErrors {
    storage.readAllBytes(url.bucket, url.path)
  }

  def createNoCompression(url: URL): PositionedDataOutputStream = retryTransientErrors {
    log.info(f"createNoCompression: $url")

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
              writer = retryTransientErrors(storage.writer(blobInfo, options: _*))
              f
            },
            BlobWriteOption.userProject,
            url.bucket,
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
        log.info(f"close: $url")
        if (!closed) {
          flush()
          retryTransientErrors {
            doHandlingRequesterPays {
              writer.close()
            }
          }
          closed = true
        }
        log.info(f"closed: $url")
      }
    }

    new WrappedPositionedDataOutputStream(os)
  }

  override def copy(src: URL, dst: URL, deleteSource: Boolean = false): Unit = {
    val srcId = BlobId.of(src.bucket, src.path)
    val dstId = BlobId.of(dst.bucket, dst.path)

    /* There is only one userProject for the whole request, the source takes precedence over the
     * target. */
    /* https://github.com/googleapis/java-storage/blob/0bd17b1f70e47081941a44f018e3098b37ba2c47/google-cloud-storage/src/main/java/com/google/cloud/storage/spi/v1/HttpStorageRpc.java#L1016-L1019 */
    def retryCopyIfRequesterPays(exc: Exception, message: String, code: Int): Unit = {
      if (message == null) {
        throw exc
      }

      val probablyNeedsRequesterPays =
        message.equals("userProjectMissing") || (code == 400 && message.contains("requester pays"))
      if (!probablyNeedsRequesterPays) {
        throw exc
      }

      val config = requesterPaysConfig match {
        case None =>
          throw exc
        case Some(RequesterPaysConfig(project, None)) =>
          Storage.CopyRequest.newBuilder()
            .setSourceOptions(BlobSourceOption.userProject(project))
            .setSource(srcId)
            .setTarget(dstId)
            .build()
        case Some(RequesterPaysConfig(project, Some(buckets))) =>
          if (buckets.contains(src.bucket) && buckets.contains(dst.bucket)) {
            Storage.CopyRequest.newBuilder()
              .setSourceOptions(BlobSourceOption.userProject(project))
              .setSource(srcId)
              .setTarget(dstId)
              .build()
          } else if (buckets.contains(src.bucket) || buckets.contains(dst.bucket)) {
            throw new RuntimeException(
              s"both ${src.bucket} and ${dst.bucket} must be specified in the requester_pays_buckets to copy between these buckets",
              exc,
            )
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

  def delete(url: URL, recursive: Boolean): Unit = retryTransientErrors {
    if (recursive) {
      var page = retryTransientErrors {
        handleRequesterPays(
          (options: Seq[BlobListOption]) =>
            storage.list(url.bucket, (BlobListOption.prefix(url.path) +: options): _*),
          BlobListOption.userProject,
          url.bucket,
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
                  blobs.asScala.foreach(storage.delete(_, options: _*))
                }
              },
              BlobSourceOption.userProject,
              url.bucket,
            )
          }
        }
        page = page.getNextPage()
      }
    } else {
      // Storage.delete is idempotent. it returns a Boolean which is false if the file did not exist
      handleRequesterPays(
        (options: Seq[BlobSourceOption]) => storage.delete(url.bucket, url.path, options: _*),
        BlobSourceOption.userProject,
        url.bucket,
      )
    }
  }

  def glob(url: URL): Array[FileListEntry] = retryTransientErrors {
    globWithPrefix(url.withPath(""), path = dropTrailingSlash(url.path))
  }

  override def listDirectory(url: URL): Array[FileListEntry] = retryTransientErrors {
    val path = if (url.path.endsWith("/")) url.path else url.path + "/"

    val blobs = retryTransientErrors {
      handleRequesterPays(
        (options: Seq[BlobListOption]) =>
          storage.list(
            url.bucket,
            (BlobListOption.prefix(path) +: BlobListOption.currentDirectory() +: options): _*
          ),
        BlobListOption.userProject,
        url.bucket,
      )
    }

    blobs.iterateAll().iterator.asScala
      .filter(b => b.getName != path) // elide directory markers created by Hadoop
      .map(b => GoogleStorageFileListEntry(b))
      .toArray
  }

  private[this] def getBlob(url: URL) = retryTransientErrors {
    handleRequesterPays(
      (options: Seq[BlobGetOption]) =>
        storage.get(url.bucket, url.path, options: _*),
      BlobGetOption.userProject _,
      url.bucket,
    )
  }

  override def fileStatus(url: URL): FileStatus = retryTransientErrors {
    if (url.path == "")
      return GoogleStorageFileListEntry.dir(url)

    val blob = getBlob(url)

    if (blob == null) {
      throw new FileNotFoundException(url.toString)
    }

    new BlobStorageFileStatus(
      url.toString,
      blob.getUpdateTimeOffsetDateTime.toInstant().toEpochMilli(),
      blob.getSize,
    )
  }

  override def fileListEntry(url: URL): FileListEntry = {
    if (url.getPath == "") {
      return GoogleStorageFileListEntry.dir(url)
    }

    val prefix = dropTrailingSlash(url.path)
    val it = retryTransientErrors {
      handleRequesterPays(
        (options: Seq[BlobListOption]) =>
          storage.list(
            url.bucket,
            (BlobListOption.prefix(prefix) +: BlobListOption.currentDirectory() +: options): _*
          ),
        BlobListOption.userProject _,
        url.bucket,
      )
    }.iterateAll().asScala.map(GoogleStorageFileListEntry.apply(_)).iterator

    fileListEntryFromIterator(url, it)
  }

  override def eTag(url: URL): Some[String] = {
    val GoogleStorageFSURL(bucket, blob) = url
    handleRequesterPays(
      (options: Seq[BlobGetOption]) =>
        retryTransientErrors {
          Some(storage.get(bucket, blob, options: _*).getEtag)
        },
      BlobGetOption.userProject,
      bucket,
    )
  }

  def makeQualified(filename: String): String = {
    if (!filename.startsWith("gs://"))
      throw new IllegalArgumentException(s"Invalid path, expected gs://bucket/path $filename")
    filename
  }
}
