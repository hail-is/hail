package is.hail.io.fs

import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.services.oauth2.AzureCloudCredentials
import is.hail.services.retryTransientErrors
import is.hail.shadedazure.com.azure.core.credential.AzureSasCredential
import is.hail.shadedazure.com.azure.core.util.HttpClientOptions
import is.hail.shadedazure.com.azure.storage.blob.{
  BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientBuilder,
}
import is.hail.shadedazure.com.azure.storage.blob.models.{
  BlobItem, BlobRange, BlobStorageException, ListBlobsOptions,
}
import is.hail.shadedazure.com.azure.storage.blob.specialized.BlockBlobClient

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

import java.io.{FileNotFoundException, OutputStream}
import java.nio.file.{Path, Paths}
import java.time.Duration

class AzureStorageFSURL(
  val account: String,
  val container: String,
  override val path: String,
  val sasToken: Option[String],
) extends FSURL[AzureStorageFSURL] {

  override def /(c: String): AzureStorageFSURL =
    if (path == "") withPath(c)
    else withPath(s"$path/$c")

  def withPath(newPath: String): AzureStorageFSURL =
    new AzureStorageFSURL(account, container, newPath, sasToken)

  def prefix: String = s"https://$account.blob.core.windows.net/$container"

  def base: String = {
    val pathPart = if (path == "") "" else s"/$path"
    prefix + pathPart
  }

  override def toString: String = {
    val sasTokenPart = sasToken.getOrElse("")
    this.base + sasTokenPart
  }
}

object AzureStorageFS {
  private val AZURE_HTTPS_URI_REGEX =
    "^https:\\/\\/([a-z0-9_\\-\\.]+)\\.blob\\.core\\.windows\\.net\\/([a-z0-9_\\-\\.]+)(\\/.*)?".r

  val RequiredOAuthScopes: Array[String] =
    Array("https://storage.azure.com/.default")

  def parseUrl(filename: String): AzureStorageFSURL = {
    AZURE_HTTPS_URI_REGEX
      .findFirstMatchIn(filename)
      .map { m =>
        val (path, sasToken) = parsePathAndQuery(m.group(3))
        new AzureStorageFSURL(m.group(1), m.group(2), path, sasToken)
      }
      .getOrElse(throw new IllegalArgumentException(
        "ABS URI must be of the form https://<ACCOUNT>.blob.core.windows.net/<CONTAINER>/<PATH>"
      ))
  }

  private[this] def parsePathAndQuery(maybeNullPath: String): (String, Option[String]) = {
    val pathAndMaybeQuery = Paths.get(if (maybeNullPath == null) ""
    else maybeNullPath.stripPrefix("/")).normalize.toString

    // Unfortunately it is difficult to tell the difference between a glob pattern and a SAS token,
    // so we make the imperfect assumption that if the query string starts with at least one
    // key-value pair we will interpret it as a SAS token and not a glob pattern
    val indexOfLastQuestionMark = pathAndMaybeQuery.lastIndexOf("?")
    if (indexOfLastQuestionMark == -1) {
      (pathAndMaybeQuery, None)
    } else {
      val (path, queryString) = pathAndMaybeQuery.splitAt(indexOfLastQuestionMark)
      queryString.split("&")(0).split("=") match {
        case Array(_, _) => (path, Some(queryString))
        case _ => (pathAndMaybeQuery, None)
      }
    }
  }
}

object AzureStorageFileListEntry {
  def apply(rootUrl: AzureStorageFSURL, blobItem: BlobItem): BlobStorageFileListEntry = {
    val url = rootUrl.withPath(blobItem.getName)
    if (blobItem.isPrefix) {
      dir(url)
    } else {
      val properties = blobItem.getProperties
      new BlobStorageFileListEntry(
        url.toString,
        properties.getLastModified.toEpochSecond,
        properties.getContentLength,
        false,
      )
    }
  }

  def dir(url: AzureStorageFSURL): BlobStorageFileListEntry =
    new BlobStorageFileListEntry(url.toString, null, 0, true)
}

case class AzureStorageFSConfig(credentials_file: Option[Path])

class AzureStorageFS(val credential: AzureCloudCredentials) extends FS {
  type URL = AzureStorageFSURL

  private[this] lazy val clients =
    mutable.Map[(String, String, Option[String]), BlobServiceClient]()

  def getServiceClient(url: URL): BlobServiceClient = {
    val k = (url.account, url.container, url.sasToken)
    clients.getOrElseUpdate(
      k, {
        val clientBuilder = url.sasToken match {
          case Some(sasToken) =>
            new BlobServiceClientBuilder().credential(new AzureSasCredential(sasToken))
          case None => new BlobServiceClientBuilder().credential(credential.value)
        }

        clientBuilder
          .endpoint(s"https://${url.account}.blob.core.windows.net")
          .clientOptions(httpClientOptions)
          .buildClient()
      },
    )
  }

  def setPublicAccessServiceClient(url: AzureStorageFSURL): Unit = {
    val blobServiceClient = new BlobServiceClientBuilder()
      .endpoint(s"https://${url.account}.blob.core.windows.net")
      .clientOptions(httpClientOptions)
      .buildClient()
    clients += ((url.account, url.container, url.sasToken) -> blobServiceClient)
  }

  override def parseUrl(filename: String): URL = AzureStorageFS.parseUrl(filename)

  override def validUrl(filename: String): Boolean =
    try {
      parseUrl(filename): Unit
      true
    } catch {
      case _: IllegalArgumentException => false
    }

  override def getConfiguration(): Unit = ()

  override def setConfiguration(config: Any): Unit = {}

  // ABS errors if you attempt credentialed access for a public container,
  // so we try once with credentials, if that fails use anonymous access for
  // that container going forward.
  def handlePublicAccessError[T](url: URL)(f: => T): T = {
    retryTransientErrors {
      try
        f
      catch {
        case e: BlobStorageException if e.getStatusCode == 401 =>
          setPublicAccessServiceClient(url)
          f
      }
    }
  }

  private lazy val httpClientOptions = new HttpClientOptions()
    .setReadTimeout(Duration.ofSeconds(5))
    .setConnectTimeout(Duration.ofSeconds(5))
    .setConnectionIdleTimeout(Duration.ofSeconds(5))
    .setWriteTimeout(Duration.ofSeconds(5))

  // Set to max timeout for blob storage of 30 seconds
  /* https://docs.microsoft.com/en-us/rest/api/storageservices/setting-timeouts-for-blob-service-operations */
  private val timeout = Duration.ofSeconds(30)

  def getBlobClient(url: URL): BlobClient = retryTransientErrors {
    getServiceClient(url).getBlobContainerClient(url.container).getBlobClient(
      url.path
    )
  }

  def getContainerClient(url: URL): BlobContainerClient = retryTransientErrors {
    getServiceClient(url).getBlobContainerClient(url.container)
  }

  override def openNoCompression(url: URL): SeekableDataInputStream = handlePublicAccessError(url) {
    val blobSize = getBlobClient(url).getProperties.getBlobSize

    val is: SeekableInputStream = new FSSeekableInputStream {
      val bbOS = new OutputStream {
        override def write(b: Array[Byte]): Unit =
          bb.put(b): Unit
        override def write(b: Int): Unit =
          bb.put(b.toByte): Unit
      }

      override def physicalSeek(newPos: Long): Unit = ()

      override def fill(): Int = {
        val pos = getPosition
        val numBytesRemainingInBlob = blobSize - pos
        val count = Math.min(numBytesRemainingInBlob, bb.capacity().toLong)
        if (count <= 0) {
          return -1
        }

        val response = retryTransientErrors {
          bb.clear()
          getBlobClient(url).downloadStreamWithResponse(
            bbOS,
            new BlobRange(pos, count),
            null,
            null,
            false,
            timeout,
            null,
          )
        }

        if (response.getStatusCode >= 200 && response.getStatusCode < 300) {
          bb.flip()
          assert(bb.position() == 0 && bb.remaining() > 0)

          bb.remaining()
        } else {
          -1
        }
      }
    }

    new WrappedSeekableDataInputStream(is)
  }

  override def createNoCompression(url: URL): PositionedDataOutputStream = retryTransientErrors {
    val blockBlobClient = getBlobClient(url).getBlockBlobClient

    val os: PositionedOutputStream = new FSPositionedOutputStream(4 * 1024 * 1024) {
      private[this] val client: BlockBlobClient = blockBlobClient
      private[this] val blobOutputStream = client.getBlobOutputStream(true)

      override def flush(): Unit = {
        bb.flip(): Unit

        if (bb.limit() > 0) {
          blobOutputStream.write(bb.array(), 0, bb.limit())
        }

        bb.clear(): Unit
      }

      override def close(): Unit = {
        if (!closed) {
          flush()
          blobOutputStream.flush()
          blobOutputStream.close()
          closed = true
        }
      }
    }

    new WrappedPositionedDataOutputStream(os)
  }

  override def delete(url: URL, recursive: Boolean): Unit = retryTransientErrors {
    val blobClient: BlobClient = getBlobClient(url)

    if (recursive) {
      val blobContainerClient = getContainerClient(url)

      val options = new ListBlobsOptions()
      val prefix = dropTrailingSlash(url.path) + "/"
      options.setPrefix(prefix)
      val prefixMatches = blobContainerClient.listBlobs(options, timeout)

      prefixMatches.forEach { blobItem =>
        assert(!blobItem.isPrefix)
        getBlobClient(url.withPath(blobItem.getName)).delete()
      }
    } else {
      try
        if (fileListEntry(url).isFile) {
          blobClient.delete()
        }
      catch {
        case _: FileNotFoundException =>
      }
    }
  }

  override def listDirectory(url: URL): Array[FileListEntry] = handlePublicAccessError(url) {
    val blobContainerClient: BlobContainerClient = getContainerClient(url)
    val statList: ArrayBuffer[FileListEntry] = ArrayBuffer()

    val prefix = dropTrailingSlash(url.path) + "/"
    // collect all children of this directory (blobs and subdirectories)
    val prefixMatches = blobContainerClient.listBlobsByHierarchy(prefix)

    prefixMatches.forEach(blobItem => statList += AzureStorageFileListEntry(url, blobItem))

    statList.toArray
  }

  override def glob(url: URL): Array[FileListEntry] = handlePublicAccessError(url) {
    globWithPrefix(prefix = url.withPath(""), path = dropTrailingSlash(url.path))
  }

  override def fileStatus(url: AzureStorageFSURL): FileStatus = retryTransientErrors {
    if (url.path == "") {
      return AzureStorageFileListEntry.dir(url)
    }

    val blobClient = getBlobClient(url)
    val blobProperties =
      try
        blobClient.getProperties
      catch {
        case e: BlobStorageException if e.getStatusCode == 404 =>
          throw new FileNotFoundException(url.toString)
      }

    new BlobStorageFileStatus(
      url.toString,
      blobProperties.getLastModified.toEpochSecond,
      blobProperties.getBlobSize,
    )
  }

  override def fileListEntry(url: URL): FileListEntry = {
    if (url.path == "")
      return AzureStorageFileListEntry.dir(url)

    val it = {
      val containerClient = getContainerClient(url)
      val options = new ListBlobsOptions().setPrefix(dropTrailingSlash(url.path))
      val prefixMatches = containerClient.listBlobsByHierarchy("/", options, timeout)
      prefixMatches.iterator()
    }.asScala.map(AzureStorageFileListEntry.apply(url, _))

    fileListEntryFromIterator(url, it)
  }

  override def eTag(url: URL): Some[String] =
    handlePublicAccessError(url) {
      Some(getBlobClient(url).getProperties.getETag)
    }

  override def makeQualified(filename: String): String = {
    parseUrl(filename): Unit
    filename
  }
}
