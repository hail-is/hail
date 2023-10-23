package is.hail.io.fs

import is.hail.shadedazure.com.azure.core.credential.{AzureSasCredential, TokenCredential}
import is.hail.shadedazure.com.azure.identity.{ClientSecretCredential, ClientSecretCredentialBuilder, DefaultAzureCredential, DefaultAzureCredentialBuilder, ManagedIdentityCredentialBuilder}
import is.hail.shadedazure.com.azure.storage.blob.models.{BlobItem, BlobProperties, BlobRange, BlobStorageException, ListBlobsOptions}
import is.hail.shadedazure.com.azure.storage.blob.specialized.BlockBlobClient
import is.hail.shadedazure.com.azure.storage.blob.{BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientBuilder}
import is.hail.shadedazure.com.azure.core.http.HttpClient
import is.hail.shadedazure.com.azure.core.util.HttpClientOptions
import is.hail.services.retryTransientErrors
import is.hail.io.fs.FSUtil.{containsWildcard, dropTrailingSlash}
import is.hail.services.Requester.httpClient
import org.apache.log4j.Logger
import org.apache.commons.io.IOUtils

import java.net.URI
import is.hail.utils._
import org.json4s
import org.json4s.jackson.JsonMethods
import org.json4s.Formats

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileNotFoundException, OutputStream}
import java.nio.file.Paths
import java.time.Duration
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import org.json4s.{DefaultFormats, Formats, JInt, JObject, JString, JValue}

abstract class AzureStorageFSURL(
  val account: String,
  val container: String,
  val path: String,
  val sasToken: Option[String]
) extends FSURL {

  def addPathComponent(c: String): AzureStorageFSURL = {
    if (path == "")
      withPath(c)
    else
      withPath(s"$path/$c")
  }
  def withPath(newPath: String): AzureStorageFSURL
  def fromString(s: String): AzureStorageFSURL = AzureStorageFS.parseUrl(s)

  def prefix: String
  def getPath: String = path

  override def toString(): String = {
    val pathPart = if (path == "") "" else s"/$path"
    val sasTokenPart = sasToken.getOrElse("")

    prefix + pathPart + sasTokenPart
  }
}

class AzureStorageFSHailAzURL(
  account: String,
  container: String,
  path: String,
  sasToken: Option[String]
) extends AzureStorageFSURL(account, container, path, sasToken) {

  override def withPath(newPath: String): AzureStorageFSHailAzURL = {
    new AzureStorageFSHailAzURL(account, container, newPath, sasToken)
  }

  override def prefix: String = s"hail-az://$account/$container"
}

class AzureStorageFSHttpsURL(
  account: String,
  container: String,
  path: String,
  sasToken: Option[String]
) extends AzureStorageFSURL(account, container, path, sasToken) {

  override def withPath(newPath: String): AzureStorageFSHttpsURL = {
    new AzureStorageFSHttpsURL(account, container, newPath, sasToken)
  }

  override def prefix: String = s"https://$account.blob.core.windows.net/$container"
}


object AzureStorageFS {
  private val HAIL_AZ_URI_REGEX = "^hail-az:\\/\\/([a-z0-9_\\-\\.]+)\\/([a-z0-9_\\-\\.]+)(\\/.*)?".r
  private val AZURE_HTTPS_URI_REGEX = "^https:\\/\\/([a-z0-9_\\-\\.]+)\\.blob\\.core\\.windows\\.net\\/([a-z0-9_\\-\\.]+)(\\/.*)?".r

  private val log = Logger.getLogger(getClass.getName)

  val schemes: Array[String] = Array("hail-az", "https")

  def parseUrl(filename: String): AzureStorageFSURL = {
    val scheme = filename.split(":")(0)
    if (scheme == "hail-az") {
      parseHailAzUrl(filename)
    } else if (scheme == "https") {
      parseHttpsUrl(filename)
    } else {
      throw new IllegalArgumentException(s"Invalid scheme, expected hail-az or https: $scheme")
    }
  }

  private[this] def parseHttpsUrl(filename: String): AzureStorageFSHttpsURL = {
    AZURE_HTTPS_URI_REGEX
      .findFirstMatchIn(filename)
      .map(m => {
        val (path, sasToken) = parsePathAndQuery(m.group(3))
        new AzureStorageFSHttpsURL(m.group(1), m.group(2), path, sasToken)
      })
      .getOrElse(throw new IllegalArgumentException("ABS URI must be of the form https://<ACCOUNT>.blob.core.windows.net/<CONTAINER>/<PATH>"))
  }

  private[this] def parseHailAzUrl(filename: String): AzureStorageFSHailAzURL = {
    HAIL_AZ_URI_REGEX
      .findFirstMatchIn(filename)
      .map(m => {
        val (path, sasToken) = parsePathAndQuery(m.group(3))
        new AzureStorageFSHailAzURL(m.group(1), m.group(2), path, sasToken)
      })
      .getOrElse(throw new IllegalArgumentException("hail-az URI must be of the form hail-az://<ACCOUNT>/<CONTAINER>/<PATH>"))
  }

  private[this] def parsePathAndQuery(maybeNullPath: String): (String, Option[String]) = {
    val pathAndMaybeQuery = Paths.get(if (maybeNullPath == null) "" else maybeNullPath.stripPrefix("/")).normalize.toString

    // Unfortunately it is difficult to tell the difference between a glob pattern and a SAS token,
    // so we make the imperfect assumption that if the query string starts with at least one
    // key-value pair we will interpret it as a SAS token and not a glob pattern
    val indexOfLastQuestionMark = pathAndMaybeQuery.lastIndexOf("?")
    if (indexOfLastQuestionMark == -1) {
      (pathAndMaybeQuery, None)
    } else {
      val (path, queryString) = pathAndMaybeQuery.splitAt(indexOfLastQuestionMark)
      queryString.split("&")(0).split("=") match {
        case Array(k, v) => (path, Some(queryString))
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
      new BlobStorageFileListEntry(url.toString, properties.getLastModified.toEpochSecond, properties.getContentLength, false)
    }
  }

  def dir(url: AzureStorageFSURL): BlobStorageFileListEntry = {
    new BlobStorageFileListEntry(url.toString, null, 0, true)
  }
}

class AzureBlobServiceClientCache(credential: TokenCredential, val httpClientOptions: HttpClientOptions) {
  private[this] lazy val clients = mutable.Map[(String, String, Option[String]), BlobServiceClient]()

  def getServiceClient(url: AzureStorageFSURL): BlobServiceClient = {
    val k = (url.account, url.container, url.sasToken)

    clients.get(k) match {
      case Some(client) => client
      case None =>
        val clientBuilder = url.sasToken match {
          case Some(sasToken) => new BlobServiceClientBuilder().credential(new AzureSasCredential(sasToken))
          case None => new BlobServiceClientBuilder().credential(credential)
        }

        val blobServiceClient = clientBuilder
          .endpoint(s"https://${url.account}.blob.core.windows.net")
          .clientOptions(httpClientOptions)
          .buildClient()
        clients += (k -> blobServiceClient)
        blobServiceClient
    }
  }

  def setPublicAccessServiceClient(url: AzureStorageFSURL): Unit = {
    val blobServiceClient = new BlobServiceClientBuilder()
      .endpoint(s"https://${url.account}.blob.core.windows.net")
      .clientOptions(httpClientOptions)
      .buildClient()
    clients += ((url.account, url.container, url.sasToken) -> blobServiceClient)
  }
}


class AzureStorageFS(val credentialsJSON: Option[String] = None) extends FS {
  type URL = AzureStorageFSURL

  override def parseUrl(filename: String): URL = AzureStorageFS.parseUrl(filename)

  override def validUrl(filename: String): Boolean =
    try {
      parseUrl(filename)
      true
    } catch {
      case _: IllegalArgumentException => false
    }

  def urlAddPathComponent(url: URL, component: String): URL = url.addPathComponent(component)

  def getConfiguration(): Unit = ()

  def setConfiguration(config: Any): Unit = { }

  // ABS errors if you attempt credentialed access for a public container,
  // so we try once with credentials, if that fails use anonymous access for
  // that container going forward.
  def handlePublicAccessError[T](url: URL)(f: => T): T = {
    retryTransientErrors {
      try {
        f
      } catch {
        case e: BlobStorageException if e.getStatusCode == 401 =>
          serviceClientCache.setPublicAccessServiceClient(url)
          f
      }
    }
  }

  private lazy val httpClientOptions = new HttpClientOptions()
    .setReadTimeout(Duration.ofSeconds(5))
    .setConnectTimeout(Duration.ofSeconds(5))
    .setConnectionIdleTimeout(Duration.ofSeconds(5))
    .setWriteTimeout(Duration.ofSeconds(5))

  private lazy val serviceClientCache = credentialsJSON match {
    case None =>
      val credential: DefaultAzureCredential = new DefaultAzureCredentialBuilder().build()
      new AzureBlobServiceClientCache(credential, httpClientOptions)
    case Some(keyData) =>
      implicit val formats: Formats = defaultJSONFormats
      val kvs = JsonMethods.parse(keyData)
      val appId = (kvs \ "appId").extract[String]
      val password = (kvs \ "password").extract[String]
      val tenant = (kvs \ "tenant").extract[String]

      val clientSecretCredential: ClientSecretCredential = new ClientSecretCredentialBuilder()
        .clientId(appId)
        .clientSecret(password)
        .tenantId(tenant)
        .build()
      new AzureBlobServiceClientCache(clientSecretCredential, httpClientOptions)
  }

  // Set to max timeout for blob storage of 30 seconds
  // https://docs.microsoft.com/en-us/rest/api/storageservices/setting-timeouts-for-blob-service-operations
  private val timeout = Duration.ofSeconds(30)

  def getBlobClient(url: URL): BlobClient = retryTransientErrors {
    serviceClientCache.getServiceClient(url).getBlobContainerClient(url.container).getBlobClient(url.path)
  }

  def getContainerClient(url: URL): BlobContainerClient = retryTransientErrors {
    serviceClientCache.getServiceClient(url).getBlobContainerClient(url.container)
  }

  def openNoCompression(url: URL): SeekableDataInputStream = handlePublicAccessError(url) {
    val blobClient: BlobClient = getBlobClient(url)
    val blobSize = blobClient.getProperties.getBlobSize

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] val client: BlobClient = blobClient

      val bbOS = new OutputStream {
        override def write(b: Array[Byte]): Unit = bb.put(b)
        override def write(b: Int): Unit = bb.put(b.toByte)
      }

      override def physicalSeek(newPos: Long): Unit = ()

      override def fill(): Int = {
        val pos = getPosition
        val numBytesRemainingInBlob = blobSize - pos
        val count = Math.min(numBytesRemainingInBlob, bb.capacity())
        if (count <= 0) {
          return -1
        }

        val response = retryTransientErrors {
          bb.clear()
          client.downloadStreamWithResponse(
            bbOS, new BlobRange(pos, count),
            null, null, false, timeout, null)
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

  override def readNoCompression(url: URL): Array[Byte] = handlePublicAccessError(url) {
    val client = getBlobClient(url)
    val size = client.getProperties.getBlobSize
    if (size < 2 * 1024 * 1024 * 1024) { // https://learn.microsoft.com/en-us/java/api/com.azure.storage.blob.specialized.blobclientbase?view=azure-java-stable#com-azure-storage-blob-specialized-blobclientbase-downloadcontent()
      retryTransientErrors {
        client.downloadContent().toBytes()
      }
    } else {
      val baos = new ByteArrayOutputStream()
      retryTransientErrors {
        client.downloadStream(baos)
      }
      baos.toByteArray()
    }
  }

  def createNoCompression(url: URL): PositionedDataOutputStream = retryTransientErrors {
    val blockBlobClient = getBlobClient(url).getBlockBlobClient

    val os: PositionedOutputStream = new FSPositionedOutputStream(4 * 1024 * 1024) {
      private[this] val client: BlockBlobClient = blockBlobClient
      private[this] val blobOutputStream = client.getBlobOutputStream(true)

      override def flush(): Unit = {
        bb.flip()

        if (bb.limit() > 0) {
          blobOutputStream.write(bb.array(), 0, bb.limit())
        }

        bb.clear()
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

  def delete(url: URL, recursive: Boolean): Unit = retryTransientErrors {
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
      try {
        if (fileListEntry(url).isFile) {
          blobClient.delete()
        }
      } catch {
        case e: FileNotFoundException =>
      }
    }
  }

  def listDirectory(url: URL): Array[FileListEntry] = handlePublicAccessError(url) {
    val blobContainerClient: BlobContainerClient = getContainerClient(url)
    val statList: ArrayBuffer[FileListEntry] = ArrayBuffer()

    val prefix = dropTrailingSlash(url.path) + "/"
    // collect all children of this directory (blobs and subdirectories)
    val prefixMatches = blobContainerClient.listBlobsByHierarchy(prefix)

    prefixMatches.forEach(blobItem => {
      statList += AzureStorageFileListEntry(url, blobItem)
    })

    statList.toArray
  }

  def glob(url: URL): Array[FileListEntry] = handlePublicAccessError(url) {
    globWithPrefix(prefix = url.withPath(""), path = dropTrailingSlash(url.path))
  }

  override def fileStatus(url: AzureStorageFSURL): FileStatus = retryTransientErrors {
    if (url.path == "") {
      return AzureStorageFileListEntry.dir(url)
    }

    val blobClient = getBlobClient(url)
    val blobProperties = try {
      blobClient.getProperties
    } catch {
      case e: BlobStorageException if e.getStatusCode == 404 =>
        throw new FileNotFoundException(url.toString)
    }

    new BlobStorageFileStatus(url.toString, blobProperties.getLastModified.toEpochSecond, blobProperties.getBlobSize)
  }

  private[this] def fileListEntryFromIterator(
    url: URL,
    it: Iterator[FileListEntry],
  ): FileListEntry = {
    val urlStr = url.toString
    val prefix = dropTrailingSlash(urlStr)
    val prefixWithSlash = prefix + "/"

    var continue = it.hasNext
    var fileFle: FileListEntry = null
    var dirFle: FileListEntry = null
    while (continue) {
      val fle = it.next()

      if (fle.isFile && fle.getActualUrl == urlStr) {
        // In Google, there could be a blob with the name "foo/". We return it iff the user
        // requested "foo/" with the trailing slash.
        fileFle = fle
      }
      if (fle.isDirectory && dropTrailingSlash(fle.getActualUrl) == prefix) {
        // In Google, "directory" entries always have a trailing slash.
        //
        // In Azure, "directory" entries never have a trailing slash.
        dirFle = fle
      }

      continue = it.hasNext && (fle.getActualUrl <= prefixWithSlash)
    }

    if (fileFle != null) {
      if (dirFle != null) {
        throw new FileAndDirectoryException(s"${url.toString} appears as both file ${fileFle.getActualUrl} and directory ${dirFle.getActualUrl}")
      } else {
        fileFle
      }
    } else {
      if (dirFle != null) {
        dirFle
      } else {
        throw new FileNotFoundException(url.toString)
      }
    }
  }

  override def fileListEntry(url: URL): FileListEntry = {
    if (url.getPath == "")
      return AzureStorageFileListEntry.dir(url)

    val it = {
      val containerClient = getContainerClient(url)
      val options = new ListBlobsOptions().setPrefix(dropTrailingSlash(url.getPath))
      val prefixMatches = containerClient.listBlobsByHierarchy("/", options, timeout)
      prefixMatches.iterator()
    }.asScala.map(AzureStorageFileListEntry.apply(url, _))

    fileListEntryFromIterator(url, it)
  }

  override def eTag(url: URL): Some[String] =
    handlePublicAccessError(url) {
      Some(getBlobClient(url).getProperties.getETag)
    }

  def makeQualified(filename: String): String = {
    parseUrl(filename)
    filename
  }
}
