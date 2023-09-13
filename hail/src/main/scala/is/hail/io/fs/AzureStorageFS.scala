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
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import org.json4s.{DefaultFormats, Formats, JInt, JObject, JString, JValue}

abstract class AzureStorageFSURL(
  val account: String,
  val container: String,
  val path: String,
  val sasToken: Option[String]
) extends FSURL[AzureStorageFSURL] {

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
    val scheme = new URI(filename).getScheme
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
  def apply(path: String, isDir: Boolean, blobProperties: BlobProperties): BlobStorageFileListEntry = {
    if (isDir) {
      new BlobStorageFileListEntry(path, null, 0, true)
    } else {
      new BlobStorageFileListEntry(path, blobProperties.getLastModified.toEpochSecond, blobProperties.getBlobSize, false)
    }
  }

  def apply(blobPath: String, blobItem: BlobItem): BlobStorageFileListEntry = {
    if (blobItem.isPrefix) {
      new BlobStorageFileListEntry(blobPath, null, 0, true)
    } else {
      val properties = blobItem.getProperties
      new BlobStorageFileListEntry(blobPath, properties.getLastModified.toEpochSecond, properties.getContentLength, false)
    }
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

  import AzureStorageFS.log

  override def validUrl(filename: String): Boolean =
    try {
      AzureStorageFS.parseUrl(filename)
      true
    } catch {
      case _: IllegalArgumentException => false
    }

  def getConfiguration(): Unit = ()

  def setConfiguration(config: Any): Unit = { }

  // ABS errors if you attempt credentialed access for a public container,
  // so we try once with credentials, if that fails use anonymous access for
  // that container going forward.
  def handlePublicAccessError[T](filename: String)(f: => T): T = {
    retryTransientErrors {
      try {
        f
      } catch {
        case e: BlobStorageException if e.getStatusCode == 401 =>
          serviceClientCache.setPublicAccessServiceClient(AzureStorageFS.parseUrl(filename))
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

  def getBlobClient(url: AzureStorageFSURL): BlobClient = retryTransientErrors {
    serviceClientCache.getServiceClient(url).getBlobContainerClient(url.container).getBlobClient(url.path)
  }

  def getContainerClient(url: AzureStorageFSURL): BlobContainerClient = retryTransientErrors {
    serviceClientCache.getServiceClient(url).getBlobContainerClient(url.container)
  }

  def openNoCompression(filename: String): SeekableDataInputStream = handlePublicAccessError(filename) {
    val url = AzureStorageFS.parseUrl(filename)
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

  override def readNoCompression(filename: String): Array[Byte] = handlePublicAccessError(filename) {
    val url = AzureStorageFS.parseUrl(filename)
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

  def createNoCompression(filename: String): PositionedDataOutputStream = retryTransientErrors {
    val blockBlobClient = getBlobClient(AzureStorageFS.parseUrl(filename)).getBlockBlobClient

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

  def delete(filename: String, recursive: Boolean): Unit = retryTransientErrors {
    val url = AzureStorageFS.parseUrl(filename)
    val blobClient: BlobClient = getBlobClient(url)

    if (recursive) {
      val blobContainerClient = getContainerClient(url)

      val options = new ListBlobsOptions()
      val prefix = dropTrailingSlash(url.path) + "/"
      options.setPrefix(prefix)
      val prefixMatches = blobContainerClient.listBlobs(options, timeout)

      prefixMatches.forEach(blobItem => {
        assert(!blobItem.isPrefix)
        getBlobClient(url.withPath(blobItem.getName)).delete()
      })
    } else {
      try {
        if (fileListEntry(filename).isFile) {
          blobClient.delete()
        }
      } catch {
        case e: FileNotFoundException =>
      }
    }
  }

  def listStatus(filename: String): Array[FileListEntry] = handlePublicAccessError(filename) {
    val url = AzureStorageFS.parseUrl(filename)

    val blobContainerClient: BlobContainerClient = getContainerClient(url)
    val statList: ArrayBuffer[FileListEntry] = ArrayBuffer()

    val prefix = dropTrailingSlash(url.path) + "/"
    // collect all children of this directory (blobs and subdirectories)
    val prefixMatches = blobContainerClient.listBlobsByHierarchy(prefix)

    prefixMatches.forEach(blobItem => {
      val blobPath = dropTrailingSlash(url.withPath(blobItem.getName).toString())
      statList += AzureStorageFileListEntry(blobPath, blobItem)
    })

    statList.toArray
  }

  def glob(filename: String): Array[FileListEntry] = handlePublicAccessError(filename) {
    val url = AzureStorageFS.parseUrl(filename)
    globWithPrefix(prefix = url.withPath(""), path = dropTrailingSlash(url.path))
  }

  override def fileListEntry(url: AzureStorageFSURL): FileListEntry = retryTransientErrors {
    if (url.path == "") {
      return new BlobStorageFileListEntry(url.toString, null, 0, true)
    }

    val blobClient: BlobClient = getBlobClient(url)
    val blobContainerClient: BlobContainerClient = getContainerClient(url)

    val prefix = dropTrailingSlash(url.path) + "/"
    val options: ListBlobsOptions = new ListBlobsOptions().setPrefix(prefix).setMaxResultsPerPage(1)
    val prefixMatches = blobContainerClient.listBlobs(options, timeout)
    val isDir = prefixMatches.iterator().hasNext

    val filename = dropTrailingSlash(url.toString)

    val blobProperties = if (!isDir) {
      try {
        blobClient.getProperties
      } catch {
        case e: BlobStorageException =>
          if (e.getStatusCode == 404)
            throw new FileNotFoundException(s"File not found: $filename")
          else
            throw e
      }
    } else
      null

    AzureStorageFileListEntry(filename, isDir, blobProperties)
  }

  def fileListEntry(filename: String): FileListEntry = handlePublicAccessError(filename) {
    fileListEntry(AzureStorageFS.parseUrl(filename))
  }

  override def eTag(filename: String): Some[String] =
    handlePublicAccessError(filename) {
      Some(getBlobClient(AzureStorageFS.parseUrl(filename)).getProperties.getETag)
    }


  def makeQualified(filename: String): String = {
    AzureStorageFS.parseUrl(filename)
    filename
  }
}
