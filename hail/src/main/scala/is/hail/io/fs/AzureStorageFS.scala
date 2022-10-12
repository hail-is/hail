package is.hail.io.fs

import com.azure.core.credential.TokenCredential
import com.azure.identity.{ClientSecretCredential, ClientSecretCredentialBuilder, DefaultAzureCredential, DefaultAzureCredentialBuilder}
import com.azure.storage.blob.models.{BlobProperties, BlobRange, ListBlobsOptions}
import com.azure.storage.blob.specialized.AppendBlobClient
import com.azure.storage.blob.{BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientBuilder}
import is.hail.services.retryTransientErrors
import is.hail.io.fs.AzureStorageFS.getAccountContainerPath
import is.hail.io.fs.FSUtil.{containsWildcard, dropTrailingSlash}
import org.apache.log4j.Logger

import java.net.URI
import is.hail.utils.{fatal, defaultJSONFormats}
import org.json4s
import org.json4s.jackson.JsonMethods
import org.json4s.Formats

import java.io.{ByteArrayInputStream, FileNotFoundException, OutputStream}
import java.nio.file.FileSystems
import java.time.Duration
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


object AzureStorageFS {
  private val pathRegex = "/([^/]+)(.*)".r

  private val log = Logger.getLogger(getClass.getName)

  val schemes: Array[String] = Array("hail-az")

  def getAccountContainerPath(filename: String): (String, String, String) = {
    val uri = new URI(filename).normalize()

    val scheme = uri.getScheme
    if (scheme == null || !schemes.contains(scheme)) {
      throw new IllegalArgumentException(s"invalid scheme, expected hail-az: $scheme")
    }

    val account = uri.getAuthority
    if (account == null) {
      throw new IllegalArgumentException(s"Invalid path, expected hail-az://accountName/containerName/blobPath: $filename")
    }

    val (container, path) = pathRegex.findFirstMatchIn(uri.getPath) match {
      case Some(filenameMatch) =>
        val container = filenameMatch.group(1)
        val path = filenameMatch.group(2)
        if (path != "") {
          assert(path.startsWith("/"))
          (container, path.substring(1))
        } else {
          (container, "")
        }
      case None =>
          fatal(s"filename $filename is not in the correct format. hail-az://account/container/blobPath")
    }

    (account, container, path)
  }
}

object AzureStorageFileStatus {
  def apply(blobProperties: BlobProperties, path: String, isDir: Boolean): BlobStorageFileStatus = {
    val modificationTime = blobProperties.getLastModified.toEpochSecond
    val size = blobProperties.getBlobSize

    new BlobStorageFileStatus(path, modificationTime, size, isDir)
  }
}

class AzureBlobServiceClientCache(credential: TokenCredential) {
  private lazy val clientBuilder: BlobServiceClientBuilder = new BlobServiceClientBuilder()
  private lazy val clients: mutable.Map[String, BlobServiceClient] = mutable.Map()

  def getServiceClient(account: String): BlobServiceClient = {
    clients.get(account) match {
      case Some(client) => client
      case None =>
        val blobServiceClient = clientBuilder
          .credential(credential)
          .endpoint(s"https://$account.blob.core.windows.net")
          .buildClient()
        clients += (account -> blobServiceClient)
        blobServiceClient
    }
  }
}


class AzureStorageFS(val credentialsJSON: Option[String] = None) extends FS {
  def getConfiguration(): Unit = ()

  def setConfiguration(config: Any): Unit = { }

  private lazy val serviceClientCache = credentialsJSON match {
    case None =>
      val credential: DefaultAzureCredential = new DefaultAzureCredentialBuilder().build()
      new AzureBlobServiceClientCache(credential)
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
      new AzureBlobServiceClientCache(clientSecretCredential)
  }

  // Set to max timeout for blob storage of 30 seconds
  // https://docs.microsoft.com/en-us/rest/api/storageservices/setting-timeouts-for-blob-service-operations
  private val timeout = Duration.ofSeconds(30)

  def getBlobServiceClient(account: String): BlobServiceClient = retryTransientErrors {
    serviceClientCache.getServiceClient(account)
  }

  def getBlobClient(account: String, container: String, path: String): BlobClient = retryTransientErrors {
    getBlobServiceClient(account).getBlobContainerClient(container).getBlobClient(path)
  }

  def getContainerClient(account: String, container: String): BlobContainerClient = retryTransientErrors {
    getBlobServiceClient(account).getBlobContainerClient(container)
  }

  def openNoCompression(filename: String): SeekableDataInputStream = retryTransientErrors {
    val (account, container, path) = getAccountContainerPath(filename)
    val blobClient: BlobClient = getBlobClient(account, container, path)
    val blobSize = blobClient.getProperties.getBlobSize

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] val client: BlobClient = blobClient

      override def fill(): Int = {
        val pos = getPosition
        val numBytesRemainingInBlob = blobSize - pos
        val count = Math.min(numBytesRemainingInBlob, bb.capacity())
        if (count <= 0) {
          return -1
        }

        val outputStreamToBuffer: OutputStream = (i: Int) => {
          bb.put(i.toByte)
        }
        val response = retryTransientErrors {
          bb.clear()
          client.downloadStreamWithResponse(
            outputStreamToBuffer, new BlobRange(pos, count),
            null, null, false, timeout, null)
        }

        if (response.getStatusCode >= 200 && response.getStatusCode < 300) {
          bb.flip()
          assert(bb.position() == 0 && bb.remaining() > 0)
          return bb.remaining()
        }

        -1
      }
    }

    new WrappedSeekableDataInputStream(is)
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = retryTransientErrors {
    val (account, container, path) = getAccountContainerPath(filename)
    val appendClient = getBlobClient(account, container, path).getAppendBlobClient
    appendClient.create(true)

    val os: PositionedOutputStream = new FSPositionedOutputStream {
      private[this] val client: AppendBlobClient = appendClient

      override def flush(): Unit = {
        bb.flip()

        if (bb.limit() > 0) {
          client.appendBlock(new ByteArrayInputStream(bb.array(), 0, bb.limit()), bb.limit())
        }

        bb.clear()
      }

      override def close(): Unit = {
        if (!closed) {
          flush()
          closed = true
        }
      }
    }

    new WrappedPositionedDataOutputStream(os)
  }

  def delete(filename: String, recursive: Boolean): Unit = retryTransientErrors {
    val (account, container, path) = getAccountContainerPath(filename)
    val blobClient: BlobClient = getBlobClient(account, container, path)

    if (recursive) {
      val blobContainerClient = getContainerClient(account, container)

      val options = new ListBlobsOptions()
      val prefix = dropTrailingSlash(path) + "/"
      options.setPrefix(prefix)
      val prefixMatches = blobContainerClient.listBlobs(options, timeout)

      prefixMatches.forEach(blobItem => {
        assert(!blobItem.isPrefix)
        getBlobClient(account, container, blobItem.getName).delete()
      })
    } else {
      try {
        if (fileStatus(filename).isFile) {
          blobClient.delete()
        }
      } catch {
        case e: FileNotFoundException =>
      }
    }
  }

  def listStatus(filename: String): Array[FileStatus] = retryTransientErrors {
    val (account, container, path) = getAccountContainerPath(filename)

    val blobContainerClient: BlobContainerClient = getContainerClient(account, container)
    val statList: ArrayBuffer[FileStatus] = ArrayBuffer()

    val prefix = dropTrailingSlash(path) + "/"
    // collect all children of this directory (blobs and subdirectories)
    val prefixMatches = blobContainerClient.listBlobsByHierarchy(prefix)

    prefixMatches.forEach(blobItem => {
      val blobFileName = s"hail-az://$account/$container/${blobItem.getName}"
      statList += fileStatus(blobFileName)
    })
    statList.toArray
  }

  def glob(filename: String): Array[FileStatus] = retryTransientErrors {
    var (account, container, path) = getAccountContainerPath(filename)
    path = dropTrailingSlash(path)

    globWithPrefix(prefix = s"hail-az://$account/$container", path = path)
  }

  def fileStatus(account: String, container: String, path: String): FileStatus = retryTransientErrors {
    if (path == "") {
      return new BlobStorageFileStatus(s"hail-az://$account/$container", null, 0, true)
    }

    val blobClient: BlobClient = getBlobClient(account, container, path)
    val blobContainerClient: BlobContainerClient = getContainerClient(account, container)

    val prefix = dropTrailingSlash(path) + "/"
    val options: ListBlobsOptions = new ListBlobsOptions().setPrefix(prefix)
    val prefixMatches = blobContainerClient.listBlobs(options, null)
    val isDir = prefixMatches.iterator().hasNext

    val filename = dropTrailingSlash(s"hail-az://$account/$container/$path")
    if (!isDir && !blobClient.exists()) {
      throw new FileNotFoundException(s"File not found: $filename")
    }

    if (isDir) {
      new BlobStorageFileStatus(path = filename, null, 0, isDir = true)
    } else {
      val blobProperties: BlobProperties = blobClient.getProperties
      AzureStorageFileStatus(blobProperties, path = filename, isDir = false)
    }
  }

  def fileStatus(filename: String): FileStatus = retryTransientErrors {
    val (account, container, path) = getAccountContainerPath(filename)
    fileStatus(account, container, path)
  }

  def makeQualified(filename: String): String = {
    if (!filename.startsWith("hail-az://"))
      throw new IllegalArgumentException(s"Invalid path, expected hail-az://accountName/containerName/blobPath: $filename")
    filename
  }

  def asCacheable(): CacheableAzureStorageFS = new CacheableAzureStorageFS(credentialsJSON, null)
}

class CacheableAzureStorageFS(
  credentialsJSON: Option[String],
  @transient val sessionID: String
) extends AzureStorageFS(credentialsJSON) with ServiceCacheableFS {
}
