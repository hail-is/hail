package is.hail.io.fs

import com.azure.core.credential.TokenCredential
import com.azure.identity.{ClientSecretCredential, ClientSecretCredentialBuilder, DefaultAzureCredential, DefaultAzureCredentialBuilder}
import com.azure.storage.blob.models.{BlobItem, BlobProperties, BlobRange, ListBlobsOptions}
import com.azure.storage.blob.specialized.AppendBlobClient
import com.azure.storage.blob.{BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientBuilder}
import is.hail.io.fs.AzureStorageFS.{getAccountContainerPath, schemes}
import is.hail.io.fs.FSUtil.dropTrailingSlash
import org.apache.log4j.Logger

import java.net.URI
import is.hail.utils.fatal
import org.json4s
import org.json4s.jackson.JsonMethods

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileNotFoundException, InputStream, OutputStream}
import java.nio.ByteBuffer
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

    val account = uri.getHost
    if (account == null) {
      throw new IllegalArgumentException(s"Invalid path, expected hail-az://accountName/containerName/blobPath: $filename")
    }

    uri.getPath

    val (container, path) = pathRegex.findFirstMatchIn(uri.getPath) match {
      case Some(filenameMatch) =>
        val container = filenameMatch.group(1)
        val path = filenameMatch.group(2)
        if (path != "") {
          assert(path.startsWith("/"))
          (container, path.substring(1))
        }
        else {
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
  @transient private lazy val clientBuilder: BlobServiceClientBuilder = new BlobServiceClientBuilder()
  @transient private lazy val clients: mutable.Map[String, BlobServiceClient] = mutable.Map()

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
  @transient private lazy val serviceClientCache = credentialsJSON match {
    case None =>
      val credential: DefaultAzureCredential = new DefaultAzureCredentialBuilder().build()
      new AzureBlobServiceClientCache(credential)
    case Some(keyData) =>
      val kvs = JsonMethods.parse(keyData) match {
        case json4s.JObject(values) => values.toMap
      }

      val appId = kvs("appId").asInstanceOf[json4s.JString].s
      val password = kvs("password").asInstanceOf[json4s.JString].s
      val tenant = kvs("tenant").asInstanceOf[json4s.JString].s

      val clientSecretCredential: ClientSecretCredential = new ClientSecretCredentialBuilder()
        .clientId(appId)
        .clientSecret(password)
        .tenantId(tenant)
        .build();
      new AzureBlobServiceClientCache(clientSecretCredential)
  }

  def getBlobServiceClient(account: String): BlobServiceClient = {
    serviceClientCache.getServiceClient(account)
  }

  def getBlobClient(filename: String): BlobClient = {
    val (account, container, path) = getAccountContainerPath(filename)
    getBlobServiceClient(account).getBlobContainerClient(container).getBlobClient(path)
  }

  def getContainerClient(filename: String): BlobContainerClient = {
    val (account, container, _) = getAccountContainerPath(filename)
    getBlobServiceClient(account).getBlobContainerClient(container)
  }

  def openNoCompression(filename: String): SeekableDataInputStream = {
    val blobClient: BlobClient = getBlobClient(filename)
    val blobProperties = blobClient.getProperties
    val blobSize = blobProperties.getBlobSize

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] val client: BlobClient = blobClient

      def fill(): Unit = {
        bb.clear()

        val outputStreamToBuffer: OutputStream = (i: Int) => {
          bb.put(i.toByte)
        }

        // calculate the minimum of the bytes remaining in the buffer and the bytes remaining
        // to be read in the blob
        val count = Math.min(blobSize - pos, bb.remaining())
        if (count <= 0) {
          eof = true
          return
        }

        client.downloadStreamWithResponse(
          outputStreamToBuffer, new BlobRange(pos, count),
          null, null, false, Duration.ofMinutes(1), null)
        pos += count
        bb.flip()

        assert(bb.position() == 0 && bb.remaining() > 0)
      }

      override def seek(newPos: Long): Unit = {
        bb.clear()
        bb.limit(0)
        pos = newPos
      }
    }

    new WrappedSeekableDataInputStream(is)
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = {
    val appendClient: AppendBlobClient = getBlobClient(filename).getAppendBlobClient
    if (!appendClient.exists()) appendClient.create()

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

  def delete(filename: String, recursive: Boolean): Unit = {
    val (account, container, path) = getAccountContainerPath(filename)
    val blobClient: BlobClient = this.getBlobClient(filename)

    val fileStatus = this.fileStatus(filename)

    if (recursive) {
      val blobContainerClient = this.getContainerClient(filename)

      val options = new ListBlobsOptions()
      options.setPrefix(dropTrailingSlash(path) + "/")
      val directoryContents = blobContainerClient.listBlobs(options, Duration.ofMinutes(1))

      directoryContents.forEach(blobItem => {
//        println(s"blobItem: ${blobItem.getName}")
        val blobFileName = s"hail-az://$account/$container/${blobItem.getName}"
        this.getBlobClient(blobFileName).delete()
      })

      if (fileStatus.isFile) blobClient.delete()
    }
    else {
        blobClient.delete()
    }
  }

  def listStatus(filename: String): Array[FileStatus] = {
    val (account, container, path) = getAccountContainerPath(filename)

    val blobContainerClient: BlobContainerClient = getContainerClient(filename)
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

  def glob(filename: String): Array[FileStatus] = {
    var (account, container, path) = getAccountContainerPath(filename)
    path = dropTrailingSlash(path)

    val components =
      if (path == "")
        Array.empty[String]
      else
        path.split("/")

    val javaFS = FileSystems.getDefault

    val ab = new mutable.ArrayBuffer[FileStatus]()
    def f(prefix: String, fs: FileStatus, i: Int): Unit = {
      assert(!prefix.endsWith("/"), prefix)

      if (i == components.length) {
        var t = fs
        if (t == null) {
          try {
            t = fileStatus(prefix)
          } catch {
            case _: FileNotFoundException =>
          }
        }
        if (t != null)
          ab += t
      }

      if (i < components.length) {
        val c = components(i)
        if (containsWildcard(c)) {
          val m = javaFS.getPathMatcher(s"glob:$c")
          for (cfs <- listStatus(prefix)) {
            val p = dropTrailingSlash(cfs.getPath)
            val d = p.drop(prefix.length + 1)
            if (m.matches(javaFS.getPath(d))) {
              f(p, cfs, i + 1)
            }
          }
        } else
          f(s"$prefix/$c", null, i + 1)
      }
    }

    f(s"hail-az://$account/$container", null, 0)
    ab.toArray
  }

  def fileStatus(filename: String): FileStatus = {
    val (account, container, path) = getAccountContainerPath(filename)

    if (path == "") {
      return new BlobStorageFileStatus(s"hail-az://$account/$container", null, 0, true)
    }

    val blobClient: BlobClient = getBlobClient(filename)
    val blobContainerClient: BlobContainerClient = getContainerClient(filename)

    val prefix = dropTrailingSlash(path) + "/"
    val options: ListBlobsOptions = new ListBlobsOptions().setPrefix(prefix)
    val prefixMatches = blobContainerClient.listBlobs(options, null)
    val isDir = prefixMatches.iterator().hasNext

    if (!isDir && !blobClient.exists()) throw new FileNotFoundException("file isn't found")

    if (isDir) {
      new BlobStorageFileStatus(dropTrailingSlash(filename), -1, -1, isDir = true)
    }
    else {
      val blobProperties: BlobProperties = blobClient.getProperties
      AzureStorageFileStatus(blobProperties, path = dropTrailingSlash(filename), isDir = false)
    }
  }

  def makeQualified(filename: String): String = {
    val uri = new URI(filename).normalize()

    val scheme = uri.getScheme
    if (scheme == null || !schemes.contains(scheme)) {
      throw new IllegalArgumentException(s"invalid scheme, expected hail-az: $scheme")
    }
    filename
  }
}
