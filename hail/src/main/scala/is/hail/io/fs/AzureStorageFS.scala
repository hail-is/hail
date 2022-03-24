package is.hail.io.fs

import com.azure.core.credential.TokenCredential
import com.azure.identity.{ClientSecretCredential, ClientSecretCredentialBuilder, DefaultAzureCredential, DefaultAzureCredentialBuilder}
import com.azure.storage.blob.models.{BlobItem, BlobProperties, BlobRange}
import com.azure.storage.blob.specialized.AppendBlobClient
import com.azure.storage.blob.{BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientBuilder}
import is.hail.io.fs.AzureStorageFS.{containsWildcard, dropTrailingSlash, getAccountContainerPath}
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


object AzureStorageFS {
  private val pathRegex = "/([^/]+)/([^/]+)/(.*)".r
  private val rootRegex = "/([^/]+)/([^/]+)".r

  private val log = Logger.getLogger(getClass.getName)

  val schemes: Array[String] = Array("hail-az", "wasb")

  def containsWildcard(path: String): Boolean = {
    var i = 0
    while (i < path.length) {
      val c = path(i)
      if (c == '\\') {
        i += 1
        if (i < path.length)
          i += 1
        else
          return false
      } else if (c == '*' || c == '{' || c == '?' || c == '[')
        return true

      i += 1
    }

    false
  }

  def getAccountContainerPath(filename: String): (String, String, String) = {
    val uri = new URI(filename).normalize()

    val scheme = uri.getScheme
    assert(scheme != null && schemes.contains(scheme), (uri.getScheme, filename))

    val account = uri.getHost
    assert(account != null)

    val (container, path) = pathRegex.findFirstMatchIn(filename) match {
      case Some(filenameMatch) =>
        (filenameMatch.group(2), filenameMatch.group(3))
      case None =>
        rootRegex.findFirstMatchIn(filename) match {
          case Some(rootMatch) =>
            (rootMatch.group(2), "")
          case None =>
            fatal(s"filename $filename is not in the correct format. hail-az://account/container/blobPath")
        }
    }

    (account, container, path)
  }

  def dropTrailingSlash(path: String): String = {
    if (path.isEmpty)
      return path

    if (path.last != '/')
      return path

    var i = path.length - 1
    while (i > 0 && path(i - 1) == '/')
      i -= 1
    path.substring(0, i)
  }
}

class AzureBlobServiceClientCache(credential: TokenCredential) {
  @transient private lazy val clientBuilder: BlobServiceClientBuilder = new BlobServiceClientBuilder()
  @transient private lazy val clients: Map[String, BlobServiceClient] = Map()

  def getServiceClient(account: String): BlobServiceClient = {
    clients.get(account) match {
      case Some(client) => client
      case None =>
        val blobServiceClient = clientBuilder
          .credential(credential)
          .endpoint(s"https://$account.blob.core.windows.net")
          .buildClient()
        clients + (account -> blobServiceClient)
        blobServiceClient
    }
  }
}


class AzureStorageFS(val serviceAccountKey: Option[String] = None) extends FS {
  @transient private lazy val storageClientCache = serviceAccountKey match {
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
    storageClientCache.getServiceClient(account)
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

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] var client: BlobClient = blobClient

      def fill(): Unit = {
        bb.clear()

        // read some bytes
        val outputStream: ByteArrayOutputStream = new ByteArrayOutputStream()
        val blobProperties = client.getProperties
        val blobSize = blobProperties.getBlobSize

        val count = Math.min(blobSize - pos, bb.remaining() - 1)
        if (count <= 0) {
          eof = true
          return
        }

        // calculate the minimum of the bytes remaining in the buffer and the bytes remaining
        // to be read in the blob
        client.downloadWithResponse(
          outputStream, new BlobRange(pos, count),
          null, null, false, Duration.ofMinutes(1), null)
        pos += count
        bb.put(outputStream.toByteArray)
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
          var toWrite: mutable.MutableList[Byte] = mutable.MutableList()
          var i = 0
          while (i < bb.remaining()) {
            toWrite += bb.get(i)
            i += 1
          }
          client.appendBlock(new ByteArrayInputStream(toWrite.toArray), toWrite.length)
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
    if (this.exists(filename)) {
      val (account, container, path) = getAccountContainerPath(filename)
      val blobClient: BlobClient = this.getBlobClient(filename)

      val fileStatus = this.fileStatus(filename)

      if (recursive) {
        val blobContainerClient = this.getContainerClient(filename)

        val directoryContents = blobContainerClient.listBlobsByHierarchy(dropTrailingSlash(path) + "/")
        var seenBlobItems: Set[String] = Set()
        val iter = directoryContents.iterator()

        def deleteRecursively(): Unit = {
          while (iter.hasNext) {
            val blobItem: BlobItem = iter.next
            if (seenBlobItems.contains(blobItem.getName)) return
            seenBlobItems += blobItem.getName
            val blobFileName = s"hail-az://$account/$container/${blobItem.getName}"
            // recursively delete all files/subfolders
            this.delete(blobFileName, recursive)
          }
        }
        deleteRecursively()

        if (fileStatus.isFile && blobClient.exists()) blobClient.delete()
      }
      else {
          if (fileStatus.isDirectory) {
            throw new IllegalArgumentException("Cannot non-recursively delete a directory")
          }
          if (blobClient.exists()) blobClient.delete()
      }
    }
  }

  def listStatus(filename: String): Array[FileStatus] = {
    val (account, container, path) = getAccountContainerPath(filename)
    val blobContainerClient: BlobContainerClient = getContainerClient(filename)

    val statusList: mutable.MutableList[FileStatus] = mutable.MutableList()
    var blobNameSet: Set[String] = Set()

    var seenBlobItems: Set[String] = Set()

    val prefix = dropTrailingSlash(path) + "/"
    val directoryContents = blobContainerClient.listBlobsByHierarchy(prefix)
    val iter = directoryContents.iterator()

    def populateStatList(): Unit = {
      while (iter.hasNext) {
        val blobItem: BlobItem = iter.next

        if (seenBlobItems.contains(blobItem.getName)) return
        seenBlobItems += blobItem.getName

        // find the first slash after the prefix
        val indexOfFirstSlashAfterPrefix: Int = blobItem.getName.indexOf("/", prefix.length)
        var blobName = blobItem.getName
        // if there is a slash after the prefix, then this blob is under a subdirectory
        if (indexOfFirstSlashAfterPrefix != -1) {
          blobName = blobItem.getName.substring(0, indexOfFirstSlashAfterPrefix)
        }

        if (!blobNameSet.contains(blobName)) {
          blobNameSet = blobNameSet + blobName
          val blobFileName = s"hail-az://$account/$container/$blobName"
          statusList += fileStatus(blobFileName)
        }
      }
    }
    populateStatList()

    statusList.toArray
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

    val directoryContents = blobContainerClient.listBlobsByHierarchy(dropTrailingSlash(path) + "/")
    var seenBlobItems: Set[String] = Set()
    var numSubBlobs: Int = 0
    val iter = directoryContents.iterator()
    def countSubBlobs(): Unit = {
      while (iter.hasNext) {
        val name = iter.next.getName
        if (seenBlobItems.contains(name)) return
        seenBlobItems += name
        numSubBlobs += 1
      }
    }
    countSubBlobs()

    if (numSubBlobs == 0 && !blobClient.exists()) throw new FileNotFoundException("file isn't found")

    val isDir = numSubBlobs > 0
    if (isDir) {
      new BlobStorageFileStatus(dropTrailingSlash(filename), -1, -1, isDir)
    }
    else {
      val blobProperties: BlobProperties = blobClient.getProperties
      val modificationTime = blobProperties.getLastModified.toEpochSecond
      val size = blobProperties.getBlobSize

      new BlobStorageFileStatus(dropTrailingSlash(filename), modificationTime, size, isDir)
    }
  }

  def makeQualified(filename: String): String = {
    assert(filename.startsWith("hail-az://"))
    filename
  }

  def deleteOnExit(filename: String): Unit =
    FSUtil.runOnExit(() => delete(filename, recursive = false))
}
