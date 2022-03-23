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

    // FIXME: Should the path include the first slash or not?
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

class AzureStorageFileStatus(path: String, modificationTime: java.lang.Long, size: Long, isDir: Boolean) extends FileStatus {
  def getPath: String = path

  def getModificationTime: java.lang.Long = modificationTime

  def getLen: Long = size

  def isDirectory: Boolean = isDir

  def isFile: Boolean = !isDir

  def isSymlink: Boolean = false

  def getOwner: String = null
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

//      {
//        "appId": "6ba2e9fc-e889-41e9-8409-2f950ca793eb",
//        "appObjectId": "7c724a87-15e0-407b-94d8-64e4d2b3009d",
//        "displayName": "haildev-test",
//        "objectId": "5505f0bf-aeef-49a3-9a1c-062987b123f9",
//        "password": "Hev7Q~FqhgMd0X8xsuzKQmHSssYEFbglhr.oM",
//        "tenant": "d6c9f2ea-d3bb-4ca9-8b14-231bac999aa6"
//      }

      val appId = kvs("appId").asInstanceOf[json4s.JString].s
      val appObjectId = kvs("appObjectId").asInstanceOf[json4s.JString].s
      val displayName = kvs("displayName").asInstanceOf[json4s.JString].s
      val objectId = kvs("objectId").asInstanceOf[json4s.JString].s
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
//    throw new NotImplementedError()
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

    val is: SeekableInputStream = new InputStream with Seekable {
      private[this] var closed: Boolean = false
      private[this] var client: BlobClient = blobClient
      private[this] var pos: Long = 0
      private[this] var eof: Boolean = false

      private[this] val bb: ByteBuffer = ByteBuffer.allocate(64 * 1024)
      bb.limit(0)

      def fill(): Unit = {
        bb.clear()

        // read some bytes
        val outputStream: ByteArrayOutputStream = new ByteArrayOutputStream()
        val blobProperties = blobClient.getProperties
        val blobSize = blobProperties.getBlobSize

        val count = Math.min(blobSize - pos, bb.remaining() - 1)
        if (count <= 0) {
          eof = true
          return
        }

        // calculate the minimum of the bytes remaining in the buffer and the bytes remaining
        // to be read in the blob
//        client.downloadStreamWithResponse(
//                      outputStream, new BlobRange(pos, count),
//                      null, null, false, Duration.ofMinutes(1), null)
//                    pos += count
//
//        bb.put(outputStream.toByteArray)


        client.downloadWithResponse(
                      outputStream, new BlobRange(pos, count),
                      null, null, false, Duration.ofMinutes(1), null)
                    pos += count

        bb.put(outputStream.toByteArray)

        bb.flip()

        assert(bb.position() == 0 && bb.remaining() > 0)
      }

      override def read(): Int = {
        if (eof)
          return -1

        if (bb.remaining() == 0) {
          fill()
          if (eof)
            return -1
        }

        pos += 1
        bb.get().toInt & 0xff
      }

      override def read(bytes: Array[Byte], off: Int, len: Int): Int = {
        if (eof)
          return -1

        if (bb.remaining() == 0) {
          fill()
          if (eof)
            return -1
        }

        val toTransfer = math.min(len, bb.remaining())
        bb.get(bytes, off, toTransfer)
        pos += toTransfer
        toTransfer
      }

      override def close(): Unit = {
        if (!closed) {
          closed = true
        }
      }

      override def seek(newPos: Long): Unit = {
        if (!closed) {
          // TODO: should blobSize be a property of this class instead of being defined here?
          val blobSize = blobClient.getProperties.getBlobSize
          if (newPos < 0 || newPos > blobSize) throw new IndexOutOfBoundsException(s"Cannot seek to position $newPos: Out of bounds [0, $blobSize]")
          pos = newPos
        }
      }

      override def getPosition: Long = pos
    }

    new WrappedSeekableDataInputStream(is)
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = {
    val appendClient: AppendBlobClient = getBlobClient(filename).getAppendBlobClient
    if (!appendClient.exists()) appendClient.create()

    val os: PositionedOutputStream = new OutputStream with Positioned {
      private[this] val bb: ByteBuffer = ByteBuffer.allocate(64 * 20)
      private[this] var pos: Long = 0
      private[this] var closed: Boolean = false
      private[this] var client: AppendBlobClient = appendClient

      override def write(i: Int): Unit = {
        if (!closed) {
          if (bb.remaining() == 0) {
            flush()
          }
          bb.put(i.toByte)
          pos += 1
        }
      }

      override def flush(): Unit = {
        if (!closed) {
          bb.flip()

          if (bb.limit() > 0) {
            var toWrite: mutable.MutableList[Byte] = mutable.MutableList()
            var i = 0
            while (i < bb.remaining()) {
              toWrite += bb.get(i)
              i += 1
            }
            appendClient.appendBlock(new ByteArrayInputStream(toWrite.toArray), toWrite.length)
          }

          bb.clear()
        }
      }

      override def close(): Unit = {
        if (!closed) {
          flush()
          closed = true
        }
      }

      override def getPosition: Long = pos
    }

    new WrappedPositionedDataOutputStream(os)
  }

  override def mkDir(dirname: String): Unit = ()

  override def delete(filename: String, recursive: Boolean): Unit = {
    // check if the file exists
    if (this.exists(filename)) {
      // if it does, get the corresponding BlobClient
      val (account, container, path) = getAccountContainerPath(filename)
      val blobClient: BlobClient = this.getBlobClient(filename)

      val fileStatus = this.fileStatus(filename)

      if (recursive) {
        // if recursive, create an iterator over all files in this "directory" (files that match its prefix)
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
            this.delete(blobFileName, recursive)
          }
        }
        deleteRecursively()

        // TODO: this check may be unnecessary
        // delete the initial blob itself if it wasn't already deleted
        if (fileStatus.isFile && blobClient.exists()) blobClient.delete()
      }
      else {
        // TODO: should this throw an exception? If so, what kind?
        // if not, just delete the file using the BlobClient
          if (fileStatus.isDirectory) {
            throw new Exception("Cannot non-recursively delete a directory")
          }
          blobClient.delete()
      }
    }
  }

  override def listStatus(filename: String): Array[FileStatus] = {
    var (account, container, path) = getAccountContainerPath(filename)

    val blobContainerClient: BlobContainerClient = getContainerClient(filename)

    val statList: mutable.MutableList[FileStatus] = mutable.MutableList()
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

        // find the first slash
        val indexOfFirstSlashAfterPrefix: Int = blobItem.getName.indexOf("/", prefix.length)
        var blobName = blobItem.getName
        if (indexOfFirstSlashAfterPrefix != -1) {
          blobName = blobItem.getName.substring(0, indexOfFirstSlashAfterPrefix)
        }

        if (!blobNameSet.contains(blobName)) {
          blobNameSet = blobNameSet + blobName
          val blobFileName = s"hail-az://$account/$container/$blobName"
          statList += fileStatus(blobFileName)
        }
      }
    }
    populateStatList()

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

  override def globAll(filenames: Iterable[String]): Array[String] =
    globAllStatuses(filenames).map(_.getPath)

  override def globAllStatuses(filenames: Iterable[String]): Array[FileStatus] = filenames.flatMap(glob).toArray

  override def fileStatus(filename: String): FileStatus = {
    var (account, container, path) = getAccountContainerPath(filename)

    if (path == "") {
      return new AzureStorageFileStatus(s"hail-az://$account/$container", null, 0, true)
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
      new AzureStorageFileStatus(dropTrailingSlash(filename), -1, -1, isDir)
    }
    else {
      val blobProperties: BlobProperties = blobClient.getProperties
      val modificationTime = blobProperties.getLastModified.toEpochSecond
      val size = blobProperties.getBlobSize

      new AzureStorageFileStatus(dropTrailingSlash(filename), modificationTime, size, isDir)
    }
  }

  override def makeQualified(path: String): String = ???

  override def deleteOnExit(path: String): Unit = ???
}
