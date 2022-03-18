package is.hail.io.fs

import com.azure.core.credential.TokenCredential
import com.azure.core.exception.ResourceNotFoundException
import com.azure.core.http.rest.PagedIterable
import com.azure.identity.{DefaultAzureCredential, DefaultAzureCredentialBuilder}
import com.azure.storage.blob.models.{BlobItem, BlobProperties, BlobRange, BlobStorageException, ListBlobsOptions}
import com.azure.storage.blob.specialized.AppendBlobClient
import com.azure.storage.blob.{BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientBuilder}
import is.hail.io.fs.AzureStorageFS.{containsWildcard, dropTrailingSlash, getAccountContainerPath}
import org.apache.log4j.Logger

import java.net.URI
import is.hail.utils.fatal

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

object AzureStorageFileStatus {
  def apply(filename: String): Unit = {
  }
}

class AzureStorageFileStatus(path: String, modificationTime: java.lang.Long, size: Long, isDir: Boolean) extends FileStatus {
  def getPath: String = path

  def getModificationTime: java.lang.Long = modificationTime

  def getLen: Long = size

  def isDirectory: Boolean = isDir

  def isFile: Boolean = !isDir

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


class AzureStorageFS extends FS {
  val credential: DefaultAzureCredential = new DefaultAzureCredentialBuilder().build()
  @transient private lazy val storageClientCache = {
    new AzureBlobServiceClientCache(credential)
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

    println(s"opening $filename")

    val is: SeekableInputStream = new InputStream with Seekable {
      private[this] var closed: Boolean = false
      private[this] var client: BlobClient = blobClient
      private[this] var pos: Long = 0
      private[this] var eof: Boolean = false

      private[this] val bb: ByteBuffer = ByteBuffer.allocate(64 * 1024)
      bb.limit(0)

      // TODO: in general, how should we be handling closed behavior?
//      override def read(): Int = {
//        if (eof || closed) {
//          println("EOF OR CLOSED")
//          return -1
//        }
//
//        val outputStream: ByteArrayOutputStream = new ByteArrayOutputStream()
//        try {
//          client.downloadStreamWithResponse(
//            outputStream, new BlobRange(pos, 1),
//            null, null, false, Duration.ofMinutes(1), null)
//          pos += 1
//        }
//        catch {
//          case e: BlobStorageException => if (e.getStatusCode == 416) {
//            println("caught an EOF blob storage exception")
//            eof = true
//            return -1
//          }
//          case e: ResourceNotFoundException => throw new FileNotFoundException("Couldn't find that file.")
//        }
//
//        val arrayWithNextByte: Array[Byte] = outputStream.toByteArray
//        println(s"pos = $pos")
////        println(s"next byte: ${arrayWithNextByte.mkString("Array(", ", ", ")")}")
//        arrayWithNextByte(0).toInt & 0xff
////        bb.get().toInt & 0xff
//      }

      def fill(): Unit = {
        bb.clear()

        // read some bytes
        val outputStream: ByteArrayOutputStream = new ByteArrayOutputStream()
        val blobProperties = blobClient.getProperties
        val blobSize = blobProperties.getBlobSize

        println(s"${blobSize}")
        println(s"$pos")
        println(s"bb remaining: ${bb.remaining()}")
        val count = Math.min(blobSize - pos, bb.remaining() - 1)
        if (count <= 0) {
          eof = true
          return
        }

        // calculate the minimum of the bytes remaining in the buffer and the bytes remaining
        // to be read in the blob
        client.downloadStreamWithResponse(
                      outputStream, new BlobRange(pos, count),
                      null, null, false, Duration.ofMinutes(1), null)
                    pos += count

        bb.put(outputStream.toByteArray)

//        var n = 0
//        while (n == 0) {
////          n = reader.read(bb)
//          // TODO: reader.read copies the contents of the blob into bb and returns number of bytes read (returning -1 if EOF)
//          if (n == -1) {
//            eof = true
//            return
//          }
//        }
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

  // TODO: need to use AppendBlobClient to be able to append data to existing blob
  def createNoCompression(filename: String): PositionedDataOutputStream = {
    println(s"Filename: $filename")
    val appendClient: AppendBlobClient = getBlobClient(filename).getAppendBlobClient
    if (!appendClient.exists()) appendClient.create()

    println("created the thing")

    val os: PositionedOutputStream = new OutputStream with Positioned {
      private[this] val bb: ByteBuffer = ByteBuffer.allocate(64 * 20)
      private[this] var pos: Long = 0
      private[this] var closed: Boolean = false
      private[this] var client: AppendBlobClient = appendClient

      override def write(i: Int): Unit = {
        if (!closed) {
          if (bb.remaining() == 0) {
            println("flushing")
            flush()
          }
          bb.put(i.toByte)
          println(s"put ${i.toByte} into buffer")
          pos += 1
        }
      }

//      override def write(bytes: Array[Byte], off: Int, len: Int): Unit = {
//        var i = off
//        var remaining = len
//        while (remaining > 0) {
//          if (bb.remaining() == 0) {
//            println("flushing in the bigger write")
//            flush()
//          }
//          val toTransfer = math.min(bb.remaining(), remaining)
//          bb.put(bytes, i, toTransfer)
//          i += toTransfer
//          remaining -= toTransfer
//          pos += toTransfer
//        }
//      }

      override def flush(): Unit = {
        if (!closed) {
          bb.flip()

          if (bb.limit() > 0) {
            println(s"position: ${bb.position()}")
            println(s"limit: ${bb.limit()}")
            println(s"remaining: ${bb.remaining()}")
            var toWrite: mutable.MutableList[Byte] = mutable.MutableList()
            var i = 0
            while (i < bb.remaining()) {
              toWrite += bb.get(i)
              i += 1
            }
            println(s"array being appended to append client: ${toWrite.toArray.mkString("Array(", ", ", ")")}")
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
        // TODO: for loop over these results and delete them all
        val listBlobsOptions = new ListBlobsOptions().setPrefix(dropTrailingSlash(path) + "/")
        val directoryContents = blobContainerClient.listBlobs(listBlobsOptions, Duration.ofMinutes(1))
        directoryContents.forEach(blobItem => {
          // figure out what blobItem.getName does so I can use it to call fileStatus
          println(s"blobItem: ${blobItem.getName}")

          val blobFileName = s"hail-az://$account/$container/${blobItem.getName}"
          this.getBlobClient(blobFileName).delete()
        })

        // TODO: this check may be unnecessary
        // delete the initial blob itself if it wasn't already deleted
        if (blobClient.exists()) blobClient.delete()
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
    println(s"filename: $filename")

    var (account, container, path) = getAccountContainerPath(filename)

    val blobContainerClient: BlobContainerClient = getContainerClient(filename)

    val statList: mutable.MutableList[FileStatus] = mutable.MutableList()
    var blobNameSet: Set[String] = Set()

    println(s"path: ${dropTrailingSlash(path) + "/"}")
    val prefix = dropTrailingSlash(path) + "/"
    val listBlobsOptions = new ListBlobsOptions().setPrefix(prefix)
    val directoryContents = blobContainerClient.listBlobs(listBlobsOptions, Duration.ofMinutes(1))
    directoryContents.forEach(blobItem => {
      // figure out what blobItem.getName does so I can use it to call fileStatus
      println(s"blobItem: ${blobItem.getName}")

      // find the first slash
      val indexOfFirstSlashAfterPrefix: Int = blobItem.getName.indexOf("/", prefix.length)
      var blobName = blobItem.getName
      if (indexOfFirstSlashAfterPrefix != -1) {
        println(s"FOUND A SLASH @ index $indexOfFirstSlashAfterPrefix")
        blobName = blobItem.getName.substring(0, indexOfFirstSlashAfterPrefix)
        println(s"blobName: $blobName")
      }

      println(blobNameSet.toArray.mkString("Array(", ", ", ")"))
      if (!blobNameSet.contains(blobName)) {
        blobNameSet = blobNameSet + blobName
        val blobFileName = s"hail-az://$account/$container/$blobName"
        println(s"blobFileName: $blobFileName")
        statList += fileStatus(blobFileName)
      }
    })

    println(blobNameSet.toArray.mkString("Array(", ", ", ")"))
    statList.toArray
  }

  def glob(filename: String): Array[FileStatus] = {
//    var (bucket, path) = getBucketPath(filename)
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
    println(filename)
    var (account, container, path) = getAccountContainerPath(filename)
    println(s"account: $account")
    println(s"container: $container")
    println(s"path: $path")

    if (path == "") {
      return new AzureStorageFileStatus(s"hail-az://$account/$container", null, 0, true)
    }

    val blobClient: BlobClient = getBlobClient(filename)
    val blobContainerClient: BlobContainerClient = getContainerClient(filename)

    var numSubBlobs: Int = 0
//    val listBlobsOptions = new ListBlobsOptions().setPrefix(dropTrailingSlash(path) + "/")
//    val directoryContents = blobContainerClient.listBlobs(listBlobsOptions, Duration.ofMinutes(1))
    val directoryContents = blobContainerClient.listBlobsByHierarchy(dropTrailingSlash(path) + "/")
    directoryContents.forEach(_ => {
      numSubBlobs += 1
    })
    println(s"numSubBlobs = ${numSubBlobs}")

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
