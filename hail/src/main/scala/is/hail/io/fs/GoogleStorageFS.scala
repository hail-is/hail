package is.hail.io.fs

import java.io.{ByteArrayInputStream, FileNotFoundException, InputStream, OutputStream}
import java.net.URI
import java.nio.ByteBuffer
import java.nio.file.FileSystems

import is.hail.utils._
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.{ReadChannel, WriteChannel}
import com.google.cloud.storage.Storage.BlobListOption
import com.google.cloud.storage.{Blob, BlobId, BlobInfo, Storage, StorageOptions}
import org.apache.commons.io.FilenameUtils
import org.apache.hadoop

import scala.collection.JavaConverters._
import scala.collection.mutable

object GoogleStorageFS {
  def containsWildcard(path: String): Boolean = {
    var i = 0
    while (i < path.length) {
      var c = path(i)
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

    return false
  }

  def getBucketPath(filename: String): (String, String) = {
    val uri = new URI(filename)

    val scheme = uri.getScheme
    assert(scheme != null && scheme == "gs", uri.getScheme)

    val bucket = uri.getHost
    assert(bucket != null)

    var path = uri.getPath
    if (path.nonEmpty && path.head == '/')
      path = path.drop(1)

    (bucket, path)
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

object GoogleStorageFileStatus {
  def apply(blob: Blob): GoogleStorageFileStatus = {
    val isDir = blob.isDirectory

    val name = GoogleStorageFS.dropTrailingSlash(blob.getName)

    new GoogleStorageFileStatus(
      s"gs://${ blob.getBucket }/$name",
      if (isDir)
        null
      else
        blob.getUpdateTime,
      blob.getSize,
      isDir)
  }
}

class GoogleStorageFileStatus(path: String, modificationTime: java.lang.Long, size: Long, isDir: Boolean) extends FileStatus {
  def getPath: String = path

  def getModificationTime: java.lang.Long = modificationTime

  def getLen: Long = size

  def isDirectory: Boolean = isDir

  def isFile: Boolean = !isDir

  def getOwner: String = null
}

class GoogleStorageFS(serviceAccountKey: String) extends FS {
  import GoogleStorageFS._

  var codecNames: IndexedSeq[String] = FastIndexedSeq(
    "is.hail.io.compress.BGzipCodec",
    "is.hail.io.compress.BGzipCodecTbi",
    "org.apache.hadoop.io.compress.GzipCodec")

  @transient private var codecs: IndexedSeq[hadoop.io.compress.CompressionCodec] = _
  
  def createCodecs(): Unit = {
    if (codecs != null)
      return

    codecs = codecNames.map { codecName =>
      val codecClass = Class.forName(codecName)
      codecClass.newInstance().asInstanceOf[hadoop.io.compress.CompressionCodec]
    }
  }

  @transient private lazy val storage: Storage = {
    StorageOptions.newBuilder()
      .setCredentials(
        ServiceAccountCredentials.fromStream(new ByteArrayInputStream(serviceAccountKey.getBytes)))
      .build()
      .getService
  }

  def getCodecs(): IndexedSeq[String] = codecNames

  def setCodecs(newCodecs: IndexedSeq[String]): Unit = {
    codecNames = newCodecs
    codecs = null
  }

  def getCodec(filename: String): hadoop.io.compress.CompressionCodec = {
    if (codecs == null)
      createCodecs()

    val ext = "." + FilenameUtils.getExtension(filename)
    codecs.foreach { codec =>
      if (codec.getDefaultExtension == ext)
        return codec
    }

    null
  }

  def openNoCompression(filename: String): SeekableDataInputStream = {
    val (bucket, path) = getBucketPath(filename)

    val is: SeekableInputStream = new InputStream with Seekable {
      val bb: ByteBuffer = ByteBuffer.allocate(64 * 1024)
      bb.limit(0)

      val reader: ReadChannel = storage.reader(bucket, path)
      var pos: Long = 0
      var eof: Boolean = false

      def fill(): Unit = {
        bb.clear()

        // read some bytes
        var n = 0
        while (n == 0) {
          n = reader.read(bb)
          if (n == -1) {
            eof = true
            return
          }
        }
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
        bb.get()
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

      def getPosition: Long = pos

      def seek(newPos: Long): Unit = {
        bb.clear()
        reader.seek(newPos)
        pos = newPos
      }
    }

    new WrappedSeekableDataInputStream(is)
  }

  def createNoCompression(filename: String): PositionedDataOutputStream = {
    val (bucket, path) = getBucketPath(filename)

    val blobId = BlobId.of(bucket, path)
    val blobInfo = BlobInfo.newBuilder(blobId)
      .build()

    val os: PositionedOutputStream = new OutputStream with Positioned {
      val bb: ByteBuffer = ByteBuffer.allocate(64 * 1024)
      var pos: Long = 0
      val write: WriteChannel = storage.writer(blobInfo)

      override def flush(): Unit = {
        bb.flip()

        while (bb.remaining() > 0)
          write.write(bb)

        bb.clear()
      }

      override def write(i: Int): Unit = {
        if (bb.remaining() == 0)
          flush()
        bb.put(i.toByte)
        pos += 1
      }

      override def write(bytes: Array[Byte], off: Int, len: Int): Unit = {
        var i = off
        var remaining = len
        while (remaining > 0) {
          if (bb.remaining() == 0)
            flush()
          val toTransfer = math.min(bb.remaining(), remaining)
          bb.put(bytes, i, toTransfer)
          i += toTransfer
          remaining -= toTransfer
          pos += toTransfer
        }
      }

      override def close(): Unit = {
        flush()
        write.close()
      }

      def getPosition: Long = pos
    }

    new WrappedPositionedDataOutputStream(os)
  }

  def mkDir(dirname: String): Unit = ()

  def delete(filename: String, recursive: Boolean): Unit = {
    val (bucket, path) = getBucketPath(filename)
    if (recursive) {
      val it = storage.list(bucket, BlobListOption.prefix(path))
        .getValues.iterator.asScala
      while (it.hasNext) {
        storage.delete(it.next().getBlobId)
      }
    } else
      storage.delete(bucket, path)
  }

  def glob(filename: String): Array[FileStatus] = {
    var (bucket, path) = getBucketPath(filename)
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

    f(s"gs://$bucket", null, 0)
    ab.toArray
  }

  def globAll(filenames: Iterable[String]): Array[String] =
    globAllStatuses(filenames).map(_.getPath)

  def globAllStatuses(filenames: Iterable[String]): Array[FileStatus] = filenames.flatMap(glob).toArray

  def listStatus(filename: String): Array[FileStatus] = {
    var (bucket, path) = getBucketPath(filename)
    if (!path.endsWith("/"))
      path = path + "/"

    val blobs = storage.list(bucket, BlobListOption.prefix(path), BlobListOption.currentDirectory())

    blobs.getValues.iterator.asScala
      .map(b => GoogleStorageFileStatus(b))
      .toArray
  }

  def fileStatus(filename: String): FileStatus = {
    var (bucket, path) = getBucketPath(filename)
    path = dropTrailingSlash(path)

    if (path == "")
      return new GoogleStorageFileStatus(s"gs://$bucket", null, 0, true)

    val blobs = storage.list(bucket, BlobListOption.prefix(path), BlobListOption.currentDirectory())

    val it = blobs.getValues.iterator.asScala
    while (it.hasNext) {
      val b = it.next()
      var name = b.getName
      while (name.endsWith("/"))
        name = name.dropRight(1)
      if (name == path)
        return GoogleStorageFileStatus(b)
    }

    throw new FileNotFoundException(filename)
  }

  def makeQualified(filename: String): String = {
    // gs is cannot be a default scheme
    assert(filename.startsWith("gs://"))
    filename
  }

  def deleteOnExit(filename: String): Unit = {
    Runtime.getRuntime.addShutdownHook(
      new Thread(
        new Runnable {
          def run(): Unit = delete(filename, recursive = false)
        }))
  }
}
