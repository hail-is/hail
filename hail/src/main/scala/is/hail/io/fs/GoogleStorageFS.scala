package is.hail.io.fs

import java.io.{ByteArrayInputStream, FileNotFoundException, InputStream, OutputStream}
import java.net.URI
import java.nio.ByteBuffer
import java.nio.file.FileSystems
import org.apache.log4j.{LogManager, Logger}
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.{ReadChannel, WriteChannel}
import com.google.cloud.storage.Storage.BlobListOption
import com.google.cloud.storage.{Blob, BlobId, BlobInfo, Storage, StorageOptions}
import is.hail.services.retryTransientErrors

import scala.collection.JavaConverters._
import scala.collection.mutable

object GoogleStorageFS {
  private val log = Logger.getLogger(getClass.getName())

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

  def getBucketPath(filename: String): (String, String) = {
    val uri = new URI(filename).normalize()

    val scheme = uri.getScheme
    assert(scheme != null && scheme == "gs", (uri.getScheme, filename))

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
  def apply(blob: Blob): BlobStorageFileStatus = {
    val isDir = blob.isDirectory

    val name = GoogleStorageFS.dropTrailingSlash(blob.getName)

    new BlobStorageFileStatus(
      s"gs://${ blob.getBucket }/$name",
      if (isDir)
        null
      else
        blob.getUpdateTime,
      blob.getSize,
      isDir)
  }
}

class GoogleStorageFS(val serviceAccountKey: Option[String] = None) extends FS {
  import GoogleStorageFS._

  @transient private lazy val storage: Storage = serviceAccountKey match {
    case None =>
      log.info("Initializing google storage client from latent credentials")
      StorageOptions.newBuilder()
        .build()
        .getService
    case Some(keyData) =>
      log.info("Initializing google storage client from service account key")
      StorageOptions.newBuilder()
        .setCredentials(
          ServiceAccountCredentials.fromStream(new ByteArrayInputStream(keyData.getBytes)))
        .build()
        .getService
  }

  def asCacheable(): CacheableGoogleStorageFS = new CacheableGoogleStorageFS(serviceAccountKey, null)

  def asCacheable(sessionID: String): CacheableGoogleStorageFS = new CacheableGoogleStorageFS(serviceAccountKey, sessionID)

  def openNoCompression(filename: String): SeekableDataInputStream = {
    val (bucket, path) = getBucketPath(filename)

    val is: SeekableInputStream = new FSSeekableInputStream {
      private[this] val reader: ReadChannel = storage.reader(bucket, path)

      override def close(): Unit = {
        if (!closed) {
          reader.close()
          closed = true
        }
      }

      override def fill(): Unit = {
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

      def seek(newPos: Long): Unit = {
        bb.clear()
        bb.limit(0)
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

    val os: PositionedOutputStream = new FSPositionedOutputStream {
        private[this] val write: WriteChannel = storage.writer(blobInfo)

        override def flush(): Unit = {
          bb.flip()

          while (bb.remaining() > 0)
            write.write(bb)

          bb.clear()
        }

        override def close(): Unit = {
          if (!closed) {
            flush()
            retryTransientErrors {
              write.close()
            }
            closed = true
          }
        }
    }

    new WrappedPositionedDataOutputStream(os)
  }

  def delete(filename: String, recursive: Boolean): Unit = {
    val (bucket, path) = getBucketPath(filename)
    if (recursive) {
      val it = retryTransientErrors {
        storage.list(bucket, BlobListOption.prefix(path))
          .getValues.iterator.asScala
      }
      while (it.hasNext) {
        storage.delete(it.next().getBlobId)
      }
    } else {
      storage.delete(bucket, path)
    }
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

  def listStatus(filename: String): Array[FileStatus] = {
    var (bucket, path) = getBucketPath(filename)
    if (!path.endsWith("/"))
      path = path + "/"

    val blobs = retryTransientErrors {
      storage.list(bucket, BlobListOption.prefix(path), BlobListOption.currentDirectory())
    }

    blobs.getValues.iterator.asScala
      .filter(b => b.getName != path) // elide directory markers created by Hadoop
      .map(b => GoogleStorageFileStatus(b))
      .toArray
  }

  def fileStatus(filename: String): FileStatus = {
    var (bucket, path) = getBucketPath(filename)
    path = dropTrailingSlash(path)

    if (path == "")
      return new BlobStorageFileStatus(s"gs://$bucket", null, 0, true)

    val blobs = retryTransientErrors {
      storage.list(bucket, BlobListOption.prefix(path), BlobListOption.currentDirectory())
    }

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

  def deleteOnExit(filename: String): Unit =
    FSUtil.runOnExit(() => delete(filename, recursive = false))
}

class CacheableGoogleStorageFS(serviceAccountKey: Option[String], @transient val sessionID: String) extends GoogleStorageFS(serviceAccountKey) with ServiceCacheableFS {
}
