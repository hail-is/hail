package is.hail.io.fs

import java.io.{ByteArrayInputStream, FileNotFoundException}
import java.net.URI
import java.nio.ByteBuffer
import java.nio.file.FileSystems
import org.apache.log4j.{LogManager, Logger}
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.{ReadChannel, WriteChannel}
import com.google.cloud.storage.Storage.BlobListOption
import com.google.cloud.storage.{Blob, BlobId, BlobInfo, Storage, StorageOptions}
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.services.retryTransientErrors

import scala.collection.JavaConverters._
import scala.collection.mutable

object GoogleStorageFS {
  private val log = Logger.getLogger(getClass.getName())

  def getBucketPath(filename: String): (String, String) = {
    val uri = new URI(filename).normalize()

    val scheme = uri.getScheme
    assert(scheme != null && scheme == "gs", (uri.getScheme, filename))

    val bucket = uri.getAuthority
    assert(bucket != null, (filename, uri.toString(), uri.getScheme, uri.getAuthority, uri.getRawAuthority(), uri.getUserInfo()))

    var path = uri.getPath
    if (path.nonEmpty && path.head == '/')
      path = path.drop(1)

    (bucket, path)
  }
}

object GoogleStorageFileStatus {
  def apply(blob: Blob): BlobStorageFileStatus = {
    val isDir = blob.isDirectory

    val name = dropTrailingSlash(blob.getName)

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

  @transient private lazy val storage: Storage = {
    val transportOptions = StorageOptions.getDefaultHttpTransportOptions().toBuilder()
      .setConnectTimeout(5000)
      .setReadTimeout(5000)
      .build()
    serviceAccountKey match {
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
          .setTransportOptions(transportOptions)
          .build()
          .getService
    }
  }

  def asCacheable(): CacheableGoogleStorageFS = new CacheableGoogleStorageFS(serviceAccountKey, null)

  def asCacheable(sessionID: String): CacheableGoogleStorageFS = new CacheableGoogleStorageFS(serviceAccountKey, sessionID)

  def openNoCompression(filename: String): SeekableDataInputStream = retryTransientErrors {
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

  def createNoCompression(filename: String): PositionedDataOutputStream = retryTransientErrors {
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

  def mkDir(dirname: String): Unit = ()

  def delete(filename: String, recursive: Boolean): Unit = retryTransientErrors {
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
      // Storage.delete is idempotent. it returns a Boolean which is false if the file did not exist
      storage.delete(bucket, path)
    }
  }

  def glob(filename: String): Array[FileStatus] = retryTransientErrors {
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

  def listStatus(filename: String): Array[FileStatus] = retryTransientErrors {
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

  def fileStatus(filename: String): FileStatus = retryTransientErrors {
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
}

class CacheableGoogleStorageFS(serviceAccountKey: Option[String], @transient val sessionID: String) extends GoogleStorageFS(serviceAccountKey) with ServiceCacheableFS {
}
