package is.hail.gcsbuildcache

import org.gradle.api.{Plugin, Action, GradleException}
import org.gradle.api.logging.Logging
import org.gradle.api.initialization.Settings
import org.gradle.caching.configuration.AbstractBuildCache
import org.gradle.caching.{BuildCacheEntryWriter, BuildCacheEntryReader, BuildCacheService, BuildCacheServiceFactory, BuildCacheKey}
import com.google.auth.oauth2.GoogleCredentials
import com.google.cloud.storage.{Bucket, StorageException, StorageOptions, BlobId, BlobInfo}
import com.google.common.io.FileBackedOutputStream
import java.nio.channels.Channels
import java.time.OffsetDateTime
import java.time.temporal.ChronoUnit


class GCSBuildCacheConfiguration(
  var gcsURL: String = "",
) extends AbstractBuildCache

class GCSBuildCache(
  conf: GCSBuildCacheConfiguration
) extends BuildCacheService {
  private[this] val log = Logging.getLogger(classOf[GCSBuildCache])
  private[this] val storage = StorageOptions.newBuilder()
    .setCredentials(GoogleCredentials.getApplicationDefault())
    .build()
    .getService

  override def store(key: BuildCacheKey, writer: BuildCacheEntryWriter): Unit = {
    val path = conf.gcsURL + "/" + key.hashCode.toString
    val blobinfo = BlobInfo.newBuilder(BlobId.fromGsUtilUri(path)).build()
    val value = new FileBackedOutputStream(8 * 1024 * 1024, true)
    writer.writeTo(value)
    val is = value.asByteSource().openBufferedStream()
    try {
      storage.create(blobinfo, is)
      System.out.println(s"cache stored $key")
    } catch {
      case exc: StorageException =>
        throw new RuntimeException(s"$key could not be stored in cache at $path", exc)
    } finally {
      is.close()
    }
  }

  override def load(key: BuildCacheKey, reader: BuildCacheEntryReader): Boolean = {
    System.out.println(s"checking cache key $key")
    val path = conf.gcsURL + "/" + key.hashCode.toString
    val blobid = BlobId.fromGsUtilUri(path)
    try {
      val blob = storage.get(blobid)
      reader.readFrom(Channels.newInputStream(blob.reader()))
      System.out.println(s"cache hit $key")

      if (blob.getCreateTimeOffsetDateTime().until(OffsetDateTime.now(), ChronoUnit.SECONDS) > 24 * 60 * 60) {
        val blobinfo = BlobInfo.newBuilder(blobid).build()
        System.out.println(s"will refreshing cache $key")
        storage.create(blobinfo, blob.getContent())
        System.out.println(s"refreshed cache $key")
      }
      return true
    } catch {
      case exc: StorageException if exc.getCode == 404 =>
        System.out.println(s"cache miss $key")
        return false
      case exc: StorageException =>
        throw new RuntimeException(s"$key could not be loaded from cache at $path", exc)
    }

    return false
  }

  override def close(): Unit = {}
}

class GCSBuildCacheServiceFactory extends BuildCacheServiceFactory[GCSBuildCacheConfiguration] {
  override def createBuildCacheService(
    conf: GCSBuildCacheConfiguration,
    describer: BuildCacheServiceFactory.Describer
  ): BuildCacheService = {
    if (conf.gcsURL == null || conf.gcsURL == "") {
      throw new GradleException("gcsURL must be set.")
    }

    describer
      .`type`("Google Cloud Storage")
      .config("gcsURL", conf.gcsURL)

    return new GCSBuildCache(conf)
  }
}

class GCSBuildCachePlugin() extends Plugin[Settings] {
  override def apply(settings: Settings): Unit = {
    settings.getBuildCache.registerBuildCacheService(classOf[GCSBuildCacheConfiguration], classOf[GCSBuildCacheServiceFactory])
  }
}
