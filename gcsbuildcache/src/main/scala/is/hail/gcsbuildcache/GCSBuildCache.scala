package is.hail.gcsbuildcache

import org.gradle.api.{Plugin, Action, GradleException}
import org.gradle.api.logging.Logging
import org.gradle.api.initialization.Settings
import org.gradle.caching.configuration.AbstractBuildCache
import org.gradle.caching.{BuildCacheEntryWriter, BuildCacheEntryReader, BuildCacheService, BuildCacheServiceFactory, BuildCacheKey}
import com.google.auth.oauth2.GoogleCredentials
import com.google.cloud.storage.{Bucket, StorageException, StorageOptions, BlobId, BlobInfo}
import com.google.cloud.storage.Storage.{BlobWriteOption, BlobTargetOption}
import com.google.common.io.FileBackedOutputStream
import java.nio.channels.Channels
import java.time.OffsetDateTime
import java.time.temporal.ChronoUnit


// Gradle requirs a no-argument constructor. A one-argument Scala constructor with default arguments
// *does not work*.
class GCSBuildCacheConfiguration() extends AbstractBuildCache {
  // Gradle requires literal setters and getters. Do not try to expose a mutable field directly to
  // gradle, you will get errors like "Could not set unknown property 'gcsURL'"
  private[this] var _gcsURL: String = null
  def setGcsURL(gcsURL: String) = _gcsURL = gcsURL
  def getGcsURL: String = _gcsURL
  def getIsPush: Boolean = isPush()
  def setIsPush(isPush: Boolean) = setPush(isPush)
  // Gradle cannot directly modify isEnabled/setEnabled because they do not follow the get/set
  // pattern.
  def getEnabled: Boolean = isEnabled()
  override def setEnabled(enabled: Boolean) = super.setEnabled(enabled)
}

class GCSBuildCache(
  conf: GCSBuildCacheConfiguration
) extends BuildCacheService {
  private[this] val log = Logging.getLogger(classOf[GCSBuildCache])
  private[this] val storage = StorageOptions.newBuilder()
    .setCredentials(GoogleCredentials.getApplicationDefault())
    .build()
    .getService

  override def store(key: BuildCacheKey, writer: BuildCacheEntryWriter): Unit = {
    try {
      val path = conf.getGcsURL + "/" + key.hashCode
      val blobinfo = BlobInfo.newBuilder(BlobId.fromGsUtilUri(path)).build()
      val value = new FileBackedOutputStream(8 * 1024 * 1024, true)
      writer.writeTo(value)
      log.lifecycle(s"storing cache $key ${key.hashCode} $path")
      val is = value.asByteSource().openBufferedStream()
      try {
        val onlyCreateIfNotExists = BlobWriteOption.doesNotExist()
        storage.create(blobinfo, is, onlyCreateIfNotExists)
        log.lifecycle(s"cache stored $key ${key.hashCode} $path")
      } catch {
        case exc: StorageException =>
          throw new RuntimeException(s"$key ${key.hashCode} could not be stored in cache at $path", exc)
      } finally {
        is.close()
      }
    } catch {
      case exc: Exception =>
        // Gradle will silence all exceptions and quietly disable the build cache, so we loudly
        // print the exception.
        exc.printStackTrace()
        throw exc
    }
  }

  override def load(key: BuildCacheKey, reader: BuildCacheEntryReader): Boolean = {
    val path = conf.getGcsURL + "/" + key.hashCode
    val blobid = BlobId.fromGsUtilUri(path)
    log.lifecycle(s"checking cache key $key ${key.hashCode} $path")
    try {
      val blob = storage.get(blobid)
      if (blob == null) {
        return false
      }
      reader.readFrom(Channels.newInputStream(blob.reader()))
      log.lifecycle(s"cache hit $key ${key.hashCode} $path")

      if (blob.getCreateTimeOffsetDateTime().until(OffsetDateTime.now(), ChronoUnit.SECONDS) > 24 * 60 * 60) {
        val blobinfo = BlobInfo.newBuilder(blobid).build()
        log.lifecycle(s"will refreshing cache $key ${key.hashCode} $path")
        val onlyUpdateIfNoOneHasBeatenMeToIt = BlobTargetOption.generationMatch(blob.getGeneration)
        storage.create(blobinfo, blob.getContent(), onlyUpdateIfNoOneHasBeatenMeToIt)
        log.lifecycle(s"refreshed cache $key ${key.hashCode} $path")
      }
      return true
    } catch {
      case exc: Exception =>
        // Gradle will silence all exceptions and quietly disable the build cache, so we loudly
        // print the exception.
        exc.printStackTrace()
        throw exc
    }
  }

  override def close(): Unit = {}
}

class GCSBuildCacheServiceFactory extends BuildCacheServiceFactory[GCSBuildCacheConfiguration] {
  override def createBuildCacheService(
    conf: GCSBuildCacheConfiguration,
    describer: BuildCacheServiceFactory.Describer
  ): BuildCacheService = {
    if (conf.getGcsURL == null || conf.getGcsURL == "") {
      throw new GradleException("gcsURL must be set.")
    }

    describer
      .`type`("Google Cloud Storage")
      .config("gcsURL", conf.getGcsURL)
      .config("isPush", conf.getIsPush.toString)
      .config("enabled", conf.getEnabled.toString)

    return new GCSBuildCache(conf)
  }
}

class GCSBuildCachePlugin() extends Plugin[Settings] {
  override def apply(settings: Settings): Unit = {
    settings.getBuildCache.registerBuildCacheService(classOf[GCSBuildCacheConfiguration], classOf[GCSBuildCacheServiceFactory])
  }
}
