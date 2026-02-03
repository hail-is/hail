package is.hail.io.fs

import is.hail.HailFeatureFlags
import is.hail.services.oauth2.{AzureCloudCredentials, GoogleCloudCredentials}
import is.hail.utils.{FastSeq, SerializableHadoopConfiguration}

import java.io.Serializable
import java.nio.file.Path

import org.apache.hadoop.conf.Configuration

class RouterFSURL(val fs: FS, _url: FSURL[_]) extends FSURL[RouterFSURL] {
  def url: fs.URL = _url.asInstanceOf[fs.URL]

  override def path: String = _url.path

  override def /(component: String): RouterFSURL =
    new RouterFSURL(fs, url / component)

  override def toString: String = _url.toString
}

case class CloudStorageFSConfig(
  azure: Option[AzureStorageFSConfig] = None,
  google: Option[GoogleStorageFSConfig] = None,
) extends Serializable

object CloudStorageFSConfig {
  def fromFlagsAndEnv(
    credentialsFile: Option[Path],
    flags: HailFeatureFlags,
    env: Map[String, String] = sys.env,
  ): CloudStorageFSConfig = {
    env.get("HAIL_CLOUD") match {
      case Some("azure") =>
        CloudStorageFSConfig(azure = Some(AzureStorageFSConfig(credentialsFile)))
      case Some("gcp") | None =>
        val rpConf = RequesterPaysConfig.fromFlags(flags)
        CloudStorageFSConfig(google = Some(GoogleStorageFSConfig(credentialsFile, rpConf)))
      case _ =>
        CloudStorageFSConfig()
    }
  }
}

object RouterFS {

  def buildRoutes(cloudConfig: CloudStorageFSConfig, env: Map[String, String] = sys.env): FS =
    new RouterFS(
      IndexedSeq.concat(
        cloudConfig.google.map { case GoogleStorageFSConfig(path, rpConfig) =>
          new GoogleStorageFS(
            GoogleCloudCredentials(path).scoped(GoogleStorageFS.RequiredOAuthScopes),
            rpConfig,
          )
        },
        cloudConfig.azure.map { case AzureStorageFSConfig(path) =>
          if (env.contains("HAIL_TERRA"))
            new TerraAzureStorageFS(
              AzureCloudCredentials(path, env)
                .scoped(TerraAzureStorageFS.RequiredOAuthScopes)
            )
          else
            new AzureStorageFS(
              AzureCloudCredentials(path, env)
                .scoped(AzureStorageFS.RequiredOAuthScopes)
            )
        },
        FastSeq(new HadoopFS(new SerializableHadoopConfiguration(new Configuration()))),
      )
    )
}

class RouterFS(fss: IndexedSeq[FS]) extends FS {
  type URL = RouterFSURL

  def lookupFS(path: String): FS =
    fss.find(_.validUrl(path)) match {
      case Some(fs) => fs
      case None => throw new IllegalArgumentException(s"Unsupported URI: $path")
    }

  override def parseUrl(filename: String): URL = {
    val fs = lookupFS(filename)
    new RouterFSURL(fs, fs.parseUrl(filename))
  }

  override def validUrl(filename: String): Boolean =
    fss.exists(_.validUrl(filename))

  override def createNoCompression(url: RouterFSURL): PositionedDataOutputStream =
    url.fs.createNoCompression(url.url)

  override def openNoCompression(url: URL): SeekableDataInputStream =
    url.fs.openNoCompression(url.url)

  override def mkDir(url: URL): Unit = url.fs.mkDir(url.url)

  override def delete(url: URL, recursive: Boolean) = url.fs.delete(url.url, recursive)

  override def listDirectory(url: URL): Array[FileListEntry] = url.fs.listDirectory(url.url)

  override def glob(url: URL): Array[FileListEntry] = url.fs.glob(url.url)

  override def fileStatus(url: URL): FileStatus = url.fs.fileStatus(url.url)

  override def fileListEntry(url: URL): FileListEntry = url.fs.fileListEntry(url.url)

  override def eTag(url: URL): Option[String] = url.fs.eTag(url.url)

  override def makeQualified(path: String): String = lookupFS(path).makeQualified(path)

  override def getConfiguration(): Any = fss.map(_.getConfiguration())

  override def setConfiguration(config: Any): Unit =
    fss.zip(config.asInstanceOf[IndexedSeq[_]]).foreach { case (fs: FS, config: Any) =>
      fs.setConfiguration(config)
    }
}
