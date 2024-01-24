package is.hail.io.fs

import scala.collection.mutable

import is.hail.shadedazure.com.azure.core.credential.{AzureSasCredential, TokenCredential, TokenRequestContext}
import is.hail.shadedazure.com.azure.identity.{DefaultAzureCredential, DefaultAzureCredentialBuilder}
import is.hail.utils._

import org.apache.http.client.utils.URIBuilder
import org.apache.http.HttpEntity
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.{ContentType, StringEntity}
import org.apache.http.impl.client.HttpClients
import org.apache.http.params.HttpParams
import org.apache.http.util.EntityUtils
import org.apache.log4j.Logger
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods


object TerraAzureStorageFS {
  private val log = Logger.getLogger(getClass.getName)
  private val TEN_MINUTES_IN_MS = 10 * 60 * 1000
}

class TerraAzureStorageFS extends FS {
  type URL = AzureStorageFSURL
  import TerraAzureStorageFS.{log, TEN_MINUTES_IN_MS}

  private[this] val azureFS = new AzureStorageFS()
  private[this] val credential = new DefaultAzureCredentialBuilder().build()
  private[this] val httpClient = HttpClients.custom().build()
  private[this] val sasTokenCache = mutable.Map[String, (String, Long)]()

  private[this] val workspaceManagerUrl = sys.env("WORKSPACE_MANAGER_URL")
  private[this] val workspaceId = sys.env("WORKSPACE_ID")
  private[this] val containerResourceId = sys.env("WORKSPACE_STORAGE_CONTAINER_ID")
  private[this] val storageContainerUrl = sys.env("WORKSPACE_STORAGE_CONTAINER_URL")

  def parseUrl(filename: String): AzureStorageFSURL = {
    val urlStr =
      if (filename.startsWith(storageContainerUrl)) {
        sasTokenCache.get(filename) match {
          case Some((sasTokenUrl, expiration)) if expiration > System.currentTimeMillis + TEN_MINUTES_IN_MS => sasTokenUrl
          case None => {
            val (sasTokenUrl, expiration) = getTerraSasToken(filename)
            sasTokenCache += (filename -> (sasTokenUrl -> expiration))
            sasTokenUrl
          }
        }
      } else {
        filename
      }

    azureFS.parseUrl(urlStr)
  }

  private def getTerraSasToken(filename: String): (String, Long) = {
    implicit val formats: Formats = DefaultFormats

    val context = new TokenRequestContext()
    context.addScopes("https://management.azure.com/.default")
    val token = credential.getToken(context).block().getToken()

    val url = s"${workspaceManagerUrl}/api/workspaces/v1/${workspaceId}/resources/controlled/azure/storageContainer/${containerResourceId}/getSasToken"
    val req = new HttpPost(url)
    req.addHeader("Authorization", s"Bearer $token")

    val tenHoursInSeconds = 10 * 3600
    val expiration = System.currentTimeMillis() + tenHoursInSeconds * 1000
    val uri = new URIBuilder(req.getURI())
      .addParameter("sasPermissions", "racwdl")
      .addParameter("sasExpirationDuration", tenHoursInSeconds.toString)
      .build()
    req.setURI(uri)

    val sasTokenUrl = using(httpClient.execute(req)) { resp =>
      val json = JsonMethods.parse(new String(EntityUtils.toString(resp.getEntity)))
      log.info(s"Created sas token client for $containerResourceId")
      (json \ "url").extract[String]
    }

    (sasTokenUrl, expiration)
  }

  def validUrl(filename: String): Boolean = azureFS.validUrl(filename)
  def getConfiguration(): Unit = azureFS.getConfiguration
  def setConfiguration(config: Any): Unit = azureFS.setConfiguration(config)
  def openNoCompression(url: URL): SeekableDataInputStream = azureFS.openNoCompression(url)
  def createNoCompression(url: URL): PositionedDataOutputStream = azureFS.createNoCompression(url)
  def delete(url: URL, recursive: Boolean): Unit = azureFS.delete(url, recursive)
  def glob(url: URL): Array[FileListEntry] = azureFS.glob(url)
  def fileListEntry(url: URL): FileListEntry = azureFS.fileListEntry(url)
  def fileStatus(url: URL): FileStatus = azureFS.fileStatus(url)
  def makeQualified(filename: String): String = azureFS.makeQualified(filename)
  def eTag(url: URL): Some[String] = azureFS.eTag(url)
  def listDirectory(url: URL): Array[FileListEntry] = azureFS.listDirectory(url)
  def urlAddPathComponent(url: URL, component: String): URL = url.addPathComponent(component)
}
