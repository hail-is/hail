package is.hail.io.fs

import is.hail.services.oauth2.AzureCloudCredentials
import is.hail.shadedazure.com.azure.storage.blob.BlobServiceClient
import is.hail.utils._

import scala.collection.mutable

import org.apache.http.client.methods.HttpPost
import org.apache.http.client.utils.URIBuilder
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

object TerraAzureStorageFS {
  private val TEN_MINUTES_IN_MS = 10 * 60 * 1000

  val RequiredOAuthScopes: Array[String] =
    Array("https://management.azure.com/.default")
}

class TerraAzureStorageFS(credential: AzureCloudCredentials) extends AzureStorageFS(credential) {
  import TerraAzureStorageFS.TEN_MINUTES_IN_MS

  private[this] val httpClient = HttpClients.custom().build()
  private[this] val sasTokenCache = mutable.Map[String, (URL, Long)]()

  private[this] val workspaceManagerUrl = sys.env("WORKSPACE_MANAGER_URL")
  private[this] val workspaceId = sys.env("WORKSPACE_ID")
  private[this] val containerResourceId = sys.env("WORKSPACE_STORAGE_CONTAINER_ID")
  private[this] val storageContainerUrl = parseUrl(sys.env("WORKSPACE_STORAGE_CONTAINER_URL"))

  override def getServiceClient(url: URL): BlobServiceClient =
    if (blobInWorkspaceStorageContainer(url)) {
      super.getServiceClient(getTerraSasToken(url))
    } else {
      super.getServiceClient(url)
    }

  def getTerraSasToken(url: URL): URL = {
    sasTokenCache.get(url.base) match {
      case Some((sasTokenUrl, expiration))
          if expiration > System.currentTimeMillis + TEN_MINUTES_IN_MS => sasTokenUrl
      case None =>
        val (sasTokenUrl, expiration) = createTerraSasToken()
        sasTokenCache += (url.base -> (sasTokenUrl -> expiration))
        sasTokenUrl
    }
  }

  private def blobInWorkspaceStorageContainer(url: URL): Boolean =
    storageContainerUrl.account == url.account && storageContainerUrl.container == url.container

  private def createTerraSasToken(): (URL, Long) = {
    implicit val formats: Formats = DefaultFormats

    val url =
      s"$workspaceManagerUrl/api/workspaces/v1/$workspaceId/resources/controlled/azure/storageContainer/$containerResourceId/getSasToken"
    val req = new HttpPost(url)
    req.addHeader("Authorization", s"Bearer ${credential.accessToken}")

    val tenHoursInSeconds = 10 * 3600
    val expiration = System.currentTimeMillis() + tenHoursInSeconds * 1000
    val uri = new URIBuilder(req.getURI())
      .addParameter("sasPermissions", "racwdl")
      .addParameter("sasExpirationDuration", tenHoursInSeconds.toString)
      .build()
    req.setURI(uri)

    val sasTokenUrl = using(httpClient.execute(req)) { resp =>
      val json = JsonMethods.parse(new String(EntityUtils.toString(resp.getEntity)))
      logger.info(s"Created sas token client for $containerResourceId")
      (json \ "url").extract[String]
    }

    (parseUrl(sasTokenUrl), expiration)
  }
}
