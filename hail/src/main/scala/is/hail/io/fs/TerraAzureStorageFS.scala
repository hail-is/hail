package is.hail.io.fs

import scala.collection.mutable

import is.hail.shadedazure.com.azure.core.credential.TokenRequestContext
import is.hail.shadedazure.com.azure.identity.{
  DefaultAzureCredential,
  DefaultAzureCredentialBuilder,
}
import is.hail.utils._

import org.apache.http.client.utils.URIBuilder
import org.apache.http.client.methods.HttpPost
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils
import org.apache.log4j.Logger
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods


object TerraAzureStorageFS {
  private val log = Logger.getLogger(getClass.getName)
  private val TEN_MINUTES_IN_MS = 10 * 60 * 1000
}

class TerraAzureStorageFS extends AzureStorageFS() {
  import TerraAzureStorageFS.{log, TEN_MINUTES_IN_MS}

  private[this] val httpClient = HttpClients.custom().build()
  private[this] val sasTokenCache = mutable.Map[String, (String, Long)]()

  private[this] val workspaceManagerUrl = sys.env("WORKSPACE_MANAGER_URL")
  private[this] val workspaceId = sys.env("WORKSPACE_ID")
  private[this] val containerResourceId = sys.env("WORKSPACE_STORAGE_CONTAINER_ID")
  private[this] val storageContainerUrl = sys.env("WORKSPACE_STORAGE_CONTAINER_URL")

  private[this] val credential: DefaultAzureCredential = new DefaultAzureCredentialBuilder().build()

  override def parseUrl(filename: String): AzureStorageFSURL = {
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

    parseUrl(urlStr)
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
}
