package is.hail.services

import is.hail.services.oauth2.AzureCloudCredentials.EnvVars.AzureApplicationCredentials
import is.hail.services.oauth2.GoogleCloudCredentials.EnvVars.GoogleApplicationCredentials
import is.hail.shadedazure.com.azure.core.credential.{TokenCredential, TokenRequestContext}
import is.hail.shadedazure.com.azure.identity.{
  ClientSecretCredentialBuilder, DefaultAzureCredentialBuilder,
}
import is.hail.utils.{defaultJSONFormats, using}

import scala.collection.JavaConverters._

import java.io.Serializable
import java.nio.file.{Files, Path}

import com.google.auth.oauth2.{GoogleCredentials, ServiceAccountCredentials}
import org.json4s.Formats
import org.json4s.jackson.JsonMethods

object oauth2 {

  sealed trait CloudCredentials extends Product with Serializable {
    def accessToken(scopes: IndexedSeq[String]): String
  }

  def CloudCredentials(credentialsPath: Path, env: Map[String, String] = sys.env)
    : CloudCredentials =
    env.get("HAIL_CLOUD") match {
      case Some("gcp") => GoogleCloudCredentials(Some(credentialsPath))
      case Some("azure") => AzureCloudCredentials(Some(credentialsPath))
      case Some(cloud) => throw new IllegalArgumentException(s"Unknown cloud: '$cloud'")
      case None => throw new IllegalArgumentException(s"HAIL_CLOUD must be set.")
    }

  def CloudScopes(env: Map[String, String] = sys.env): Array[String] =
    env.get("HAIL_CLOUD") match {
      case Some("gcp") =>
        Array(
          "https://www.googleapis.com/auth/userinfo.profile",
          "https://www.googleapis.com/auth/userinfo.email",
          "openid",
        )
      case Some("azure") =>
        sys.env.get("HAIL_AZURE_OAUTH_SCOPE").toArray
      case Some(cloud) =>
        throw new IllegalArgumentException(s"Unknown cloud: '$cloud'.")
      case None =>
        throw new IllegalArgumentException(s"HAIL_CLOUD must be set.")
    }

  case class GoogleCloudCredentials(value: GoogleCredentials) extends CloudCredentials {
    override def accessToken(scopes: IndexedSeq[String]): String = {
      value.refreshIfExpired()
      value.createScoped(scopes.asJava).getAccessToken.getTokenValue
    }
  }

  object GoogleCloudCredentials {
    object EnvVars {
      val GoogleApplicationCredentials = "GOOGLE_APPLICATION_CREDENTIALS"
    }

    def apply(keyPath: Option[Path], env: Map[String, String] = sys.env): GoogleCloudCredentials =
      GoogleCloudCredentials(
        keyPath.orElse(env.get(GoogleApplicationCredentials).map(Path.of(_))) match {
          case Some(path) => using(Files.newInputStream(path))(ServiceAccountCredentials.fromStream)
          case None => GoogleCredentials.getApplicationDefault
        }
      )
  }

  sealed trait AzureCloudCredentials extends CloudCredentials {
    def value: TokenCredential

    override def accessToken(scopes: IndexedSeq[String]): String =
      value.getTokenSync(new TokenRequestContext().setScopes(scopes.asJava)).getToken
  }

  object AzureCloudCredentials {
    object EnvVars {
      val AzureApplicationCredentials = "AZURE_APPLICATION_CREDENTIALS"
    }

    def apply(keyPath: Option[Path], env: Map[String, String] = sys.env): AzureCloudCredentials =
      keyPath.orElse(env.get(AzureApplicationCredentials).map(Path.of(_))) match {
        case Some(path) => AzureClientSecretCredentials(path)
        case None => AzureDefaultCredentials
      }
  }

  private case object AzureDefaultCredentials extends AzureCloudCredentials {
    @transient override lazy val value: TokenCredential =
      new DefaultAzureCredentialBuilder().build()
  }

  private case class AzureClientSecretCredentials(path: Path) extends AzureCloudCredentials {
    @transient override lazy val value: TokenCredential =
      using(Files.newInputStream(path)) { is =>
        implicit val fmts: Formats = defaultJSONFormats
        val kvs = JsonMethods.parse(is)
        new ClientSecretCredentialBuilder()
          .clientId((kvs \ "appId").extract[String])
          .clientSecret((kvs \ "password").extract[String])
          .tenantId((kvs \ "tenant").extract[String])
          .build()
      }
  }
}
