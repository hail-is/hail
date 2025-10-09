package is.hail.services

import is.hail.services.oauth2.AzureCloudCredentials.AzureTokenRefreshMinutes
import is.hail.services.oauth2.AzureCloudCredentials.EnvVars.AzureApplicationCredentials
import is.hail.services.oauth2.GoogleCloudCredentials.EnvVars.GoogleApplicationCredentials
import is.hail.shadedazure.com.azure.core.credential.{
  AccessToken, TokenCredential, TokenRequestContext,
}
import is.hail.shadedazure.com.azure.identity.{
  ClientSecretCredentialBuilder, DefaultAzureCredentialBuilder,
}
import is.hail.utils.{defaultJSONFormats, using}

import scala.jdk.CollectionConverters._

import java.io.Serializable
import java.nio.file.{Files, Path}
import java.time.OffsetDateTime

import com.google.auth.oauth2.{GoogleCredentials, ServiceAccountCredentials}
import org.json4s.Formats
import org.json4s.jackson.JsonMethods

object oauth2 {

  sealed trait CloudCredentials extends Product with Serializable {
    def accessToken: String
  }

  def CloudCredentials(
    keyPath: Path,
    scopes: IndexedSeq[String],
    env: Map[String, String] = sys.env,
  ): CloudCredentials =
    env.get("HAIL_CLOUD") match {
      case Some("gcp") => GoogleCloudCredentials(Some(keyPath), scopes, env)
      case Some("azure") => AzureCloudCredentials(Some(keyPath), scopes, env)
      case Some(cloud) => throw new IllegalArgumentException(s"Unknown cloud: '$cloud'")
      case None => throw new IllegalArgumentException(s"HAIL_CLOUD must be set.")
    }

  case class GoogleCloudCredentials(value: GoogleCredentials) extends CloudCredentials {
    override def accessToken: String = {
      value.refreshIfExpired()
      value.getAccessToken.getTokenValue
    }
  }

  object GoogleCloudCredentials {
    object EnvVars {
      val GoogleApplicationCredentials = "GOOGLE_APPLICATION_CREDENTIALS"
    }

    def apply(keyPath: Option[Path], scopes: IndexedSeq[String], env: Map[String, String] = sys.env)
      : GoogleCloudCredentials =
      GoogleCloudCredentials {
        val creds: GoogleCredentials =
          keyPath.orElse(env.get(GoogleApplicationCredentials).map(Path.of(_))) match {
            case Some(path) =>
              using(Files.newInputStream(path))(ServiceAccountCredentials.fromStream)
            case None =>
              GoogleCredentials.getApplicationDefault
          }

        creds.createScoped(scopes: _*)
      }
  }

  sealed trait AzureCloudCredentials extends CloudCredentials {

    def value: TokenCredential
    def scopes: IndexedSeq[String]

    @transient private[this] var token: AccessToken = _

    override def accessToken: String = {
      refreshIfRequired()
      token.getToken
    }

    private[this] def refreshIfRequired(): Unit =
      if (!isExpired) token.getToken
      else synchronized {
        if (isExpired) {
          token = value.getTokenSync(new TokenRequestContext().setScopes(scopes.asJava))
        }

        token.getToken: Unit
      }

    private[this] def isExpired: Boolean =
      token == null || OffsetDateTime.now.plusMinutes(AzureTokenRefreshMinutes.toLong).isBefore(
        token.getExpiresAt
      )
  }

  object AzureCloudCredentials {
    object EnvVars {
      val AzureApplicationCredentials = "AZURE_APPLICATION_CREDENTIALS"
    }

    private[AzureCloudCredentials] val AzureTokenRefreshMinutes = 5

    def apply(keyPath: Option[Path], scopes: IndexedSeq[String], env: Map[String, String] = sys.env)
      : AzureCloudCredentials =
      keyPath.orElse(env.get(AzureApplicationCredentials).map(Path.of(_))) match {
        case Some(path) => AzureClientSecretCredentials(path, scopes)
        case None => AzureDefaultCredentials(scopes)
      }
  }

  private case class AzureDefaultCredentials(scopes: IndexedSeq[String])
      extends AzureCloudCredentials {
    @transient override lazy val value: TokenCredential =
      new DefaultAzureCredentialBuilder().build()
  }

  private case class AzureClientSecretCredentials(path: Path, scopes: IndexedSeq[String])
      extends AzureCloudCredentials {
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
