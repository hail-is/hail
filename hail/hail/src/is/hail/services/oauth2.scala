package is.hail.services

import is.hail.macros.void
import is.hail.services.oauth2.AzureCloudCredentials.AzureTokenRefreshMinutes
import is.hail.services.oauth2.AzureCloudCredentials.EnvVars.AzureApplicationCredentials
import is.hail.shadedazure.com.azure.core.credential.{
  AccessToken, TokenCredential, TokenRequestContext,
}
import is.hail.shadedazure.com.azure.identity.{
  ClientSecretCredentialBuilder, DefaultAzureCredentialBuilder,
}
import is.hail.utils.{jsonToBytes, using}

import scala.collection.JavaConverters._

import java.io.{ByteArrayInputStream, Serializable}
import java.nio.file.{Files, Path}
import java.time.OffsetDateTime

import com.google.auth.oauth2.GoogleCredentials
import org.json4s.{DefaultFormats, Formats, JValue}
import org.json4s.jackson.JsonMethods

object oauth2 {

  sealed trait CloudCredentials extends Product with Serializable {
    def accessToken: String
    def scoped(scopes: Array[String]): CloudCredentials
  }

  implicit lazy val fmts: Formats = DefaultFormats

  def HailCredentials(env: Map[String, String] = sys.env): Option[CloudCredentials] =
    for {
      config <-
        env
          .get("XDG_CONFIG_HOME")
          .map(Path.of(_))
          .orElse(env.get("HOME").map(Path.of(_, ".config")))

      identity = config.resolve("hail/identity.json").toFile
      if identity.exists()

      json <- JsonMethods.parseOpt(identity)
    } yield (json \ "idp").extract[String] match {
      case "Google" => GoogleCloudCredentials.fromJson(json \ "credentials")
      case "Microsoft" => AzureCloudCredentials.fromJson(json \ "credentials")
      case other => throw new IllegalArgumentException(s"Unknown identity provider: '$other'")
    }

  def CloudCredentials(keyPath: Option[Path], env: Map[String, String] = sys.env)
    : CloudCredentials =
    env.get("HAIL_CLOUD") match {
      case Some("gcp") => GoogleCloudCredentials(keyPath)
      case Some("azure") => AzureCloudCredentials(keyPath, env)
      case Some(cloud) => throw new IllegalArgumentException(s"Unknown cloud: '$cloud'")
      case _ => throw new IllegalStateException("'HAIL_CLOUD' must be set.")
    }

  case class GoogleCloudCredentials(value: GoogleCredentials) extends CloudCredentials {
    override def accessToken: String = {
      value.refreshIfExpired()
      value.getAccessToken.getTokenValue
    }

    override def scoped(scopes: Array[String]): GoogleCloudCredentials =
      GoogleCloudCredentials(value.createScoped(scopes: _*))
  }

  object GoogleCloudCredentials {
    def fromJson(jv: JValue): GoogleCloudCredentials =
      GoogleCloudCredentials {
        GoogleCredentials.fromStream(
          new ByteArrayInputStream(jsonToBytes(jv))
        )
      }

    def apply(keyPath: Option[Path]): GoogleCloudCredentials =
      GoogleCloudCredentials {
        keyPath match {
          case Some(path) =>
            using(Files.newInputStream(path))(GoogleCredentials.fromStream)
          case None =>
            GoogleCredentials.getApplicationDefault
        }
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

    override def scoped(scopes: Array[String]): AzureCloudCredentials

    private[this] def refreshIfRequired(): Unit =
      if (!isExpired) void(token.getToken)
      else synchronized {
        if (isExpired) {
          token = value.getTokenSync(new TokenRequestContext().setScopes(scopes.asJava))
        }

        void(token.getToken)
      }

    private[this] def isExpired: Boolean =
      token == null || OffsetDateTime.now.plusMinutes(AzureTokenRefreshMinutes).isBefore(
        token.getExpiresAt
      )
  }

  object AzureCloudCredentials {
    object EnvVars {
      val AzureApplicationCredentials = "AZURE_APPLICATION_CREDENTIALS"
    }

    val DefaultOAuth2Scopes: Array[String] =
      Array(".default")

    private[AzureCloudCredentials] val AzureTokenRefreshMinutes = 5

    def fromJson(jv: JValue): AzureCloudCredentials =
      AzureClientSecretCredentials(jv, DefaultOAuth2Scopes)

    def apply(keyPath: Option[Path], env: Map[String, String] = sys.env): AzureCloudCredentials =
      keyPath.orElse(env.get(AzureApplicationCredentials).map(Path.of(_))) match {
        case Some(path) =>
          using(Files.newInputStream(path)) { in =>
            AzureClientSecretCredentials(JsonMethods.parse(in), DefaultOAuth2Scopes)
          }
        case None =>
          AzureDefaultCredentials(DefaultOAuth2Scopes)
      }
  }

  private case class AzureDefaultCredentials(scopes: IndexedSeq[String])
      extends AzureCloudCredentials {
    @transient override lazy val value: TokenCredential =
      new DefaultAzureCredentialBuilder().build()

    override def scoped(scopes: Array[String]): AzureDefaultCredentials =
      copy(scopes)
  }

  private case class AzureClientSecretCredentials(secret: JValue, scopes: IndexedSeq[String])
      extends AzureCloudCredentials {
    @transient override lazy val value: TokenCredential =
      new ClientSecretCredentialBuilder()
        .clientId((secret \ "appId").extract[String])
        .clientSecret((secret \ "password").extract[String])
        .tenantId((secret \ "tenant").extract[String])
        .build()

    override def scoped(scopes: Array[String]): AzureClientSecretCredentials =
      copy(scopes = scopes)
  }
}
