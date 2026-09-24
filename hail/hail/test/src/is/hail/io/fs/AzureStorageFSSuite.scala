package is.hail.io.fs

import is.hail.TestUtils._
import is.hail.services.oauth2.AzureCloudCredentials

import org.junit.jupiter.api.{BeforeAll, Test}
import org.junit.jupiter.api.Assumptions.assumeTrue

class AzureStorageFSSuite extends FSSuite {
  @BeforeAll
  def beforeclass(): Unit =
    assumeTrue(
      System.getenv("HAIL_CLOUD") == "azure" &&
        root != null &&
        fsResourcesRoot != null,
      "not in Azure",
    )

  override lazy val fs: FS =
    new AzureStorageFS(
      AzureCloudCredentials(None)
        .scoped(AzureStorageFS.RequiredOAuthScopes)
    )

  @Test def testMakeQualified(): Unit = {
    val qualifiedFileName = "https://account.blob.core.windows.net/container/path"
    assertEq(fs.makeQualified(qualifiedFileName), qualifiedFileName)

    val unqualifiedFileName = "https://account/container/path"
    intercept[IllegalArgumentException] {
      fs.makeQualified(unqualifiedFileName)
    }: Unit
  }
}
