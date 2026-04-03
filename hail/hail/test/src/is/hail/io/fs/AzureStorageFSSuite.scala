package is.hail.io.fs

import is.hail.services.oauth2.AzureCloudCredentials

class AzureStorageFSSuite extends FSSuite {
  override def beforeAll(): Unit = {
    super.beforeAll()
    assume(
      System.getenv("HAIL_CLOUD") == "azure" &&
        root != null &&
        fsResourcesRoot != null,
      "not in Azure",
    )
  }

  override lazy val fs: FS =
    new AzureStorageFS(
      AzureCloudCredentials(None)
        .scoped(AzureStorageFS.RequiredOAuthScopes)
    )

  test("MakeQualified") {
    val qualifiedFileName = "https://account.blob.core.windows.net/container/path"
    assertEquals(fs.makeQualified(qualifiedFileName), qualifiedFileName)

    val unqualifiedFileName = "https://account/container/path"
    intercept[IllegalArgumentException] {
      fs.makeQualified(unqualifiedFileName)
    }
  }
}
