package is.hail.io.fs

import is.hail.services.oauth2.GoogleCloudCredentials

class GoogleStorageFSSuite extends FSSuite {
  override def beforeAll(): Unit = {
    super.beforeAll()
    assume(
      System.getenv("HAIL_CLOUD") == "gcp" &&
        root != null &&
        fsResourcesRoot != null,
      "not in GCP",
    )
  }

  override lazy val fs: FS =
    new GoogleStorageFS(
      GoogleCloudCredentials(None)
        .scoped(GoogleStorageFS.RequiredOAuthScopes),
      None,
    )

  test("MakeQualified") {
    val qualifiedFileName = "gs://bucket/path"
    assertEquals(fs.makeQualified(qualifiedFileName), qualifiedFileName)

    intercept[IllegalArgumentException] {
      fs.makeQualified("not-gs://bucket/path")
    }
  }
}
