package is.hail.io.fs

import is.hail.HailSuite
import is.hail.backend.ExecuteContext
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.{HailSuite, TestUtils}
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.apache.hadoop.fs.FileAlreadyExistsException

import java.io.FileNotFoundException

trait FSSuite extends TestNGSuite {
  val root: String = System.getenv("HAIL_TEST_STORAGE_URI")

  def fsResourcesRoot: String = System.getenv("HAIL_FS_TEST_CLOUD_RESOURCES_URI")

  def tmpdir: String = System.getenv("HAIL_TEST_STORAGE_URI")

  def fs: FS

  /* Structure of src/test/resources/fs:
     /a
     /adir
     /adir/x
     /az
     /dir
     /dir/x
     /zzz
   */
  def r(s: String): String = s"$fsResourcesRoot$s"

  def t(extension: String = null): String = ExecuteContext.createTmpPathNoCleanup(tmpdir, "fs-suite-tmp", extension)

  def pathsRelRoot(root: String, statuses: Array[FileListEntry]): Set[String] = {
    statuses.map { status =>
      var p = status.getPath
      assert(p.startsWith(root), s"$p $root")
      p.drop(root.length)
    }.toSet
  }

  def pathsRelResourcesRoot(statuses: Array[FileListEntry]): Set[String] = pathsRelRoot(fsResourcesRoot, statuses)

  @Test def testExistsOnDirectory(): Unit = {
    assert(fs.exists(r("/dir")))
    assert(fs.exists(r("/dir/")))

    assert(!fs.exists(r("/does_not_exist")))
    assert(!fs.exists(r("/does_not_exist_dir/")))
  }

  @Test def testExistsOnFile(): Unit = {
    assert(fs.exists(r("/a")))

    assert(fs.exists(r("/zzz")))
    assert(!fs.exists(r("/z"))) // prefix
  }

  @Test def testFileStatusOnFile(): Unit = {
    // file
    val f = r("/a")
    val s = fs.fileStatus(f)
    assert(s.getPath == f)
    assert(s.getLen == 12)
  }

  @Test def testFileListEntryOnFile(): Unit = {
    // file
    val f = r("/a")
    val s = fs.fileListEntry(f)
    assert(s.getPath == f)
    assert(s.isFile)
    assert(!s.isDirectory)
    assert(s.getLen == 12)
  }

  @Test def testFileStatusOnDirIsFailure(): Unit = {
    val f = r("/dir")
    TestUtils.interceptException[FileNotFoundException](r("/dir"))(
      fs.fileStatus(r("/dir"))
    )
  }

  @Test def testFileListEntryOnDir(): Unit = {
    // file
    val f = r("/dir")
    val s = fs.fileListEntry(f)
    assert(s.getPath == f)
    assert(!s.isFile)
    assert(s.isDirectory)
  }

  @Test def testFileListEntryOnDirWithSlash(): Unit = {
    // file
    val f = r("/dir/")
    val s = fs.fileListEntry(f)
    assert(s.getPath == f.dropRight(1))
    assert(!s.isFile)
    assert(s.isDirectory)
  }

  @Test def testFileListEntryOnMissingFile(): Unit = {
    try {
      fs.fileListEntry(r("/does_not_exist"))
    } catch {
      case _: FileNotFoundException =>
        return
    }
    assert(false)
  }

  @Test def testFileListEntryRoot(): Unit = {
    val s = fs.fileListEntry(root)
    assert(s.getPath == root)
  }

  @Test def testFileListEntryRootWithSlash(): Unit = {
    if (root.endsWith("/"))
      return

    val s = fs.fileListEntry(s"$root/")
    assert(s.getPath == root)
  }

  @Test def testDeleteRecursive(): Unit = {
    val d = t()
    fs.mkDir(d)
    fs.touch(s"$d/x")
    fs.touch(s"$d/y")
    fs.mkDir(s"$d/subdir")
    fs.touch(s"$d/subdir/z")
    fs.mkDir(s"$d/dir2")
    fs.touch(s"$d/dir2/a")
    fs.touch(s"$d/dir2/b")
    fs.mkDir(s"$d/subdir/another_list")
    fs.touch(s"$d/subdir/another_list/file1")

    assert(fs.exists(s"$d/subdir/z"))

    fs.delete(d, recursive = true)

    assert(!fs.exists(s"$d"))
    assert(!fs.exists(s"$d/subdir/z"))
  }

  @Test def testDeleteFileDoesntExist(): Unit = {
    val d = t()
    fs.mkDir(d)
    fs.delete(s"$d/foo", recursive = false)
    fs.delete(s"$d/foo", recursive = true)
  }

  @Test def testListDirectory(): Unit = {
    val statuses = fs.listDirectory(r(""))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az", "/dir", "/zzz"))
  }

  @Test def testListDirectoryWithSlash(): Unit = {
    val statuses = fs.listDirectory(r("/"))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az", "/dir", "/zzz"))
  }

  @Test def testGlobOnDir(): Unit = {
    val statuses = fs.glob(r(""))
    assert(pathsRelResourcesRoot(statuses) == Set(""))
  }

  @Test def testGlobMissingFile(): Unit = {
    val statuses = fs.glob(r("/does_not_exist_dir/does_not_exist"))
    assert(pathsRelResourcesRoot(statuses) == Set())
  }

  @Test def testGlobFilename(): Unit = {
    val statuses = fs.glob(r("/a*"))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/a", "/adir", "/az")}")
  }

  @Test def testGlobFilenameMatchSingleCharacter(): Unit = {
    val statuses = fs.glob(r("/a?"))
    assert(pathsRelResourcesRoot(statuses) == Set("/az"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/az")}")
  }

  @Test def testGlobFilenameMatchSingleCharacterInMiddleOfName(): Unit = {
    val statuses = fs.glob(r("/a?ir"))
    assert(pathsRelResourcesRoot(statuses) == Set("/adir"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/adir")}")
  }

  @Test def testGlobDirnameMatchSingleCharacterInMiddleOfName(): Unit = {
    val statuses = fs.glob(r("/a?ir/x"))
    assert(pathsRelResourcesRoot(statuses) == Set("/adir/x"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/adir/x")}")
  }

  @Test def testGlobMatchDir(): Unit = {
    val statuses = fs.glob(r("/*dir/x"))
    assert(pathsRelResourcesRoot(statuses) == Set("/adir/x", "/dir/x"),
      s"${statuses} ${pathsRelResourcesRoot(statuses)} ${Set("/adir/x", "/dir/x")}")
  }

  @Test def testGlobRoot(): Unit = {
    val statuses = fs.glob(root)
    // empty with respect to root (self)
    assert(pathsRelRoot(root, statuses) == Set(""))
  }

  @Test def testFileEndingWithPeriod: Unit = {
    val f = fs.makeQualified(t())
    fs.touch(f + "/foo.")
    val statuses = fs.listDirectory(f)
    assert(statuses.length == 1, statuses)
    val status = statuses(0)
    if (this.isInstanceOf[AzureStorageFSSuite]) {
      // https://github.com/Azure/azure-sdk-for-java/issues/36674
      assert(status.getPath == f + "/foo")
    } else {
      assert(status.getPath == f + "/foo.")
    }
  }

  @Test def testGlobRootWithSlash(): Unit = {
    if (root.endsWith("/"))
      return

    val statuses = fs.glob(s"$root/")
    assert(pathsRelRoot(root, statuses) == Set(""))
  }

  @Test def testWriteRead(): Unit = {
    val s = "this is a test string"
    val f = t()

    using(fs.createNoCompression(f)) { os =>
      val b = s.getBytes
      os.write(b)
    }

    assert(fs.exists(f))

    using(fs.openNoCompression(f)) { is =>
      val read = new String(IOUtils.toByteArray(is))
      assert(read == s)
    }

    fs.delete(f, false)

    assert(!fs.exists(f))
  }

  @Test def testWriteReadCompressed(): Unit = {
    val s = "this is a test string"
    val f = t(extension = ".bgz")

    using(fs.create(f)) { os =>
      val b = s.getBytes
      os.write(b)
    }

    assert(fs.exists(f))

    using(fs.open(f)) { is =>
      val read = new String(IOUtils.toByteArray(is))
      assert(read == s)
    }

    fs.delete(f, false)

    assert(!fs.exists(f))
  }

  @Test def testWritePreexisting(): Unit = {
    val s1 = "first"
    val s2 = "second"
    val f = t()

    using(fs.create(f)) { _.write(s1.getBytes) }
    assert(fs.exists(f))
    using(fs.open(f)) { is =>
      val read = new String(IOUtils.toByteArray(is))
      assert(read == s1)
    }

    using(fs.create(f)) { _.write(s2.getBytes) }
    assert(fs.exists(f))
    using(fs.open(f)) { is =>
      val read = new String(IOUtils.toByteArray(is))
      assert(read == s2)
    }
  }

  @Test def testGetCodecExtension(): Unit = {
    assert(fs.getCodecExtension("foo.vcf.bgz") == ".bgz")
  }

  @Test def testStripCodecExtension(): Unit = {
    assert(fs.stripCodecExtension("foo.vcf.bgz") == "foo.vcf")
  }

  @Test def testReadWriteBytes(): Unit = {
    val f = t()

    using(fs.create(f)) { os =>
      os.write(1)
      os.write(127)
      os.write(255)
    }

    assert(fs.exists(f))

    using(fs.open(f)) { is =>
      assert(is.read() == 1)
      assert(is.read() == 127)
      assert(is.read() == 255)
    }

    fs.delete(f, false)

    assert(!fs.exists(f))
  }

  @Test def testReadWriteBytesLargerThanBuffer(): Unit = {
    val f = t()

    val numWrites = 1000000
    using(fs.create(f)) { os =>
      os.write(1)
      os.write(127)
      os.write(255)

      var i = 0
      while (i < numWrites) {
        os.write(i)
        i = i + 1
      }
    }

    assert(fs.exists(f))

    using(fs.open(f)) { is =>
      assert(is.read() == 1)
      assert(is.read() == 127)
      assert(is.read() == 255)

      var i = 0
      while (i < numWrites) {
        val readFromIs = is.read()
        assert(readFromIs == (i & 0xff), s"${i} ${i & 0xff} ${readFromIs}")
        i = i + 1
      }
    }

    fs.delete(f, false)

    assert(!fs.exists(f))
  }

  @Test def testDropTrailingSlash(): Unit = {
    assert(dropTrailingSlash("") == "")
    assert(dropTrailingSlash("/foo/bar") == "/foo/bar")
    assert(dropTrailingSlash("foo/bar/") == "foo/bar")
    assert(dropTrailingSlash("/foo///") == "/foo")
    assert(dropTrailingSlash("///") == "")
  }

  @Test def testSeekMoreThanMaxInt(): Unit = {
    val f = t()
    using (fs.create(f)) { os =>
      val eight_mib = 8 * 1024 * 1024
      val arr = Array.fill(eight_mib){0.toByte}
      var i = 0
      // 256 * 8MiB = 2GiB
      while (i < 256) {
        os.write(arr, 0, eight_mib)
        i = i + 1
      }
      os.write(10)
      os.write(20)
      os.write(30)
    }

    assert(fs.exists(f))

    using(fs.open(f, fs.getCodecFromPath(f))) { is =>
      is match {
        case base: Seekable => base.seek(Int.MaxValue + 2.toLong)
        case base: org.apache.hadoop.fs.Seekable => base.seek(Int.MaxValue + 2.toLong)
      }
      assert(is.read() == 20)
      assert(is.read() == 30)
    }

    fs.delete(f, false)
    assert(!fs.exists(f))
  }

  @Test def testSeekAndReadStraddlingBufferSize(): Unit = {
    val data = Array.tabulate(251)(_.toByte)
    val f = t()
    using(fs.create(f)) { os =>
      var i = 0
      // 66058 replicates are 8MB of data
      while (i < 70000) {
        os.write(data)
        i += 1
      }
    }

    using(fs.openNoCompression(f)) { is =>

      is.seek(251)
      assert(is.read() == 0)
      assert(is.read() == 1)

      val seekPos = 8 * 1024 * 1024 - 512
      is.seek(8 * 1024 * 1024 - 512)
      assert(is.getPosition == seekPos)
      val toRead = new Array[Byte](512)
      is.readFully(toRead)

      (0 until toRead.length).foreach { i =>
        assert(toRead(i) == ((seekPos + i) % 251).toByte)
      }
    }
  }

  @Test def largeDirectoryOperations(): Unit = {
    val prefix = s"$tmpdir/fs-suite/delete-many-files/${ java.util.UUID.randomUUID() }"
    for (i <- 0 until 2000) {
      fs.touch(s"$prefix/$i.suffix")
    }

    assert(fs.listDirectory(prefix).size == 2000)
    assert(fs.glob(prefix + "/" + "*.suffix").size == 2000)

    assert(fs.exists(prefix))
    fs.delete(prefix, recursive = true)
    if (fs.exists(prefix)) {
      // NB: TestNGSuite.assert does not have a lazy message argument so we must use an if to protect this list
      //
      // see: https://www.scalatest.org/scaladoc/1.7.2/org/scalatest/testng/TestNGSuite.html
      assert(false, s"files not deleted:\n${ fs.listDirectory(prefix).map(_.getPath).mkString("\n") }")
    }
  }

  @Test def testSeekAfterEOF(): Unit = {
    val prefix = s"$tmpdir/fs-suite/delete-many-files/${ java.util.UUID.randomUUID() }"
    val p = s"$prefix/seek_file"
    using(fs.createCachedNoCompression(p)) { os =>
      os.write(1.toByte)
      os.write(2.toByte)
      os.write(3.toByte)
      os.write(4.toByte)
    }

    using(fs.openNoCompression(p)) { is =>
      assert(is.read() == 1.toByte)
      is.seek(3)
      assert(is.read() == 4.toByte)
      assert(is.read() == (-1).toByte)
      is.seek(0)
      assert(is.read() == 1.toByte)
    }
  }

  @Test def fileAndDirectoryIsError(): Unit = {
    val d = t()
    fs.mkDir(d)
    fs.touch(s"$d/x/file")
    try {
      fs.touch(s"$d/x")
      fs.fileListEntry(s"$d/x")
      assert(false)
    } catch {
      // Hadoop, in particular, errors when you touch an object whose name is a prefix of another object.
      case exc: FileAndDirectoryException if exc.getMessage() == s"$d/x appears as both file $d/x and directory $d/x/." =>
      case exc: FileNotFoundException if exc.getMessage() == s"$d/x (Is a directory)" =>
    }
  }

  @Test def testETag(): Unit = {
    val etag = fs.eTag(s"$fsResourcesRoot/a")
    if (fs.parseUrl(fsResourcesRoot).toString.startsWith("file:")) {
      // only the local file system should lack etags.
      assert(etag.isEmpty)
    } else {
      assert(etag.nonEmpty)
    }
  }

  @Test def fileAndDirectoryIsErrorEvenIfPrefixedFileIsNotLexicographicallyFirst(): Unit = {
    val d = t()
    fs.mkDir(d)
    fs.touch(s"$d/x")
    // fs.touch(s"$d/x ") // Hail does not support spaces in path names
    fs.touch(s"$d/x!")
    fs.touch(s"$d/x${'"'}")
    fs.touch(s"$d/x#")
    fs.touch(s"$d/x$$")
    // fs.touch(s"$d/x%") // Azure dislikes %'s
    // java.lang.IllegalArgumentException: URLDecoder: Incomplete trailing escape (%) pattern
    //     at java.net.URLDecoder.decode(URLDecoder.java:187)
    //     at is.hail.shadedazure.com.azure.storage.common.Utility.decode(Utility.java:88)
    //     at is.hail.shadedazure.com.azure.storage.common.Utility.urlDecode(Utility.java:55)
    //     at is.hail.shadedazure.com.azure.storage.blob.specialized.BlobAsyncClientBase.<init>(BlobAsyncClientBase.java:238)
    //     at is.hail.shadedazure.com.azure.storage.blob.specialized.BlobAsyncClientBase.<init>(BlobAsyncClientBase.java:202)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobAsyncClient.<init>(BlobAsyncClient.java:154)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerAsyncClient.getBlobAsyncClient(BlobContainerAsyncClient.java:194)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerAsyncClient.getBlobAsyncClient(BlobContainerAsyncClient.java:172)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerClient.getBlobClient(BlobContainerClient.java:98)
    //     at is.hail.io.fs.AzureStorageFS.$anonfun$getBlobClient$1(AzureStorageFS.scala:255)
    fs.touch(s"$d/x&")
    fs.touch(s"$d/x'")
    fs.touch(s"$d/x)")
    fs.touch(s"$d/x(")
    fs.touch(s"$d/x*")
    fs.touch(s"$d/x+")
    fs.touch(s"$d/x,")
    fs.touch(s"$d/x-")
    // fs.touch(s"$d/x.") // https://github.com/Azure/azure-sdk-for-java/issues/36674
    try {
      fs.touch(s"$d/x/file")
      fs.fileListEntry(s"$d/x")
      assert(false)
    } catch {
      // Hadoop, in particular, errors when you touch an object whose name is a prefix of another object.
      case exc: FileAndDirectoryException if exc.getMessage() == s"$d/x appears as both file $d/x and directory $d/x/." =>
      case exc: FileAlreadyExistsException if exc.getMessage() == s"Destination exists and is not a directory: $d/x" =>
    }
  }

  @Test def fileListEntrySeesDirectoryEvenIfPrefixedFileIsNotLexicographicallyFirst(): Unit = {
    val d = t()
    fs.mkDir(d)
    // fs.touch(s"$d/x ") // Hail does not support spaces in path names
    fs.touch(s"$d/x!")
    fs.touch(s"$d/x${'"'}")
    fs.touch(s"$d/x#")
    fs.touch(s"$d/x$$")
    // fs.touch(s"$d/x%") // Azure dislikes %'s
    // java.lang.IllegalArgumentException: URLDecoder: Incomplete trailing escape (%) pattern
    //     at java.net.URLDecoder.decode(URLDecoder.java:187)
    //     at is.hail.shadedazure.com.azure.storage.common.Utility.decode(Utility.java:88)
    //     at is.hail.shadedazure.com.azure.storage.common.Utility.urlDecode(Utility.java:55)
    //     at is.hail.shadedazure.com.azure.storage.blob.specialized.BlobAsyncClientBase.<init>(BlobAsyncClientBase.java:238)
    //     at is.hail.shadedazure.com.azure.storage.blob.specialized.BlobAsyncClientBase.<init>(BlobAsyncClientBase.java:202)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobAsyncClient.<init>(BlobAsyncClient.java:154)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerAsyncClient.getBlobAsyncClient(BlobContainerAsyncClient.java:194)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerAsyncClient.getBlobAsyncClient(BlobContainerAsyncClient.java:172)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerClient.getBlobClient(BlobContainerClient.java:98)
    //     at is.hail.io.fs.AzureStorageFS.$anonfun$getBlobClient$1(AzureStorageFS.scala:255)
    fs.touch(s"$d/x&")
    fs.touch(s"$d/x'")
    fs.touch(s"$d/x)")
    fs.touch(s"$d/x(")
    fs.touch(s"$d/x*")
    fs.touch(s"$d/x+")
    fs.touch(s"$d/x,")
    fs.touch(s"$d/x-")
    // fs.touch(s"$d/x.") // https://github.com/Azure/azure-sdk-for-java/issues/36674
    fs.touch(s"$d/x/file")

    val fle = fs.fileListEntry(s"$d/x")
    assert(fle.isDirectory)
    assert(!fle.isFile)
  }

  @Test def fileListEntrySeesFileEvenWithPeersPreceedingThePositionOfANonPresentDirectoryEntry(): Unit = {
    val d = t()
    fs.mkDir(d)
    fs.touch(s"$d/x")
    // fs.touch(s"$d/x ") // Hail does not support spaces in path names
    fs.touch(s"$d/x!")
    fs.touch(s"$d/x${'"'}")
    fs.touch(s"$d/x#")
    fs.touch(s"$d/x$$")
    // fs.touch(s"$d/x%") // Azure dislikes %'s
    // java.lang.IllegalArgumentException: URLDecoder: Incomplete trailing escape (%) pattern
    //     at java.net.URLDecoder.decode(URLDecoder.java:187)
    //     at is.hail.shadedazure.com.azure.storage.common.Utility.decode(Utility.java:88)
    //     at is.hail.shadedazure.com.azure.storage.common.Utility.urlDecode(Utility.java:55)
    //     at is.hail.shadedazure.com.azure.storage.blob.specialized.BlobAsyncClientBase.<init>(BlobAsyncClientBase.java:238)
    //     at is.hail.shadedazure.com.azure.storage.blob.specialized.BlobAsyncClientBase.<init>(BlobAsyncClientBase.java:202)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobAsyncClient.<init>(BlobAsyncClient.java:154)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerAsyncClient.getBlobAsyncClient(BlobContainerAsyncClient.java:194)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerAsyncClient.getBlobAsyncClient(BlobContainerAsyncClient.java:172)
    //     at is.hail.shadedazure.com.azure.storage.blob.BlobContainerClient.getBlobClient(BlobContainerClient.java:98)
    //     at is.hail.io.fs.AzureStorageFS.$anonfun$getBlobClient$1(AzureStorageFS.scala:255)
    fs.touch(s"$d/x&")
    fs.touch(s"$d/x'")
    fs.touch(s"$d/x)")
    fs.touch(s"$d/x(")
    fs.touch(s"$d/x*")
    fs.touch(s"$d/x+")
    fs.touch(s"$d/x,")
    fs.touch(s"$d/x-")
    // fs.touch(s"$d/x.") // https://github.com/Azure/azure-sdk-for-java/issues/36674

    val fle = fs.fileListEntry(s"$d/x")
    assert(!fle.isDirectory)
    assert(fle.isFile)
    assert(fle.getPath == fs.parseUrl(s"$d/x").toString)
  }
}

class HadoopFSSuite extends HailSuite with FSSuite {
  override val root: String = "file:/"

  override lazy val fsResourcesRoot: String = "file:" + new java.io.File("./src/test/resources/fs").getCanonicalPath

  override lazy val tmpdir: String = ctx.tmpdir
}
