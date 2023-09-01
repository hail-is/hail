package is.hail.fs

import java.io.FileNotFoundException
import is.hail.fs.azure.AzureStorageFSSuite
import is.hail.HailSuite
import is.hail.backend.ExecuteContext
import is.hail.io.fs.FSUtil.dropTrailingSlash
import is.hail.io.fs.{FS, FileStatus, GoogleStorageFS, Seekable}
import is.hail.utils._
import org.apache.commons.codec.binary.Hex
import org.apache.commons.io.IOUtils
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

trait FSSuite extends TestNGSuite {
  def root: String

  def fsResourcesRoot: String

  def fs: FS

  def tmpdir: String

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

  def pathsRelRoot(root: String, statuses: Array[FileStatus]): Set[String] = {
    statuses.map { status =>
      var p = status.getPath
      assert(p.startsWith(root), s"$p $root")
      p.drop(root.length)
    }.toSet
  }

  def pathsRelResourcesRoot(statuses: Array[FileStatus]): Set[String] = pathsRelRoot(fsResourcesRoot, statuses)

  @Test def testExists(): Unit = {
    assert(fs.exists(r("/a")))

    assert(fs.exists(r("/zzz")))
    assert(!fs.exists(r("/z"))) // prefix

    assert(fs.exists(r("/dir")))
    assert(fs.exists(r("/dir/")))

    assert(!fs.exists(r("/does_not_exist")))
    assert(!fs.exists(r("/does_not_exist_dir/")))
  }

  @Test def testFileStatusOnFile(): Unit = {
    // file
    val f = r("/a")
    val s = fs.fileStatus(f)
    assert(s.getPath == f)
    assert(s.isFile)
    assert(!s.isDirectory)
    assert(s.getLen == 12)
  }

  @Test def testFileStatusOnDir(): Unit = {
    // file
    val f = r("/dir")
    val s = fs.fileStatus(f)
    assert(s.getPath == f)
    assert(!s.isFile)
    assert(s.isDirectory)
  }

  @Test def testFileStatusOnDirWithSlash(): Unit = {
    // file
    val f = r("/dir/")
    val s = fs.fileStatus(f)
    assert(s.getPath == f.dropRight(1))
    assert(!s.isFile)
    assert(s.isDirectory)
  }

  @Test def testFileStatusOnMissingFile(): Unit = {
    try {
      fs.fileStatus(r("/does_not_exist"))
    } catch {
      case _: FileNotFoundException =>
        return
    }
    assert(false)
  }

  @Test def testFileStatusRoot(): Unit = {
    val s = fs.fileStatus(root)
    assert(s.getPath == root)
  }

  @Test def testFileStatusRootWithSlash(): Unit = {
    if (root.endsWith("/"))
      return

    val s = fs.fileStatus(s"$root/")
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

  @Test def testListStatusDir(): Unit = {
    val statuses = fs.listStatus(r(""))
    assert(pathsRelResourcesRoot(statuses) == Set("/a", "/adir", "/az", "/dir", "/zzz"))
  }

  @Test def testListStatusDirWithSlash(): Unit = {
    val statuses = fs.listStatus(r("/"))
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
}

class HadoopFSSuite extends HailSuite with FSSuite {
  val root: String = "file:/"

  lazy val fsResourcesRoot: String = "file:" + new java.io.File("./src/test/resources/fs").getCanonicalPath

  lazy val tmpdir: String = ctx.tmpdir
}
