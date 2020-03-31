package is.hail.backend.service

import java.nio.file.{FileSystems, Files}

import com.google.cloud.storage.Storage.BlobListOption
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.ir.ExecuteContext
import is.hail.io.bgen.IndexBgen
import is.hail.io.fs.GoogleStorageFS
import is.hail.utils._
import py4j.GatewayServer

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

object ServiceBackendGateway {
  def main(args: Array[String]): Unit = {
    val gatewayServer = new GatewayServer();
    gatewayServer.start();
    System.out.println("Gateway Server Started");
  }
}

object ServiceBackend {
  def main(args: Array[String]): Unit = {
    val p = FileSystems.getDefault.getPath("/home/cotton/cseed-gsa-key.json")
    val fs = new GoogleStorageFS(new String(Files.readAllBytes(p)))

    val storage = fs.storage

    for (b <- storage.list("hail-cseed-k0ox4", BlobListOption.prefix("x")).getValues.asScala) {
      println(b.getName)
      println(b.getOwner)
    }


    /*
    val statuses = fs.glob("gs://hail-cseed-k0ox4/h*")
    println(statuses(0).getPath)
     */

    /*
    val is = fs.openNoCompression("gs://hail-cseed-k0ox4/hello.txt")
    val b = new Array[Byte](1024)
    var i = 0
    while (i < 10) {
      val n = is.read(b)
      println(n)
      if (n > 0) {
        println(
          new String(b, 0, n))
      }
      i += 1
    } */

    // println(s"content $content")

    /*
    val storage = StorageOptions.newBuilder()
      .setCredentials(
        ServiceAccountCredentials.fromStream(new FileInputStream("/home/cotton/cseed-gsa-key.json")))
      .build()
      .getService

    val bucket = storage.get("hail-cseed-k0ox4")

    val blob = bucket.get("hello.txt")

    val content = new String(blob.getContent())

    println(s"content $content")
     */
  }

  def apply(): ServiceBackend = new ServiceBackend()
}

case class User(
  username: String,
  fs: GoogleStorageFS)

class ServiceBackend() extends Backend {
  private[this] val users = mutable.Map[String, User]()

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex[T: ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    collection.zipWithIndex.map { case (t, i) => f(t, i) }
  }

  def stop(): Unit = ()

  def pyAddUser(username: String, serviceAccountKey: String): Unit = {
    require(!users.contains(username))
    users += username -> User(username, new GoogleStorageFS(serviceAccountKey))
  }

  def pyRemoveUser(username: String): Unit = {
    require(users.contains(username))
    users -= username
  }

  def pyIndexBgen(
    username: String,
    files: java.util.List[String],
    indexFileMap: java.util.Map[String, String],
    rg: String,
    contigRecoding: java.util.Map[String, String],
    skipInvalidLoci: Boolean) {
    val user = users(username)
    ExecuteContext.scoped(this, user.fs) { ctx =>
      IndexBgen(ctx,
        if (files != null)
          files.asScala.toArray
      else
          null,
        if (indexFileMap != null)
          indexFileMap.asScala.toMap
        else
          null,
        Some(rg),
        if (contigRecoding != null)
          contigRecoding.asScala.toMap
        else
          null,
        skipInvalidLoci)
    }
    info(s"Number of BGEN files indexed: ${ files.size() }")
  }
}
