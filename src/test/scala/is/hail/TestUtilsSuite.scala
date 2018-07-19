package is.hail

import java.io.File
import java.lang.reflect.Modifier
import java.net.URI

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.utils.ArrayBuilder
import org.testng.annotations.{DataProvider, Test}

class TestUtilsSuite extends SparkSuite {

  def getClasses(packageName: String): Array[Class[_]] = {
    val classLoader = Thread.currentThread().getContextClassLoader()
    assert(classLoader != null)
    val path = packageName.replace(".", "/")
    val resources = classLoader.getResources(path)
    val dirs = new ArrayBuilder[File]()
    while (resources.hasMoreElements) {
      val resource = resources.nextElement()
      val uri = new URI(resource.toString)
      dirs += new File(uri.getPath())
    }
    val classes = new ArrayBuilder[Class[_]]()
    for (directory <- dirs.result()) {
      classes ++= findClasses(directory, packageName)
    }
    classes.result()
  }

  def findClasses(dir: File, packageName: String): Array[Class[_]] = {
    val classes = new ArrayBuilder[Class[_]]()
    if (!dir.exists())
      return classes.result()

    val files = dir.listFiles()
    assert(files != null)
    for (file <- files) {
      if (file.isDirectory)
        classes ++= findClasses(file, packageName + "." + file.getName)
      else if (file.getName.endsWith(".class")) {
        classes += Class.forName(packageName + "." + file.getName.substring(0, file.getName.length - 6))
      }
    }
    classes.result()
  }

  // Fixes Issue #3951 -- DataProviders not failing tests
  // Comment in this blog post: http://rolf-engelhard.de/2011/10/fail-instead-of-skip-a-test-when-testngs-dataprovider-throws-an-exception/
  @Test def testDataProviders() {
    var i = 0
    for (testClass <- getClasses("is.hail")) {
      if (!Modifier.isAbstract(testClass.getModifiers)) {
        for (method <- testClass.getMethods()) {
          if (method.isAnnotationPresent(classOf[DataProvider])) {
            method.invoke(testClass.newInstance())
            i += 1
          }
        }
      }
    }
    assert(i != 0)
  }

  @Test def matrixEqualityTest() {
    val M = DenseMatrix((1d, 0d), (0d, 1d))
    val M1 = DenseMatrix((1d, 0d), (0d, 1.0001d))
    val V = DenseVector(0d, 1d)
    val V1 = DenseVector(0d, 0.5d)

    TestUtils.assertMatrixEqualityDouble(M, DenseMatrix.eye(2))
    TestUtils.assertMatrixEqualityDouble(M, M1, 0.001)
    TestUtils.assertVectorEqualityDouble(V, 2d * V1)

    intercept[Exception](TestUtils.assertVectorEqualityDouble(V, V1))
    intercept[Exception](TestUtils.assertMatrixEqualityDouble(M, M1))
  }

  @Test def constantVectorTest() {
    assert(TestUtils.isConstant(DenseVector()))
    assert(TestUtils.isConstant(DenseVector(0)))
    assert(TestUtils.isConstant(DenseVector(0, 0)))
    assert(TestUtils.isConstant(DenseVector(0, 0, 0)))
    assert(!TestUtils.isConstant(DenseVector(0, 1)))
    assert(!TestUtils.isConstant(DenseVector(0, 0, 1)))
  }

  @Test def removeConstantColsTest(): Unit = {
    val M = DenseMatrix((0, 0, 1, 1, 0),
                        (0, 1, 0, 1, 1))

    val M1 = DenseMatrix((0, 1, 0),
                         (1, 0, 1))

    assert(TestUtils.removeConstantCols(M) == M1)
  }
}
